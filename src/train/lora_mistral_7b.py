import json

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from src.utils import json_parser_from_chat_response

DATASET = "../../data/data.csv" 
PROJECT = "rewrite-prompt-finetuned-v2"
BASE_MODEL_NAME = "mistral-7b-instruct-v0.2"
RUN_NAME = BASE_MODEL_NAME + "-" + PROJECT
OUTPUT_DIR = "../../weights/" + RUN_NAME

def create_dataset_item(df_row):
    json_input = {
        "original_text": df_row.original_text,
        "rewritten_text": df_row.rewritten_text,
    }
    json_output = {"prompt": df_row.rewrite_prompt}
    return {
        "input": json.dumps(json_input),
        "output": json.dumps(json_output),
    }

def load_dataset(df, max_size = None):
    if max_size is not None and max_size < len(df):
        df = df.sample(n=max_size)
    df = df.sample(frac=1)
    all_data = [create_dataset_item(row) for _, row in df.iterrows()]
    return Dataset.from_list(all_data)

def formatting_func(example):
    output_texts = []
    for i in range(len(example['input'])):
        text = f"""<s> [INST]
        I will give you a JSON with following structure:
        {{
            'original_text': 'An original piece of text.'
            'rewritten_text': 'A version of original_text that was rewritten by an LLM according to a specific prompt.'
        }}

        Given the task of understanding how text is rewritten by analyzing the original_text and rewritten_text, your goal is to deduce the specific instructions or prompt that was most likely used to generate the rewritten text from the original text. Consider the changes made in terms of style, tone, structure, and content. Assess whether the rewrite focuses on summarization, paraphrasing, stylistic alteration (e.g., formal to informal), or any specific content changes (e.g., making the text more concise, expanding on ideas, or altering the perspective). Follow this steps:

        1. Read the original_text: Start by thoroughly understanding the content, style, tone, and purpose of the original text. Note any key themes, technical terms, and the overall message.
        2. Analyze the rewritten_text: Examine how the rewritten text compares to the original. Identify what has been changed, added, or omitted. Pay close attention to changes in style (formal, informal), tone (serious, humorous), structure (paragraph order, sentence structure), and any shifts in perspective or emphasis.
        3. Infer the Prompt: Based on your analysis, infer the most likely prompt that guided the rewriting process. Your inference should account for the observed changes in style, tone, structure, and content. Specify the type of task (e.g., summarize, paraphrase, make more accessible to a general audience), any specific directions evident from the changes, and any specific stylistic choice.

        Based on your analysis return the prompt as if you were given the instruction your self like:
        "Rewrite this text..."
        "Transform this ... into ... based on the style of ..."
        
        Make the prompt short and direct using a maximum of 20 words.

        Return your answer using the following JSON structure:
        {{"prompt": "Your best guess for the prompt used"}}
    
            
        Return a valid JSON as output and nothing more.
        
        -----------------------
        Input: {example['input'][i]} [/INST] {example['output'][i]}</s>
        """
        output_texts.append(text)
    return output_texts

def extract_output(text):
    try:
        return json_parser_from_chat_response(text).get("prompt", "Make this text better")
    except:
        return "Make this text better"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

base_model = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id =  tokenizer.unk_token_id
tokenizer.padding_side = 'right'


compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)
model = AutoModelForCausalLM.from_pretrained(
    base_model, 
    device_map="auto", 
    quantization_config=quant_config
)
model.config.use_cache = False
model.config.pretraining_tp = 1
model.config.pad_token_id = tokenizer.pad_token_id

model = prepare_model_for_kbit_training(model)


lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    bias="none",
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
print_trainable_parameters(model)

data = pd.read_csv(DATASET)
train = data.loc[data.split == "train"]
val = data.loc[data.split == "val"]

train_dataset = load_dataset(train)
val_dataset = load_dataset(val)
response_template = "[/INST]"
data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    peft_config=lora_config,
    formatting_func=formatting_func,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=2048,
    args=transformers.TrainingArguments(
        output_dir=OUTPUT_DIR,
        warmup_steps=0,
        per_device_train_batch_size=8,
        max_steps=30000,
        learning_rate=2.5e-4, # Want a small lr for finetuning
        optim="paged_adamw_8bit",
        logging_steps=500,
        logging_dir="./logs",
        bf16=True,
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="steps",
        eval_steps=500,
        do_eval=True,
        save_total_limit=5,
    ),
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=False)