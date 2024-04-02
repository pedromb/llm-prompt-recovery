import json

import pandas as pd
import transformers
from tqdm.auto import tqdm
from datasets import Dataset
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from src.utils import json_parser_from_chat_response, DATA_PATH
from unsloth import FastLanguageModel

DATASET =  str(DATA_PATH / "new_data_for_training.csv" )
PROJECT = "mistral-7b-lora-finetuned-unsloth"
BASE_MODEL_NAME = "mistral-7b-instruct-v0.2"
RUN_NAME = BASE_MODEL_NAME + "-" + PROJECT
RESUME_FROM_CHECKPOINT = "/home/llm-prompt-recovery/data/weights/mistral-7b-instruct-v0.2-mistral-7b-lora-finetuned-unsloth/checkpoint-675"
OUTPUT_DIR = str(DATA_PATH  / "weights" / RUN_NAME)

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

def load_dataset(df, tokenizer, max_size = None):
    if max_size is not None and max_size < len(df):
        df = df.sample(n=max_size)
    df = df.sample(frac=1)
    all_data = [create_dataset_item(row) for _, row in df.iterrows()]
    final_data = []
    for entry in tqdm(all_data, desc="Filtering data..."):
        text = format_single(entry["input"], entry["output"])
        _input_ids = tokenizer.batch_encode_plus([text])["input_ids"][0]
        total_len = len(_input_ids)
        if total_len < 2048:
            final_data.append(entry)
    return Dataset.from_list(final_data)

def format_single(input, output):
    return f"""<s> [INST]
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
        Input: {input} [/INST] {output} </s>
        """
    
def formatting_func(example):
    return [
        format_single(example['input'][i], example['output'][i]) for i in range(len(example['input']))
    ]

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


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

tokenizer.pad_token_id =  tokenizer.unk_token_id
tokenizer.padding_side = 'right'


model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = True,
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)



data = pd.read_csv(DATASET)
cluster_to_split = json.load(open("/home/llm-prompt-recovery/data/cluster_to_split.json"))
data["split"] = data.cluster.apply(lambda x: cluster_to_split.get(str(x), "train"))
train = data.loc[data.split == "train"]
val = data.loc[data.split == "val"].sample(200)

train_dataset = load_dataset(train, tokenizer)
val_dataset = load_dataset(val, tokenizer)
response_template = "[/INST]"
data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    formatting_func=formatting_func,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    max_seq_length=2048,
    args=transformers.TrainingArguments(
        output_dir=OUTPUT_DIR,
        warmup_steps=0,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=5e-5, # Want a small lr for finetuning
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        logging_steps=25,
        logging_dir="./logs",
        bf16=True,
        save_strategy="steps",
        save_steps=25,
        evaluation_strategy="no",
        save_total_limit=5,
    ),
    data_collator=data_collator,
)

trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)