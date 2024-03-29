import json

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer
from src.utils import json_parser_from_chat_response, DATA_PATH

DPO_TRAIN = str(DATA_PATH / "dpo_train.json") 
DPO_EVAL = str(DATA_PATH / "dpo_eval.json") 
PROJECT = "mistral-dpo-from-zero-shot"
BASE_MODEL_NAME = "mistral-7b-instruct-v0.2"
RUN_NAME = BASE_MODEL_NAME + "-" + PROJECT
OUTPUT_DIR = str(DATA_PATH / RUN_NAME) 


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


train_dataset = Dataset.from_list(json.load(open(DPO_TRAIN)))
eval_dataset = Dataset.from_list(json.load(open(DPO_EVAL)))


base_model = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token_id =  tokenizer.unk_token_id
tokenizer.padding_side = 'left'
tokenizer.truncation_side = 'left'


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

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    peft_config=lora_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    max_length=2048,
    max_prompt_length=2048,
    beta=0.1,
    loss_type="sigmoid",
    args=transformers.TrainingArguments(
        output_dir=OUTPUT_DIR,
        warmup_steps=0,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
        max_steps=500,
        learning_rate=5e-5,
        optim="adamw_torch_fused",
        lr_scheduler_type="cosine",
        logging_steps=10,
        logging_dir="./logs",
        bf16=False,
        save_strategy="steps",
        save_steps=10,
        evaluation_strategy="steps",
        eval_steps=10,
        do_eval=True,
        save_total_limit=5,   
    )
)

trainer.train(resume_from_checkpoint=False)