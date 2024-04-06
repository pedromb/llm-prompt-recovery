"""
Merge the finetuned lora adapters with the main model and save the weights
"""
import torch
from unsloth import FastLanguageModel
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

PEFT_MODEL = "/home/llm-prompt-recovery/data/weights/mistral-7b-instruct-v0.2-finetuned-subject-prompt/checkpoint-1250"
OUTPUT_PATH = "/home/llm-prompt-recovery/data/weights/mistral-7b-finetuned-subject-prompt"


model, tokenizer = FastLanguageModel.from_pretrained(
    PEFT_MODEL,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

model.save_pretrained_merged(OUTPUT_PATH, tokenizer, save_method = "merged_4bit_forced")
