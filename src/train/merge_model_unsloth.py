import torch
from unsloth import FastLanguageModel
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

PEFT_MODEL = "/home/llm-prompt-recovery/data/weights/mistral-7b-instruct-v0.2-mistral-7b-lora-finetuned-unsloth/checkpoint-675"
OUTPUT_PATH = "/home/llm-prompt-recovery/data/weights/mistral-7b-finetuned-lora-unsloth"


model, tokenizer = FastLanguageModel.from_pretrained(
    PEFT_MODEL,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

model.save_pretrained_merged(OUTPUT_PATH, tokenizer, save_method = "merged_4bit_forced")
