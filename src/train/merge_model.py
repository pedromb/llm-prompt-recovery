import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PEFT_MODEL = "../../weights/my-model"
OUTPUT_PATH = "../../weights/my-merged-model"


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if "mistral" in BASE_MODEL:
    tokenizer.pad_token = tokenizer.eos_token


compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, 
    device_map="auto", 
    quantization_config=quant_config, 
)
model.config.use_cache = False
model.config.pretraining_tp = 1


model = PeftModel.from_pretrained(model, PEFT_MODEL)

model = model.merge_and_unload()
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)