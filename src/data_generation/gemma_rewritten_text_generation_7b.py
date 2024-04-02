# Setup the environment

GEMMA_PYTORCH_PATH="/home/llm-prompt-recovery/gemma_pytorch"
BATCH_SIZE = 32
VARIANT = "7b-it-quant" 
MACHINE_TYPE = "cuda" 
WEIGHTS_DIR = "../gemma_weights"
INPUT_SAMPLES_JSON = "prompts_with_original_text.json"
OUTPUT_SAMPLES_JSON = "prompts_with_rewritten_text.json"

import json
import sys
from copy import deepcopy
from pathlib import Path
from tqdm import tqdm

sys.path.append(GEMMA_PYTORCH_PATH)
import contextlib
import os

import torch
from gemma.config import get_config_for_7b
from gemma.model import GemmaForCausalLM


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
  """Sets the default torch dtype to the given dtype."""
  torch.set_default_dtype(dtype)
  yield
  torch.set_default_dtype(torch.float)

# Model Config.
model_config = get_config_for_7b()
model_config.tokenizer = os.path.join(WEIGHTS_DIR, "tokenizer.model")
model_config.quant = "quant" in VARIANT

# Model.
device = torch.device(MACHINE_TYPE)
with _set_default_tensor_type(model_config.get_dtype()):
  model = GemmaForCausalLM(model_config)
  ckpt_path = os.path.join(WEIGHTS_DIR, f'gemma-{VARIANT}.ckpt')
  model.load_weights(ckpt_path)
  model = model.to(device).eval()

def get_prompt(original_text: str, rewrite_prompt: str):
    USER_CHAT_TEMPLATE = "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n"
    initial_prompt = f"""{rewrite_prompt}\n\n{original_text}"""
    prompt = (
        USER_CHAT_TEMPLATE.format(prompt=initial_prompt)
        + "<start_of_turn>model\n"
    )
    return prompt


input_samples = json.load(open(INPUT_SAMPLES_JSON)) if Path(INPUT_SAMPLES_JSON).exists() else []
final_samples = json.load(open(OUTPUT_SAMPLES_JSON)) if Path(OUTPUT_SAMPLES_JSON).exists else []
final_samples_ids = [i["id"] for i in final_samples]
input_samples = [i for i in input_samples if i["id"] not in final_samples_ids]

for i in tqdm(range(0, len(input_samples), BATCH_SIZE)):
    data = input_samples[i:i+BATCH_SIZE]
    prompts = []
    for entry in data:
        prompts.append(get_prompt(entry["original_text"], entry["prompt"]))
    results = model.generate(prompts, device=device, output_len=500)
    for idx, res in enumerate(results):
        new_entry = deepcopy(data[idx])
        new_entry["rewritten_text"] = res
        final_samples.append(new_entry)
    json.dump(final_samples, open(OUTPUT_SAMPLES_JSON, "w"))