from unsloth import FastLanguageModel
from tqdm.auto import tqdm
import json
import torch
import pandas as pd
import random

max_seq_length = 2048
BATCH_SIZE = 2
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2b-it-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
    token = ""
)
FastLanguageModel.for_inference(model)

def get_prompt(original_text: str, rewrite_prompt: str):
    USER_CHAT_TEMPLATE = "<bos><start_of_turn>user\n{prompt}<end_of_turn>\n"
    initial_prompt = f"""{rewrite_prompt}\n\n{original_text}"""
    prompt = (
        USER_CHAT_TEMPLATE.format(prompt=initial_prompt)
        + "<start_of_turn>model\n"
    )
    return prompt

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

processed_prompts = set()
old_data = pd.read_csv("/home/pbernardo/github/llm-prompt-recovery/data/data.csv")
new_data = json.load(open("/home/pbernardo/github/llm-prompt-recovery/data/new_data.json", "r"))
new_data_df = pd.DataFrame(new_data)
prompt_variations = json.load(open("/home/pbernardo/github/llm-prompt-recovery/data/prompt_variations.json", "r"))
prompts_selected = set(json.load(open("/home/pbernardo/github/llm-prompt-recovery/data/prompts_selected_new.json", "r")))
prompts_selected_og = []     
for rewrite_prompt in tqdm(new_data_df.rewrite_prompt.unique()):
    for og_prompt, vars in prompt_variations.items():
        if og_prompt in processed_prompts:
            continue
        if rewrite_prompt in vars:
            processed_prompts.add(og_prompt)

for og_prompt, vars in tqdm(prompt_variations.items()):
    if og_prompt in prompts_selected:
        prompts_selected_og.append(og_prompt)
        continue
    for v in vars:
        if v in prompts_selected:
            prompts_selected_og.append(og_prompt)
            break

prompts_selected_og = [i for i in prompts_selected_og if i not in processed_prompts]
print(len(processed_prompts), len(prompt_variations) - len(processed_prompts))
while True:
    if len(prompts_selected_og) == 0:
        prompts_left = [i for i in prompt_variations if i not in processed_prompts]
        random.shuffle(prompts_left)
        if not prompts_left:
            break
    else:
        prompts_left = prompts_selected_og
    prompt = prompts_left.pop()
    og_data = old_data.loc[old_data.rewrite_prompt == prompt]
    input_samples = []
    for _, row in og_data.iterrows():
        for var in prompt_variations[prompt]:
            input_samples.append({
                "original_text": row.original_text,
                "rewrite_prompt": var,
                "cluster": row.cluster
            })
    total_len = len(input_samples)
    for i in tqdm(range(0, len(input_samples), BATCH_SIZE), desc = f"Prompts Left {len(prompts_left)}"):
        entries = input_samples[i:i+BATCH_SIZE]
        prompts_to_feed = [get_prompt(x["original_text"], x["rewrite_prompt"]) for x in entries]
        inputs = tokenizer(prompts_to_feed, padding=True, return_tensors="pt").to(DEVICE)
        generate_ids = model.generate(
            **inputs, 
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=768, 
            temperature=1, 
            do_sample=True
        )
        generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for idx, res in enumerate(generated_text):
            new_entry = {
                "original_text": entries[idx]["original_text"],
                "rewrite_prompt": entries[idx]["rewrite_prompt"],
                "cluster": entries[idx]["cluster"]
            }
            new_entry["rewritten_text"] = res
            new_data.append(new_entry)
        json.dump(new_data, open("/home/pbernardo/github/llm-prompt-recovery/data/new_data.json", "w"), indent=4, ensure_ascii=False)
    processed_prompts.add(prompt)