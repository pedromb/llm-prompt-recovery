from unsloth import FastLanguageModel
from tqdm.auto import tqdm
import json
import torch
import pandas as pd
import random

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2b-it-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
    token = ""
)

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

processed_prompts = set()
old_data = pd.read_csv("/home/pbernardo/github/llm-prompt-recovery/data/data.csv")
new_data = json.load(open("/home/pbernardo/github/llm-prompt-recovery/data/new_data.json", "r"))
prompt_variations = json.load(open("/home/pbernardo/github/llm-prompt-recovery/data/prompt_variations.json", "r"))
for entry in new_data:
    for og_prompt, vars in prompt_variations.items():
        if entry["rewrite_prompt"] in vars:
            processed_prompts.add(og_prompt)
while True:
    try:
        prompt_variations = json.load(open("/home/pbernardo/github/llm-prompt-recovery/data/prompt_variations.json", "r"))
    except:
        continue
    prompts_left = [i for i in prompt_variations if i not in processed_prompts]
    random.shuffle(prompts_left)
    if not prompts_left:
        prompts_left = [i for i in prompt_variations if i not in processed_prompts]
        if not prompts_left:
            break
    prompt = prompts_left.pop()
    og_data = old_data.loc[old_data.rewrite_prompt == prompt]
    total_len = len(og_data) * len(prompt_variations[prompt])
    new_pbar = tqdm(range(total_len), leave=False)
    for _, row in og_data.iterrows():
        for var in prompt_variations[prompt]:
            new_entry = {
                "original_text": row.original_text,
                "rewrite_prompt": var,
                "cluster": row.cluster
            }
            prompt_to_feed = get_prompt(row.original_text, var)
            inputs = tokenizer(prompt_to_feed, return_tensors="pt").to(DEVICE)
            generate_ids = model.generate(
                **inputs, 
                pad_token_id=tokenizer.pad_token_id,
                max_new_tokens=768, 
                temperature=1, 
                do_sample=True
            )
            generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            new_entry["rewritten_text"] = generated_text
            new_data.append(new_entry)
            json.dump(new_data, open("/home/pbernardo/github/llm-prompt-recovery/data/new_data.json", "w"), indent=4, ensure_ascii=False)
            new_pbar.update(1)
    processed_prompts.add(prompt)