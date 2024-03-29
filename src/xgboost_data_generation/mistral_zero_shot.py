import json

import torch
from tqdm.auto import tqdm
from pathlib import Path
import pickle as pkl
import pandas as pd
import numpy as np
from src.utils import json_parser_from_chat_response, DATA_PATH
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tqdm.pandas()

RANDOM_PROJECTION_PATH = str(DATA_PATH / "random_projection_logits_mistral_zero_shot.pkl")
CURRENT_OUTPUT_PATH = str(DATA_PATH / "mistral_zero_shot_output.pkl")
DATA_PATH = str(DATA_PATH / "data.csv")
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 1

def get_rewrite_prompt(original_text: str, transformed_text: str):
    json_input = json.dumps({
        "original_text": original_text, 
        "rewritten_text": transformed_text
    }, indent=4)
    return f"""
        I will give you a JSON with following structure:
        {{
            'original_text': 'An original piece of text.'
            'rewritten_text': 'A version of original_text that was rewritten by an LLM according to a specific prompt.'
        }}

        Given the task of understanding how text is rewritten by analyzing the original_text and rewritten_text, your goal is to deduce the specific instructions or prompt that was most likely used to generate the rewritten text from the original text. Consider the changes made in terms of style, tone, structure, and content. Assess whether the rewrite focuses on summarization, paraphrasing, stylistic alteration (e.g., formal to informal), or any specific content changes (e.g., making the text more concise, expanding on ideas, or altering the perspective). Follow this steps:

        1. Read the original_text: Start by thoroughly understanding the content, style, tone, and purpose of the original text. Note any key themes, technical terms, and the overall message.
        2. Analyze the rewritten_text: Examine how the rewritten text compares to the original. Identify what has been changed, added, or omitted. Pay close attention to changes in style (formal, informal), tone (serious, humorous), structure (paragraph order, sentence structure), and any shifts in perspective or emphasis.
        3. Infer the Prompt: Based on your analysis, infer the most likely prompt that guided the rewriting process. Your inference should account for the observed changes in style, tone, structure, and content. Specify the type of task (e.g., summarize, paraphrase, make more accessible to a general audience), any specific directions evident from the changes, and any specific stylistic choice (e.g., 'as a poem', 'as a song', 'in the style of Shakespeare', etc...)

        Based on your analysis return the prompt as if you were given the instruction your self like:
        "Rewrite this text..."
        "Transform this ... into ... based on the style of ..."
        
        Make the prompt short and direct using a maximum of 20 words.


        Return your answer using the following JSON structure:
        {{"prompt": "Your best guess for the prompt used"}}
        

            
        Return a valid JSON as output and nothing more.
        
        -----------------------
        Input: 
        
        {json_input}
    """

def get_chat_template(tokenizer, original_text: str, transformed_text: str):
    prompt = get_rewrite_prompt(original_text, transformed_text)
    chat = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt

def run_inference(model, tokenizer, prompts, do_sample = False):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    generation_result = model.generate(
        **inputs, pad_token_id=2, max_new_tokens=768, temperature=1, do_sample=do_sample,
        return_dict_in_generate=True, output_logits=True
    )
    generate_ids = generation_result["sequences"][:, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    logits = [[] for _ in range(len(generation_result["sequences"]))]
    for l in generation_result["logits"]:
        for idx, v in enumerate(l):
            logits[idx].append(v)
    final_res = []
    for i in range(len(generation_result["sequences"])):
        final_res.append({
            "text": generated_text[i],
            "logits": torch.stack(logits[i]).cpu().numpy(),
            "output_ids": generate_ids[i].cpu().numpy()
        })
    return final_res

def batch_get_rewrite_prompts(model, tokenizer, original_text, rewritten_text):
    default_prompt = "Make this text better"
    og_rw_text = list(zip(original_text, rewritten_text))
    prompts = [get_chat_template(tokenizer, og_text, rw_text) for og_text, rw_text in og_rw_text]
    output = run_inference(model, tokenizer, prompts, do_sample=False)
    final_output = []
    for idx, res in enumerate(output):
        tries = 5
        while tries > 0:
            try:
                if tries != 5:
                    og_text, rw_text = og_rw_text[idx]
                    prompt = [get_chat_template(tokenizer, og_text, rw_text)]
                    res = run_inference(model, tokenizer, prompt, do_sample=True)[0]
                new_prompt = json_parser_from_chat_response(res["text"]).get('prompt', default_prompt)
                break
            except Exception as e:
                print(e)
                if tries == 0:
                    new_prompt = default_prompt
                else:
                    tries -= 1
        res["rewrite_prompt"] = new_prompt
        final_output.append(res)
    return final_output

def batch_predict(
        df, 
        mistral,
        mistral_tokenizer, 
        current_output,
        random_projection,
    ):
    mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        original_texts = df.original_text.values[i:i+BATCH_SIZE]
        rewritten_texts = df.rewritten_text.values[i:i+BATCH_SIZE]
        entries_ids = df.id.values[i:i+BATCH_SIZE]
        results = batch_get_rewrite_prompts(mistral, mistral_tokenizer, original_texts, rewritten_texts)
        for idx, entry_id in enumerate(entries_ids):
            logits = results[idx]["logits"]
            to_pad = 80 - len(logits)
            if to_pad < 0:
                logits = logits[:80,:]
            elif to_pad > 0:
                padding = np.zeros((to_pad, logits.shape[1]))
                logits = np.concatenate([logits, padding], axis=0)
            reduced_logits = random_projection.transform(logits).flatten().astype(np.float16)
            new_entry = {
                "projected_logits": reduced_logits,
                "rewrite_prompt": results[idx]["rewrite_prompt"]
            }
            current_output[entry_id] = new_entry
            pkl.dump(current_output, open(CURRENT_OUTPUT_PATH, "wb"))


def predict(df, current_output, random_projection):
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    mistral = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",  device_map="auto", quantization_config=quant_config
    )
    mistral.config.use_cache = False
    mistral.config.pretraining_tp = 1
    mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    return batch_predict(df, mistral, mistral_tokenizer, current_output, random_projection)


df = pd.read_csv(DATA_PATH)
current_output = pkl.load(open(CURRENT_OUTPUT_PATH, "rb"))  if Path(CURRENT_OUTPUT_PATH).exists() else {}
random_projection = pkl.load(open(RANDOM_PROJECTION_PATH, "rb"))
df = df.loc[~df.id.isin(current_output)]
predict(df, current_output, random_projection)