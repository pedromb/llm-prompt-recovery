import json

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import json_parser_from_chat_response
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tqdm.pandas()

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
BATCH_SIZE = 16
MODEL_CHECKPOINT = "/home/llm-prompt-recovery/data/weights/mistral-7b-finetuned-lora-unsloth"

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
        3. Infer the Prompt: Based on your analysis, infer the most likely prompt that guided the rewriting process. Your inference should account for the observed changes in style, tone, structure, and content. Specify the type of task (e.g., summarize, paraphrase, make more accessible to a general audience), any specific directions evident from the changes, and any specific stylistic choice.

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
    generate_ids = model.generate(**inputs, pad_token_id=2, max_new_tokens=768, temperature=1, do_sample=do_sample)
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return generated_text

def batch_get_rewrite_prompts(model, tokenizer, original_text, rewritten_text):
    default_prompt = "Make this text better"
    prompts = []
    for og_text, rw_text in zip(original_text, rewritten_text):
        prompt = get_rewrite_prompt(og_text, rw_text)
        chat = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
    generate_ids = model.generate(**inputs, pad_token_id=2, max_new_tokens=768, temperature=1, do_sample=False)
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    rewrite_prompts = []
    for rw_prompt in generated_text:
        try:
            new_prompt = json_parser_from_chat_response(rw_prompt).get('prompt', default_prompt)
        except:
            new_prompt = default_prompt
        rewrite_prompts.append(new_prompt)
    return rewrite_prompts

def batch_predict(df, mistral, mistral_tokenizer):
    final_prompts = [] 
    mistral_tokenizer.pad_token = mistral_tokenizer.eos_token
    for i in tqdm(range(0, len(df), BATCH_SIZE)):
        original_texts = df.original_text.values[i:i+BATCH_SIZE]
        rewritten_texts = df.rewritten_text.values[i:i+BATCH_SIZE]
        prompts = batch_get_rewrite_prompts(mistral, mistral_tokenizer, original_texts, rewritten_texts)
        final_prompts.extend(prompts)
    df["rewrite_prompt"] = final_prompts
    return df

def predict(df):
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )
    mistral = AutoModelForCausalLM.from_pretrained(
       MODEL_CHECKPOINT,  device_map="auto", quantization_config=quant_config
    )
    mistral.config.use_cache = False
    mistral.config.pretraining_tp = 1
    mistral_tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    mistral_tokenizer.padding_side = "left"
    return batch_predict(df, mistral, mistral_tokenizer)