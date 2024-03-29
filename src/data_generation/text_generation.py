import json
from tqdm.notebook import tqdm
from copy import deepcopy
from pathlib import Path
from src.utils import run_gemini_prompt, run_openai_prompt

PROMPTS_JSON = "prompts.json"
OUTPUT_JSON = "prompts_with_original_text.json"

def get_prompt(input_desc: dict):
    json_input = json.dumps(input_desc)
    return (
        f"""
        I wil give you a JSON input that describes charactertistics of a text. Your job is to generate a text that fits the description.
        Use a maximum of 400 words in your response. Return only the text and nothing else.

        JSON input:

        {json_input}
        """
    ).strip()

def gemini_generation(input_desc: dict):
    responses = run_gemini_prompt(get_prompt(input_desc), generation_config={"temperature": 1})
    final_text = "".join(responses)
    return final_text
    
def gpt_35_generation(input_desc: dict):
    completion = run_openai_prompt(get_prompt(input_desc), model="gpt-3.5-turbo", generation_config={"temperature": 1})
    return completion.choices[0].message.content

prompts = json.load(open(PROMPTS_JSON, "r")) if Path(PROMPTS_JSON).exists() else []
final_samples = json.load(open(OUTPUT_JSON, "r")) if Path(OUTPUT_JSON).exists() else []
for p in tqdm(prompts):
    try:
        input_chars = p["input_characteristics"]
        try:
            gemini_text = gemini_generation(input_chars)
        except Exception as e:
            gemini_text = None
        try:
            gpt_35_text = gpt_35_generation(input_chars)
        except Exception as e:
            gpt_35_text = None
    except:
        continue
    if gemini_text:
        new_entry = deepcopy(p)
        new_entry["original_text"] = gemini_text
        new_entry["model"] = "gemini"
        final_samples.append(new_entry)
    if gpt_35_text:
        new_entry = deepcopy(p)
        new_entry["original_text"] = gpt_35_text
        new_entry["model"] = "gpt-3.5"
        final_samples.append(new_entry)
    json.dump(final_samples, open(OUTPUT_JSON, "w"), indent=4)