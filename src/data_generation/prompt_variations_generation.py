import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from src.utils import json_parser_from_chat_response, run_gemini_prompt, run_openai_prompt, DATA_PATH

ORIGINAL_DATA = str(DATA_PATH / "data.csv")
OUTPUT_PATH = str(DATA_PATH / "prompt_variations.json")
def get_prompt(original_prompt: str, variation: str):
    return (
        f"""
        I will provide you with a text that represents a prompt that was used to transform a text.
        Your job is to return a variation of the same prompt by {variation}.

        Keep the intent input and output types the same. If the input and output is not described in the original prompt don't make assumptions when proposing a variation, keep everything generic in this case.

        Return only the new prompt using the following JSON structure:
        {{"prompt": "The variation of the original prompt"}}

        Return a valid JSON and nothing else. The output will be parsed so if its not a valid JSON it will faile, so please always provide a valid JSON and nothing more.

        ----------------
        Input: {original_prompt}
        """
    ).strip()

def gpt_35_generation(og_prompt, var):
    completion = run_openai_prompt(get_prompt(og_prompt, var), model="gpt-3.5-turbo", generation_config={"response_format": {"type": "json_object"}, "temperature": 1})
    return json.loads(completion.choices[0].message.content).get("prompt", None)

original_prompts = pd.read_csv(ORIGINAL_DATA).rewrite_prompt.unique()
new_prompts = json.load(open(OUTPUT_PATH, "r")) if Path(OUTPUT_PATH).exists() else {}
print(len(new_prompts))
variations = [
    "making the prompt much more detailed, add as much details to the prompt as you can think",
    "adding new elements to the instruction, add one two new aspects to it",
    "adding a focus on a specific aspect of the instruction, this aspect can also be something new",
    "making the prompt more concise",
    "making the writing style of the prompt itself more informal",
    "making the writing style of the prompt itself more formal"
]
original_prompts = [i for i in original_prompts if i not in new_prompts or len(new_prompts[i]) > 0]
for og_prompt in tqdm(original_prompts):
    if og_prompt in new_prompts and len(new_prompts[og_prompt]) > 0:
        continue
    if og_prompt not in new_prompts:
        new_prompts[og_prompt] = []
    for var in variations:
        try:
            new_prompt = gpt_35_generation(og_prompt, var)
            new_prompts[og_prompt].append(new_prompt)
            json.dump(new_prompts, open(OUTPUT_PATH, "w"), ensure_ascii=False, indent=True)
        except Exception as e:
            continue
