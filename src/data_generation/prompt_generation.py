import json
from pathlib import Path
from tqdm.notebook import tqdm
from src.utils import json_parser_from_chat_response, run_gemini_prompt, run_openai_prompt

PROMPTS_JSON = "prompts.json"
N_PROMPTS = 10000
def get_prompt(n: int = 100):
    return (
        f"""
        You are a text transformation prompt creator. Your job is to generate a random prompt that can be used to ask an LLM to transform, edit or rewrite a piece of text. Your prompts should never ask for translations between languages, nor involve generating other type of media (like song melodies, paintings, logos, audios, etc.) and not involve special type of text such as programming language, code, or anything similar. Again, the input and outputs should also be some kind of TEXT and TEXT ONLY. You should also not ask to generate text from scratch, every prompt should be directed at transforming a given input. You shall return both the actual prompt you propose but also the extra information in a JSON format as specified below:
        
        OUTPUT FORMAT
        {{
            "prompt": "The actual proposed prompt",
            "input_characteristics": {{
                "type": "What type of input should this prompt be used for, e.g.: email, poem, song, scientific article, etc.",
                "style": "The style of the writing: e.g.: formal, informal, conversational, sci fi, fantasy, inspired by a specific artistic, etc.",
                "domain": "What should the subject matter of the input be,",
                "sentiment": "Overall sentintime of the input text",
                "complexity": "Assess the complexity level.",
                "length": "How long the text should be in generic terms (long, concise, short, abstract, etc.)",
                "specific_characteristics": "Specific characteristics of the input, for example, if include grammar mistakes, or mentions to a political figure, or any specific king of content, etc.".
            }},
            "transformation_characteristics": {{
                "type_of_transformation": "Define the transformation, e.g., summarization, paraphrasing, rewriting in a different style, rewriting in a different format, changing the target audience, correcting spelling and grammar, etc.",
            }},
            "output_characteristics": {{...this should have the same format as input_characteristics but for the expected output}}
        }}
                
        Generate {n} examples for me, make the actual prompt short and concise using a maximum of 20 words. Be creative, include different types of transformations and styles dont focus only on the examples given here. Return a JSON with the format {{"examples": [your list of examples here]}} and nothing more.
        """
    ).strip()

def gemini_generation(n: int = 100):
    responses = run_gemini_prompt(get_prompt(n), generation_config={"temperature": 1})
    final_text = "".join(responses)
    return json_parser_from_chat_response(final_text).get("examples", [])
    
def gpt_35_generation(n: int= 100):
    completion = run_openai_prompt(get_prompt(n), model="gpt-3.5-turbo", generation_config={"response_format": {"type": "json_object"}, "temperature": 1})
    return json.loads(completion.choices[0].message.content).get("examples", [])

prompts = json.load(open(PROMPTS_JSON, "r")) if Path(PROMPTS_JSON).exists() else []
pbar = tqdm(total=N_PROMPTS)
pbar.update(len(prompts))
while len(prompts) < N_PROMPTS:
    try:
        gemini_prompts = gemini_generation(10)
    except Exception as e:
        gemini_prompts = []
    try:
        gpt_35_prompts = gpt_35_generation(10)
    except Exception as e:
        gpt_35_prompts = []
    all_new_prompts = gemini_prompts + gpt_35_prompts
    prompts = prompts + all_new_prompts
    json.dump(prompts, open(PROMPTS_JSON, "w"), indent=4)
    pbar.update(len(all_new_prompts))