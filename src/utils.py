import json
import re
from typing import List, Optional

import vertexai
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from vertexai.generative_models import GenerativeModel

vertexai.init(project="", location="")
gemini_model = GenerativeModel("gemini-pro")
open_ai = OpenAI(api_key="")

def run_openai_prompt(
    prompt: str, model: str = "gpt-3.5-turbo", generation_config: Optional[dict] = None
) -> ChatCompletion:
    return open_ai.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "user",
            "content": prompt
            }
        ],
        **generation_config
    )


def run_gemini_prompt(
    prompt: List[str], generation_config: Optional[dict] = None
) -> List[str]:
    responses = gemini_model.generate_content(
        [prompt], generation_config=generation_config, stream=True
    )
    return [response.text for response in responses]


def json_parser_from_chat_response(text: str) -> dict:
    try:
        # Greedy search for 1st json candidate.
        match = re.search(
            r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
        )
        json_str = ""
        if match:
            json_str = match.group()
        json_object = json.loads(json_str, strict=False)
        return json_object
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse json from completion {text}. Got: {e}")
