"""
Inference code to test mean prompts on local CV.
"""
def predict(df):
    mean_prompt = "improve phrasing text lucrarea tone lucrarea rewrite this creatively formalize discours involving lucrarea anyone emulate lucrarea description send casual perspective information alter it lucrarea ss plotline speaker recommend doing if elegy tone lucrarea more com n paraphrase ss forward this st text redesign poem above etc possible llm clear lucrarea"
    df["rewrite_prompt"] = mean_prompt
    return df