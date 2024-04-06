"""
Runs local CV using one of the modules in the inference folder.
This will use both the original data and the data with variations included.
It will get both, select all splits that are not train, remove from entries
where the original_text leaks in and do an equal sample from the original data
and the new one.

Results are shown on both per prompt and per cluster
"""
import pandas as pd
import typer
from tqdm.auto import tqdm
import json
tqdm.pandas()
import importlib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import DATA_PATH

def process_prompt(prompt):
    # Makes lower case and removes punctuation
    prompt = prompt.lower()
    prompt = ''.join(e for e in prompt if e.isalnum() or e.isspace())
    return prompt

def evaluate(inference_module):
    mod = importlib.import_module(inference_module)
    predict_method = getattr(mod, "predict")
    cluster_to_split = json.load(open(str(DATA_PATH / "cluster_to_split.json")))
    cluster_to_split = {int(k): v for k, v in cluster_to_split.items()}
    new_data = pd.read_csv("/home/llm-prompt-recovery/data/new_data_for_training.csv")
    df = pd.read_csv(str(DATA_PATH / "data.csv"))
    
    df["split"] = df.cluster.map(cluster_to_split)
    new_data["split"] = new_data.cluster.map(cluster_to_split)

    df.dropna(inplace=True)
    new_data.dropna(inplace=True)

    new_df_curr = new_data
    new_train_original_text = new_data.loc[new_data.split == "train"].original_text.unique()

    test_df_curr = df
    train_original_text = df.loc[df.split == "train"].original_text.unique()

    test_df_curr = test_df_curr.loc[
        (~test_df_curr.original_text.isin(train_original_text)) &
        (~test_df_curr.original_text.isin(new_train_original_text))
    ]
    new_df_curr = new_df_curr.loc[
        (~new_df_curr.original_text.isin(train_original_text)) &
        (~new_df_curr.original_text.isin(new_train_original_text))
    ]
    test_df_curr = test_df_curr.sample(n=min(len(test_df_curr), len(new_df_curr)), random_state=1)
    new_df_curr = new_df_curr.sample(n=min(len(test_df_curr), len(new_df_curr)), random_state=1)
    print(len(test_df_curr), len(new_df_curr))

    test_df = pd.concat([new_df_curr, test_df_curr]).drop_duplicates().dropna()
    test_df = test_df.loc[test_df.rewritten_text.str.strip() != ""]
    test_df = test_df.loc[test_df.rewrite_prompt.str.strip() != ""]
    test_df = test_df.loc[test_df.original_text.str.strip() != ""]
    
    rewrite_prompt_gt = test_df["rewrite_prompt"].values
    test_df.drop("rewrite_prompt", axis=1)
    res = predict_method(test_df)
    extra_columns = [] 
    if isinstance(res, tuple):
        pred_df, extra_columns = res
    else:
        pred_df = res
    st_model = SentenceTransformer('sentence-transformers/sentence-t5-base')
    pred_df["rewrite_prompt_og"] = rewrite_prompt_gt
    scs = lambda row: abs((cosine_similarity(row["actual_embeddings"], row["pred_embeddings"])) ** 3)[0][0]
    pred_df["actual_embeddings"] = pred_df["rewrite_prompt_og"].progress_apply(lambda x: st_model.encode(process_prompt(x), normalize_embeddings=True, show_progress_bar=False).reshape(1, -1))
    pred_df["pred_embeddings"] = pred_df["rewrite_prompt"].progress_apply(lambda x: st_model.encode(process_prompt(x), normalize_embeddings=True, show_progress_bar=False).reshape(1, -1))

    pred_df["score"] = pred_df.apply(scs, axis=1)
    print(pred_df.score.describe())
    print(pred_df.groupby("cluster").score.mean().describe())
    final_columns = ["original_text", "rewritten_text", "rewrite_prompt_og", "rewrite_prompt", "score", "cluster", *extra_columns]
    pred_df[final_columns].to_csv(str(DATA_PATH / "eval.csv"), index=False)


if __name__ == "__main__":
    typer.run(evaluate)