import pandas as pd
import numpy as np
import typer
from tqdm.auto import tqdm
import json
tqdm.pandas()
import importlib

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import DATA_PATH

def evaluate(
        inference_module, 
        df_path = str(DATA_PATH / "data.csv"),
        output_path = str(DATA_PATH / "eval.csv"),
        cluster_to_split = str(DATA_PATH / "cluster_to_split.json")
    ):
    mod = importlib.import_module(inference_module)
    predict_method = getattr(mod, "predict")
    cluster_to_split = json.load(open(cluster_to_split))
    cluster_to_split = {int(k): v for k, v in cluster_to_split.items()}
    df = pd.read_csv(df_path)
    df["split"] = df.cluster.map(cluster_to_split)
    df.dropna(inplace=True)
    test_df = df.loc[df.split == "val"]
    rewrite_prompt_gt = test_df["rewrite_prompt"].values
    test_df.drop("rewrite_prompt", axis=1)
    pred_df = predict_method(test_df)
    st_model = SentenceTransformer('sentence-transformers/sentence-t5-base')
    pred_df["rewrite_prompt_og"] = rewrite_prompt_gt
    scs = lambda row: abs((cosine_similarity(row["actual_embeddings"], row["pred_embeddings"])) ** 3)[0][0]
    pred_df["actual_embeddings"] = pred_df["rewrite_prompt_og"].progress_apply(lambda x: st_model.encode(x, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1))
    pred_df["pred_embeddings"] = pred_df["rewrite_prompt"].progress_apply(lambda x: st_model.encode(x, normalize_embeddings=True, show_progress_bar=False).reshape(1, -1))
    pred_df["score"] = pred_df.apply(scs, axis=1)
    # Actual score is an average over a 10 random sample of a single prompt per cluster
    scores = []
    for _ in range(10):
        sampled_data = []
        for _, g in pred_df.groupby("cluster"):
            sampled_data.append(g.sample(n=1))
        sampled_data = pd.concat(sampled_data)
        scores.append(sampled_data.score.mean())
    print(f"Result: {np.mean(scores)}")
    pred_df[["original_text", "rewritten_text", "rewrite_prompt_og", "rewrite_prompt", "score"]].to_csv(output_path, index=False)


if __name__ == "__main__":
    typer.run(evaluate)