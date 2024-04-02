from sentence_transformers import SentenceTransformer
import xgboost as xgb
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
tqdm.pandas()
import numpy as np
import pickle as pkl
from pathlib import Path
from sklearn.model_selection import GroupKFold, train_test_split
from src.utils import DATA_PATH

DATASET = str(DATA_PATH / "data.csv")
MODEL_OUTPUTS = str(DATA_PATH / "mistral_zero_shot_output.pkl")
PRECOMPUTED_FEATURES = str(DATA_PATH / "precomputed_features_mistral_zero_shot_xgboost.pkl")

OUTPUT_PATH = str(DATA_PATH / "weights/xgboost_model.json")

data = pd.read_csv(DATASET)
model_outputs = pkl.load(open(MODEL_OUTPUTS, "rb"))
precomputed_features = pkl.load(open(PRECOMPUTED_FEATURES, "rb")) if Path(PRECOMPUTED_FEATURES).exists() else {}
data = data.loc[data.id.isin(model_outputs)]
embedding_model = SentenceTransformer('sentence-transformers/sentence-t5-base')

def calc_score(id, rewrite_prompt, rewrite_prompt_pred):
    if id in precomputed_features:
        return precomputed_features[id]["score"]
    emb = embedding_model.encode([rewrite_prompt, rewrite_prompt_pred], normalize_embeddings=True)
    cos_sim = cosine_similarity(emb)[0]
    return cos_sim[1]**3

def prepare_features(x):
    if x.id in precomputed_features:
        return precomputed_features[x.id]["features"]
    embeddings_original_text = embedding_model.encode([x.original_text], show_progress_bar = False)
    embeddings_rewrite_prompt_pred = embedding_model.encode([x.rewrite_prompt_pred], show_progress_bar = False)
    embeddings_rewritten_text = embedding_model.encode([x.rewritten_text], show_progress_bar = False)
    projected_logits = model_outputs[x.id]["projected_logits"].reshape(1, -1)
    return np.concatenate([embeddings_original_text, projected_logits, embeddings_rewrite_prompt_pred, embeddings_rewritten_text], axis=1).flatten().astype(np.float16)


data["rewrite_prompt_pred"] = data.id.progress_apply(lambda x: model_outputs[x]["rewrite_prompt"])
data["score"] = data.progress_apply(lambda x: calc_score(x.id, x.rewrite_prompt, x.rewrite_prompt_pred), axis=1)
data["features"] = pd.Series(data.progress_apply(prepare_features, axis=1))
for _, row in tqdm(data.iterrows(), total=len(data), desc="Saving features"):
    if row.id in precomputed_features:
        continue
    precomputed_features[row.id] = {
        "features": row.features,
        "score": row.score
    }
with open(PRECOMPUTED_FEATURES, "wb") as f:
    pkl.dump(precomputed_features, f)


folds = GroupKFold(n_splits=10)
for i, (train_index, test_index) in enumerate(folds.split(data.features, data.score, data.cluster)):
    train = data.iloc[train_index]
    test = data.iloc[test_index]
    X = np.stack(train.features)
    y = train.score.values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    xgb_reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        max_depth=3,
        n_estimators=9999,
        colsample_bytree=0.8,
        subsample=0.8,
        learning_rate=0.03,
    )
    xgb_reg.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        verbose=True,
        early_stopping_rounds=50
    )
    # train only one fold
    break

test["pred_score"] = xgb_reg.predict(np.stack(test.features))
print(test[["score", "pred_score"]].corr())
test["error"] = abs(test.pred_score - test.score)
print(test.error.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
xgb_reg.save_model(OUTPUT_PATH)