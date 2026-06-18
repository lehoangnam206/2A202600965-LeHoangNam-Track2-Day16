import time
import json
import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

DATA_PATH = "creditcard.csv"

print("Loading data...")
t0 = time.time()
df = pd.read_csv(DATA_PATH)
load_time = time.time() - t0

print("Dataset shape:", df.shape)
print("Class distribution:")
print(df["Class"].value_counts())

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print("Training LightGBM...")
t1 = time.time()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
)
train_time = time.time() - t1

best_iteration = getattr(model, "best_iteration_", None)
if best_iteration is None or best_iteration <= 0:
    best_iteration = model.n_estimators

print("Evaluating...")
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

auc = roc_auc_score(y_test, y_prob)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

one_row = X_test.iloc[[0]]
t2 = time.time()
_ = model.predict_proba(one_row)
one_row_latency_ms = (time.time() - t2) * 1000

batch_1000 = X_test.iloc[:1000]
t3 = time.time()
_ = model.predict_proba(batch_1000)
batch_time = time.time() - t3
throughput_rows_per_sec = 1000 / batch_time

metrics = {
    "load_data_time_sec": round(load_time, 4),
    "training_time_sec": round(train_time, 4),
    "best_iteration": int(best_iteration),
    "auc_roc": round(auc, 6),
    "accuracy": round(accuracy, 6),
    "f1_score": round(f1, 6),
    "precision": round(precision, 6),
    "recall": round(recall, 6),
    "inference_latency_1_row_ms": round(one_row_latency_ms, 6),
    "inference_throughput_1000_rows_per_sec": round(throughput_rows_per_sec, 2),
}

print("\n========== BENCHMARK RESULTS ==========")
for k, v in metrics.items():
    print(f"{k}: {v}")

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

pd.DataFrame([metrics]).to_csv("metrics.csv", index=False)

print("\nSaved metrics.json and metrics.csv")
