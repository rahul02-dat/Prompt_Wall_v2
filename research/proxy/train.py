"""
Research R3 — train the cheap proxy classifier on oracle-labeled data.

Input: a JSONL dataset where each line is
  {"text": "...", "label": "injected" | "clean", "source": "melon" | "icon"}
produced by running the oracle scripts over your trajectory dataset
(research/data/) and dumping results.

The proxy here is intentionally NOT a large language model — a gradient
boosted tree over cheap features. This is the cheapest reasonable
baseline to beat before considering a small (1-3B) LLM proxy; if this
already gets good accuracy, that itself is a research finding (you may
not need an LLM at all for this detection task).
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `app.*` imports
from research.proxy.features import extract_features, features_to_vector, FEATURE_NAMES


def load_dataset(path: str) -> tuple[list[list[float]], list[int]]:
    X, y = [], []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            features = extract_features(row["text"])
            X.append(features_to_vector(features))
            y.append(1 if row["label"] == "injected" else 0)
    return X, y


def train(dataset_path: str, model_out: str = "research/proxy/proxy_model.joblib"):
    X, y = load_dataset(dataset_path)
    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {len(y)} examples ({y.sum()} injected, {len(y) - y.sum()} clean)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    print("\n--- Classification report ---")
    print(classification_report(y_test, preds, target_names=["clean", "injected"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")

    print("\n--- Feature importances ---")
    for name, importance in sorted(
        zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]
    ):
        print(f"  {name}: {importance:.4f}")

    joblib.dump(clf, model_out)
    print(f"\nSaved proxy model to {model_out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the cheap proxy classifier")
    parser.add_argument("--dataset", required=True, help="Path to JSONL oracle-labeled dataset")
    parser.add_argument("--out", default="research/proxy/proxy_model.joblib")
    args = parser.parse_args()

    train(args.dataset, args.out)
