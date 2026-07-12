"""
Research R4 — evaluate proxy vs. oracle: accuracy tradeoff AND the
latency/cost savings, which is the actual headline result for this
research gap (Gap 3: computational scalability of MELON/ICON).
"""
import json
import time
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.proxy.features import extract_features, features_to_vector


def evaluate_proxy_speed(model_path: str, dataset_path: str) -> dict:
    clf = joblib.load(model_path)

    texts, labels = [], []
    with open(dataset_path) as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(1 if row["label"] == "injected" else 0)

    start = time.time()
    predictions = []
    for text in texts:
        features = extract_features(text)
        vec = np.array([features_to_vector(features)])
        pred = clf.predict(vec)[0]
        predictions.append(pred)
    proxy_latency = time.time() - start

    correct = sum(1 for p, l in zip(predictions, labels) if p == l)
    accuracy = correct / len(labels) if labels else 0.0

    return {
        "n_examples": len(labels),
        "proxy_total_latency_sec": round(proxy_latency, 4),
        "proxy_latency_per_example_ms": round((proxy_latency / len(labels)) * 1000, 3) if labels else 0,
        "proxy_accuracy_vs_oracle_labels": round(accuracy, 4),
    }


def print_cost_comparison(proxy_stats: dict, oracle_seconds_per_example: float):
    """
    oracle_seconds_per_example: measured separately by timing
    MelonOracle.evaluate() / IconOracle.evaluate() calls — pass in
    your own measured number here, this isn't computed automatically
    since it requires the GPU-loaded oracle model.
    """
    proxy_per_ex = proxy_stats["proxy_latency_per_example_ms"] / 1000
    speedup = oracle_seconds_per_example / proxy_per_ex if proxy_per_ex else float("inf")

    print("--- Cost comparison ---")
    print(f"Oracle: {oracle_seconds_per_example:.3f} sec/example")
    print(f"Proxy:  {proxy_per_ex:.6f} sec/example")
    print(f"Speedup: {speedup:.0f}x")
    print(f"Proxy accuracy vs oracle labels: {proxy_stats['proxy_accuracy_vs_oracle_labels']:.2%}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate proxy speed/accuracy vs oracle")
    parser.add_argument("--model", default="research/proxy/proxy_model.joblib")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--oracle-seconds-per-example",
        type=float,
        default=None,
        help="Measured oracle latency to compare against (from timing melon_baseline.py / icon_baseline.py)",
    )
    args = parser.parse_args()

    stats = evaluate_proxy_speed(args.model, args.dataset)
    print(json.dumps(stats, indent=2))

    if args.oracle_seconds_per_example:
        print()
        print_cost_comparison(stats, args.oracle_seconds_per_example)
