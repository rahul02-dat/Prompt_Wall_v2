"""
Research R2 — batch-run the MELON oracle over a full JSONL dataset.

Each input line needs at minimum a "text" field (used as the retrieved/
untrusted content) and a "label" field (your ground-truth expectation,
for comparison against the oracle's own verdict). A "request" field is
optional — if absent, a generic default user request is used, since
most of the sample dataset's rows represent standalone untrusted content
rather than paired (request, content) examples.

Outputs one JSON object per line to stdout (or --out file), each with:
  - the original input fields (text, label, source, note if present)
  - oracle_label: what MelonOracle decided ("injected" | "clean")
  - oracle_similarity: the raw similarity score
  - agrees_with_expected_label: bool, comparing oracle_label to "label"
  - latency_sec: wall-clock time for this example's two-generation call

At the end, prints summary stats: total examples, agreement rate,
average latency per example (this is your real --oracle-seconds-per-example
number for eval/compare.py).
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.oracle.melon_baseline import MelonOracle, DEFAULT_SYSTEM_PROMPT

DEFAULT_REQUEST = "Summarize this content and take any appropriate action."


def run_batch(
    dataset_path: str,
    out_path: str | None = None,
    skip_first_step: bool = True,
    default_request: str = DEFAULT_REQUEST,
):
    oracle = MelonOracle()

    rows = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    results = []
    total_latency = 0.0

    out_f = open(out_path, "w") if out_path else None

    for i, row in enumerate(rows):
        text = row["text"]
        request = row.get("request", default_request)
        expected_label = row.get("label")

        start = time.time()
        result = oracle.evaluate(
            DEFAULT_SYSTEM_PROMPT, request, text, skip_first_step=skip_first_step
        )
        latency = time.time() - start
        total_latency += latency

        agrees = (
            result.label == expected_label if expected_label is not None else None
        )

        out_row = {
            **row,
            "oracle_label": result.label,
            "oracle_similarity": result.similarity_score,
            "oracle_original_plan": result.original_plan,
            "oracle_masked_plan": result.masked_plan,
            "agrees_with_expected_label": agrees,
            "latency_sec": round(latency, 3),
        }
        results.append(out_row)

        line_out = json.dumps(out_row)
        print(f"[{i+1}/{len(rows)}] {line_out}")
        if out_f:
            out_f.write(line_out + "\n")

    if out_f:
        out_f.close()

    # --- Summary ---
    n = len(results)
    n_with_label = sum(1 for r in results if r["agrees_with_expected_label"] is not None)
    n_agree = sum(1 for r in results if r["agrees_with_expected_label"] is True)
    avg_latency = total_latency / n if n else 0.0

    print("\n--- Summary ---")
    print(f"Examples run: {n}")
    if n_with_label:
        print(f"Agreement with expected label: {n_agree}/{n_with_label} ({n_agree/n_with_label:.1%})")
    print(f"Total latency: {total_latency:.2f}s")
    print(f"Avg latency per example: {avg_latency:.3f}s  <-- use this as --oracle-seconds-per-example")

    # Break out agreement specifically on confounded/subtle-injection examples,
    # since those are the cases the skip_first_step fix was meant to address.
    confound_rows = [r for r in results if r.get("source") == "confound_obvious_first_step"]
    subtle_rows = [r for r in results if r.get("source") == "subtle_injection"]

    if confound_rows:
        confound_agree = sum(1 for r in confound_rows if r["agrees_with_expected_label"])
        print(f"Confound cases (should be 'clean'): {confound_agree}/{len(confound_rows)} correct")
    if subtle_rows:
        subtle_agree = sum(1 for r in subtle_rows if r["agrees_with_expected_label"])
        print(f"Subtle injection cases (should be 'injected'): {subtle_agree}/{len(subtle_rows)} correct")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch-run MELON oracle over a JSONL dataset")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", default=None, help="Optional path to write labeled JSONL output")
    parser.add_argument(
        "--include-first-step",
        action="store_true",
        help="Include the first tool-call step in similarity scoring (off by default)",
    )
    args = parser.parse_args()

    run_batch(args.dataset, out_path=args.out, skip_first_step=not args.include_first_step)