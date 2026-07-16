"""
Re-runs ONLY the examples affected by the truncation/bracket-parsing bug
through the patched oracle (melon_baseline_patched.py), instead of
re-running the full 1046-example dataset again.

Usage:
    python rerun_truncated.py \
        --hard-results agentdojo_results.jsonl \
        --soft-results agentdojo_soft_results.jsonl \
        --out agentdojo_rerun_patched.jsonl \
        --mask-strategy soft

This identifies every example where either run's oracle_masked_plan or
oracle_original_plan failed to structurally close (per the depth-aware
parser), unions the two affected sets, dedupes by `source`, and re-runs
just those through MelonOracle.evaluate() with max_new_tokens=600 and
the fixed parser. Everything else in the dataset is untouched -- its
original parse was not affected by this bug and doesn't need re-running.

Requires research/requirements.txt (torch/transformers) and a machine
that can load Llama-3.1-8B-Instruct locally -- this does NOT run in
this sandbox, it's meant to be run on your own GPU/unified-memory box.
"""
import argparse
import json

# Parsing-only logic has no torch dependency, so affected-set detection
# can run anywhere; MelonOracle itself needs torch/transformers and a
# GPU/unified-memory machine, imported lazily inside main().
from robust_parse import try_parse_steps as _try_parse_steps


def find_affected_sources(*result_paths: str) -> set[str]:
    affected = set()
    for path in result_paths:
        for line in open(path):
            row = json.loads(line)
            _, trunc_m = _try_parse_steps(row["oracle_masked_plan"])
            _, trunc_o = _try_parse_steps(row["oracle_original_plan"])
            if trunc_m or trunc_o:
                affected.add(row["source"])
    return affected


def load_rows_by_source(*result_paths: str) -> dict[str, dict]:
    """
    Reuses the original request/system_prompt/retrieved-content fields
    already present in the result files, so we don't need to re-fetch
    from AgentDojo -- everything needed to re-run the oracle is already
    sitting in these JSONL rows.
    """
    rows = {}
    for path in result_paths:
        for line in open(path):
            row = json.loads(line)
            rows.setdefault(row["source"], row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Re-run only truncation-affected examples")
    parser.add_argument("--hard-results", required=True)
    parser.add_argument("--soft-results", required=True)
    parser.add_argument("--out", default="agentdojo_rerun_patched.jsonl")
    parser.add_argument("--mask-strategy", choices=["hard", "soft"], default="soft")
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--only-suite", default=None,
                         help="Restrict to one AgentDojo suite (workspace/travel/banking/slack) "
                              "for a fast diagnostic pass instead of the full affected set.")
    parser.add_argument("--sample-limit", type=int, default=None,
                         help="Only process the first N affected examples (after --only-suite "
                              "filtering) -- useful for a quick eyeball pass before committing "
                              "GPU time to the full affected set.")
    parser.add_argument("--still-truncated-from", default=None,
                         help="Path to a previous rerun output (e.g. agentdojo_rerun_patched.jsonl). "
                              "If given, restricts this run to sources that were STILL truncated "
                              "in that file, rather than the original hard/soft-flagged set -- for "
                              "iterating on max_new_tokens without re-running everything again.")
    args = parser.parse_args()

    affected = find_affected_sources(args.hard_results, args.soft_results)
    print(f"Found {len(affected)} unique examples affected by truncation in either original run.")

    if args.still_truncated_from:
        still = {
            json.loads(l)["source"]
            for l in open(args.still_truncated_from)
            if json.loads(l)["new_original_truncated"] or json.loads(l)["new_masked_truncated"]
        }
        affected &= still
        print(f"Restricted to {len(affected)} sources still truncated in {args.still_truncated_from}.")

    all_rows = load_rows_by_source(args.hard_results, args.soft_results)

    targets = sorted(affected)
    if args.only_suite:
        targets = [s for s in targets if s.split(":")[1] == args.only_suite]
        print(f"Restricted to suite '{args.only_suite}': {len(targets)} examples.")
    if args.sample_limit:
        targets = targets[:args.sample_limit]
        print(f"Restricted to first {len(targets)} examples (--sample-limit).")

    from melon_baseline import MelonOracle  # requires torch/transformers
    oracle = MelonOracle()

    n_flipped = 0
    n_still_truncated = 0
    n_looping = 0

    with open(args.out, "w") as out_f:
        for i, source in enumerate(targets, 1):
            row = all_rows[source]
            result = oracle.evaluate(
                system_prompt=row["system_prompt"],
                user_request=row["request"],
                retrieved_content=row["text"],
                mask_strategy=args.mask_strategy,
                max_new_tokens=args.max_new_tokens,
            )

            old_label = row["label"]  # ground-truth expected label, unchanged
            agrees_now = (result.label == old_label)

            record = {
                "source": source,
                "label": old_label,
                "attack_present": row["attack_present"],
                "new_oracle_label": result.label,
                "new_similarity": result.similarity_score,
                "new_max_step_similarity": result.max_step_similarity,
                "new_original_truncated": result.original_truncated,
                "new_masked_truncated": result.masked_truncated,
                "new_original_token_count": result.original_token_count,
                "new_masked_token_count": result.masked_token_count,
                "new_masked_is_looping": result.masked_is_looping,
                "new_masked_tool_sequence": result.masked_tool_sequence,
                # Full plan text -- essential for actually diagnosing WHY
                # something is still truncated, not just knowing that it is.
                "new_original_plan": result.original_plan,
                "new_masked_plan": result.masked_plan,
                "agrees_with_expected_label": agrees_now,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()  # so a partial run is still readable if interrupted

            if agrees_now and result.label == "injected":
                n_flipped += 1
            if result.original_truncated or result.masked_truncated:
                n_still_truncated += 1
            if result.masked_is_looping:
                n_looping += 1

            if i % 10 == 0 or i == len(targets):
                print(f"  ...{i}/{len(targets)} done")

    print(f"\nDone. Wrote {len(targets)} re-evaluated examples to {args.out}")
    print(f"Now correctly labeled 'injected' (previously mislabeled): {n_flipped}")
    print(f"Still truncated even at max_new_tokens={args.max_new_tokens}: {n_still_truncated}")
    print(f"Of those, flagged as a repeating tool-call loop (not just long): {n_looping}")
    print("Inspect new_masked_plan / new_masked_tool_sequence by hand on a few still-truncated, "
          "non-looping rows to tell genuine multi-step task length apart from a degenerate loop "
          "before deciding whether to raise --max-new-tokens further or add a stop condition.")


if __name__ == "__main__":
    main()