# Research Track: Cheap Proxy Models for MELON/ICON-style Detection

Targets research gap 3 from the project plan: MELON's masked re-execution
doubles inference cost, and ICON needs attention-layer access impossible on
closed APIs. Can a cheap proxy model approximate their detection signal at
a fraction of the cost, run at the gateway layer on every request?

## Structure
- `oracle/` — expensive, ground-truth baselines (MELON masked re-run, ICON
  attention entropy). Run these offline to generate labels, not live traffic.
- `proxy/` — cheap feature extraction + classifier training (student model).
- `data/` — labeled datasets (JSONL: `{"text": ..., "label": "injected"|"clean"}`).
  `sample_dataset.jsonl` is a tiny hand-written seed set for pipeline testing —
  nowhere near enough for a real result.
- `eval/` — compares proxy speed + accuracy against oracle.

## Setup
Requires more than the gateway's base requirements — GPU-capable PyTorch +
transformers, run separately from the live gateway to avoid memory contention:
```
pip install -r research/requirements.txt
```

## Workflow
1. **Generate oracle labels** (expensive, offline, run once per dataset):
   ```
   python research/oracle/melon_baseline.py --request "..." --content "..."
   python research/oracle/icon_baseline.py --text "..."
   ```
   Wrap these in a batch script over your full trajectory dataset (e.g. from
   AgentDojo) and dump results to JSONL matching `data/sample_dataset.jsonl`'s format.

2. **Train the cheap proxy**:
   ```
   python research/proxy/train.py --dataset research/data/your_dataset.jsonl
   ```

3. **Evaluate proxy vs. oracle** (accuracy + the actual speedup number):
   ```
   python research/eval/compare.py \
     --dataset research/data/your_dataset.jsonl \
     --oracle-seconds-per-example <measured from step 1>
   ```

## Model choice (24GB unified memory)
- **Oracle**: `meta-llama/Llama-3.1-8B-Instruct`, fp16, full attention access
  (`attn_implementation="eager"` — required for ICON's attention extraction).
  ~16GB, fits comfortably with headroom.
- **Proxy**: intentionally NOT an LLM to start — gradient-boosted trees over
  cheap lexical/heuristic features (reuses Phase 1's `app/heuristics.py` and
  `app/sanitize.py`). If this beats a reasonable accuracy bar, that's itself
  a finding — you may not need an LLM proxy at all. If not, the natural next
  step is a small (1-3B) LLM proxy, kept deliberately smaller than the oracle.

## Known gaps / next steps
- `sample_dataset.jsonl` is a toy seed set (8 examples) — real results need
  hundreds+ labeled trajectories, ideally sourced from AgentDojo's task suite.
- MELON's plan-similarity uses token-level Jaccard as a placeholder — swap
  for structured tool-call comparison once your agent emits function-calling
  output rather than free-text plans.
- Oracle latency isn't measured automatically — time your own
  `MelonOracle.evaluate()` / `IconOracle.evaluate()` calls and pass the
  result into `eval/compare.py --oracle-seconds-per-example`.
