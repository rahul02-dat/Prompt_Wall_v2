"""
Research R1 — MELON baseline (expensive, ground-truth oracle).

MELON's core insight: if an agent is hijacked by indirect prompt injection,
its tool-call plan becomes driven by the injected content rather than the
user's actual request. Detect this by re-running the plan generation with
the user's intent masked out, and comparing whether the same tool calls
still get proposed.

This is deliberately expensive (2x inference per trajectory) — it's the
label generator for training the cheap proxy in research/proxy/, not
something meant to run in the live gateway path.

Requires: transformers, torch, accelerate
Model: meta-llama/Llama-3.1-8B-Instruct (or any local instruct model with
       tool-calling capability) — fp16, not quantized, run locally.
"""
import json
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ORACLE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

MASK_TOKEN = "[REDACTED USER REQUEST]"


@dataclass
class MelonResult:
    original_plan: str
    masked_plan: str
    similarity_score: float
    label: str  # "injected" | "clean"


class MelonOracle:
    def __init__(self, model_id: str = ORACLE_MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.model.eval()

    def _generate_plan(self, system_prompt: str, user_request: str, retrieved_content: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Request: {user_request}\n\nRetrieved content:\n{retrieved_content}\n\nWhat tool call(s) should be made? Respond with the tool call plan only.",
            },
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,  # deterministic — we need reproducible comparisons
            )

        generated = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated.strip()

    def _plan_similarity(self, plan_a: str, plan_b: str) -> float:
        """
        Cheap proxy for "same tool calls proposed": token-level Jaccard
        similarity. Swap for a proper tool-call parser + exact-match
        comparison once your agent has structured tool-call output
        (function calling) rather than free-text plans.
        """
        tokens_a = set(plan_a.lower().split())
        tokens_b = set(plan_b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def evaluate(
        self,
        system_prompt: str,
        user_request: str,
        retrieved_content: str,
        similarity_threshold: float = 0.7,
    ) -> MelonResult:
        original_plan = self._generate_plan(system_prompt, user_request, retrieved_content)
        masked_plan = self._generate_plan(system_prompt, MASK_TOKEN, retrieved_content)

        similarity = self._plan_similarity(original_plan, masked_plan)

        # High similarity between masked and unmasked plans means the plan
        # didn't depend on the user's actual request — a strong injection signal.
        label = "injected" if similarity >= similarity_threshold else "clean"

        return MelonResult(
            original_plan=original_plan,
            masked_plan=masked_plan,
            similarity_score=similarity,
            label=label,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MELON oracle on a single example")
    parser.add_argument("--system", default="You are a helpful assistant with access to email and file tools.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--content", required=True, help="Retrieved/untrusted content to test")
    args = parser.parse_args()

    oracle = MelonOracle()
    result = oracle.evaluate(args.system, args.request, args.content)
    print(json.dumps(asdict(result), indent=2))
