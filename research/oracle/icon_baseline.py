"""
Research R1 — ICON baseline (expensive, ground-truth oracle).

ICON's core insight: successful prompt injections cause "attention collapse"
— the model's attention abnormally concentrates on a small set of
adversarial trigger tokens. Detect this via a Focus Intensity Score (FIS)
computed from attention entropy statistics across layers/timesteps.

Requires full attention-weight access, which is why this can't run against
closed APIs (OpenAI/Anthropic) — only local, weights-accessible models.
Deliberately expensive: extracts and analyzes attention over all layers.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ORACLE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass
class IconResult:
    min_entropy: float
    mean_entropy: float
    std_entropy: float
    focus_intensity_score: float
    label: str  # "injected" | "clean"


class IconOracle:
    def __init__(self, model_id: str = ORACLE_MODEL_ID, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            attn_implementation="eager",  # required to get attention weights out
        ).to(self.device)
        self.model.eval()

    def _attention_entropies(self, text: str) -> list[float]:
        """
        Runs a forward pass and computes per-layer attention entropy,
        averaged over heads, for the final token's attention distribution
        (i.e. what the model is "focusing on" when about to generate next).
        Low entropy = attention concentrated on very few tokens = potential
        attention collapse signal per the ICON paper.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        entropies = []
        for layer_attn in outputs.attentions:
            # layer_attn shape: [batch, heads, seq_len, seq_len]
            last_token_attn = layer_attn[0, :, -1, :]  # [heads, seq_len]
            probs = last_token_attn.clamp(min=1e-9)
            entropy_per_head = -(probs * probs.log()).sum(dim=-1)  # [heads]
            entropies.append(entropy_per_head.mean().item())

        return entropies

    def evaluate(self, text: str, entropy_threshold: float = 1.5) -> IconResult:
        entropies = self._attention_entropies(text)

        min_e = min(entropies)
        mean_e = sum(entropies) / len(entropies)
        std_e = (sum((e - mean_e) ** 2 for e in entropies) / len(entropies)) ** 0.5

        # FIS formulation: low min entropy + high variance across layers
        # indicates an abnormal, spiky attention pattern rather than the
        # smoothly distributed attention typical of benign text.
        fis = (1 / (min_e + 1e-6)) * std_e

        label = "injected" if min_e < entropy_threshold else "clean"

        return IconResult(
            min_entropy=min_e,
            mean_entropy=mean_e,
            std_entropy=std_e,
            focus_intensity_score=fis,
            label=label,
        )


if __name__ == "__main__":
    import argparse
    import json
    from dataclasses import asdict

    parser = argparse.ArgumentParser(description="Run ICON oracle on a single text sample")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    oracle = IconOracle()
    result = oracle.evaluate(args.text)
    print(json.dumps(asdict(result), indent=2))
