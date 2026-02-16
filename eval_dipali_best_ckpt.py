# ============================================================
# 0. IMPORTS
# ============================================================
import os
import json
import torch
import sacrebleu
import numpy as np
import pandas as pd
import unicodedata

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# 1. GLOBAL CONFIG (MATCH TRAINING)
# ============================================================
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

BASE_MODEL_ID = "google/gemma-3-270m-it"

OUTPUT_ROOT = Path("./gemma3_outputs")
CKPT_ROOT   = OUTPUT_ROOT / "checkpoints"
EVAL_OUT    = OUTPUT_ROOT / "final_eval_outputs"
EVAL_OUT.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "ai4bharat/Pralekha"
DEV_SPLIT  = "dev"
TEST_SPLIT = "test"
MAX_TGT_LEN = 2400
DEVICE = "cuda"

# ============================================================
# 2. HELPERS (UNCHANGED)
# ============================================================
def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

# ============================================================
# 3. LOAD DEV LOGS → SELECT BEST CKPT
# ============================================================
print("📊 Selecting BEST checkpoint using DEV eval_loss...")

trainer_state_path = CKPT_ROOT / "trainer_state.json"
assert trainer_state_path.exists(), "❌ trainer_state.json not found"

with open(trainer_state_path) as f:
    state = json.load(f)

eval_entries = [
    x for x in state["log_history"]
    if "eval_loss" in x and "step" in x
]

assert len(eval_entries) > 0, "❌ No eval logs found"

best_entry = min(eval_entries, key=lambda x: x["eval_loss"])
BEST_STEP = best_entry["step"]
BEST_CKPT = CKPT_ROOT / f"checkpoint-{BEST_STEP}"

assert BEST_CKPT.exists(), f"❌ {BEST_CKPT} not found"

print(f"🏆 BEST CHECKPOINT → checkpoint-{BEST_STEP}")
print(f"📉 Lowest DEV loss → {best_entry['eval_loss']:.4f}")

# ============================================================
# 4. LOAD TEST SET (ONCE)
# ============================================================
print("📥 Loading Pralekha TEST split...")
test_ds = load_dataset(DATASET_NAME, TEST_SPLIT, split="eng_hin")

# ============================================================
# 5. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 6. LOAD BASE + LORA → MERGE
# ============================================================
print("🔄 Loading & merging BEST checkpoint...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16
).to(DEVICE)

model = PeftModel.from_pretrained(base_model, BEST_CKPT)
model = model.merge_and_unload()
model.eval()

# ============================================================
# 7. FINAL TEST EVALUATION
# ============================================================
metrics = {
    "ENG_to_HIN": {"preds": [], "refs": []},
    "HIN_to_ENG": {"preds": [], "refs": []}
}

results = []

print("🚀 Running FINAL TEST evaluation...")

for sample in tqdm(test_ds, desc="Final Test Eval"):
    pairs = [
        ("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", sample["src_txt"], sample["tgt_txt"]),
        ("HIN_to_ENG", "Translate to ENGLISH:", sample["tgt_txt"], sample["src_txt"]),
    ]

    for mode, instr, src, ref in pairs:
        prompt = (
            f"<start_of_turn>user\n{instr}\n{src}"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_TGT_LEN,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1
            )

        pred = tokenizer.decode(
            output[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        metrics[mode]["preds"].append(pred)
        metrics[mode]["refs"].append(ref)

        results.append({
            "mode": mode,
            "src": src,
            "ref": ref,
            "pred": pred
        })

# ============================================================
# 8. METRICS
# ============================================================
e2h_bleu, e2h_chrf = calc_metrics(
    metrics["ENG_to_HIN"]["preds"],
    metrics["ENG_to_HIN"]["refs"]
)

h2e_bleu, h2e_chrf = calc_metrics(
    metrics["HIN_to_ENG"]["preds"],
    metrics["HIN_to_ENG"]["refs"]
)

summary = {
    "best_checkpoint": f"checkpoint-{BEST_STEP}",
    "dev_eval_loss": round(best_entry["eval_loss"], 4),
    "ENG_to_HIN_BLEU": e2h_bleu,
    "ENG_to_HIN_chrF": e2h_chrf,
    "HIN_to_ENG_BLEU": h2e_bleu,
    "HIN_to_ENG_chrF": h2e_chrf
}

# ============================================================
# 9. SAVE OUTPUTS
# ============================================================
with open(EVAL_OUT / "final_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)

with open(EVAL_OUT / "raw_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n✅ FINAL EVALUATION COMPLETE")
print(f"🏆 Best checkpoint: checkpoint-{BEST_STEP}")
print(f"ENG→HIN | BLEU: {e2h_bleu} | chrF: {e2h_chrf}")
print(f"HIN→ENG | BLEU: {h2e_bleu} | chrF: {h2e_chrf}")
print(f"📁 Outputs saved → {EVAL_OUT}")
