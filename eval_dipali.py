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
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# 1. GLOBAL CONFIG
# ============================================================
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

BASE_MODEL_ID = "google/gemma-3-270m-it"

CHECKPOINT_ROOT = Path("./gemma3_outputs/checkpoints")
OUTPUT_ROOT = Path("./checkpoint_eval_outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "ai4bharat/Pralekha"
EVAL_SPLIT = "test"
MAX_TGT_LEN = 2400
DEVICE = "cuda"

# ============================================================
# 2. HELPERS
# ============================================================
def is_devanagari(text):
    for ch in text:
        try:
            if "DEVANAGARI" in unicodedata.name(ch):
                return True
        except ValueError:
            continue
    return False

def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

# ============================================================
# 3. LOAD DATA ONCE (OFFICIAL TEST SET)
# ============================================================
print("📥 Loading Pralekha test split...")
dataset = load_dataset(DATASET_NAME, EVAL_SPLIT, split="eng_hin")

# ============================================================
# 4. DISCOVER CHECKPOINTS
# ============================================================
checkpoints = sorted(
    [p for p in CHECKPOINT_ROOT.iterdir() if p.name.startswith("checkpoint-")],
    key=lambda x: int(x.name.split("-")[-1])
)

assert len(checkpoints) > 0, "❌ No checkpoints found"

print(f"✅ Found {len(checkpoints)} checkpoints")

# ============================================================
# 5. LOAD TOKENIZER (ONCE)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 6. MAIN CHECKPOINT EVAL LOOP
# ============================================================
summary_rows = []

for ckpt in checkpoints:
    step = int(ckpt.name.split("-")[-1])
    print(f"\n🚀 Evaluating checkpoint-{step}")

    ckpt_out = OUTPUT_ROOT / ckpt.name
    ckpt_out.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # 6.1 LOAD BASE MODEL + LoRA → MERGE
    # --------------------------------------------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16
    ).to(DEVICE)

    model = PeftModel.from_pretrained(base_model, ckpt)
    model = model.merge_and_unload()
    model.eval()

    # --------------------------------------------------------
    # 6.2 STORAGE
    # --------------------------------------------------------
    results = []
    metrics = {
        "ENG_to_HIN": {"preds": [], "refs": []},
        "HIN_to_ENG": {"preds": [], "refs": []}
    }

    # --------------------------------------------------------
    # 6.3 EVALUATION
    # --------------------------------------------------------
    for sample in tqdm(dataset, desc=f"Step {step}"):
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

            results.append({
                "mode": mode,
                "source": src,
                "reference": ref,
                "prediction": pred
            })

            metrics[mode]["preds"].append(pred)
            metrics[mode]["refs"].append(ref)

    # --------------------------------------------------------
    # 6.4 METRICS
    # --------------------------------------------------------
    e2h_bleu, e2h_chrf = calc_metrics(
        metrics["ENG_to_HIN"]["preds"],
        metrics["ENG_to_HIN"]["refs"]
    )

    h2e_bleu, h2e_chrf = calc_metrics(
        metrics["HIN_to_ENG"]["preds"],
        metrics["HIN_to_ENG"]["refs"]
    )

    summary_rows.append({
        "step": step,
        "ENG_to_HIN_BLEU": e2h_bleu,
        "ENG_to_HIN_chrF": e2h_chrf,
        "HIN_to_ENG_BLEU": h2e_bleu,
        "HIN_to_ENG_chrF": h2e_chrf
    })

    # --------------------------------------------------------
    # 6.5 SAVE JSONL (DIRECTION-WISE)
    # --------------------------------------------------------
    eng_path = ckpt_out / "eng_to_hin.jsonl"
    hin_path = ckpt_out / "hin_to_eng.jsonl"

    with open(eng_path, "w", encoding="utf-8") as fe, \
         open(hin_path, "w", encoding="utf-8") as fh:

        for r in results:
            line = {
                "src": r["source"],
                "ref": r["reference"],
                "pred": r["prediction"]
            }
            if r["mode"] == "ENG_to_HIN":
                fe.write(json.dumps(line, ensure_ascii=False) + "\n")
            else:
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    # Save raw JSON
    with open(ckpt_out / "raw_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------------
    # 6.6 CLEANUP GPU
    # --------------------------------------------------------
    del model
    del base_model
    torch.cuda.empty_cache()

# ============================================================
# 7. SAVE METRICS TABLE
# ============================================================
df = pd.DataFrame(summary_rows).sort_values("step")
csv_path = OUTPUT_ROOT / "checkpoint_metrics.csv"
df.to_csv(csv_path, index=False)

print(f"\n📊 Metrics CSV saved → {csv_path}")

# ============================================================
# 8. PLOTS
# ============================================================
plt.figure()
plt.plot(df["step"], df["ENG_to_HIN_BLEU"], label="ENG→HIN BLEU")
plt.plot(df["step"], df["HIN_to_ENG_BLEU"], label="HIN→ENG BLEU")
plt.xlabel("Training Step")
plt.ylabel("BLEU")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "bleu_vs_steps.png")
plt.close()

plt.figure()
plt.plot(df["step"], df["ENG_to_HIN_chrF"], label="ENG→HIN chrF")
plt.plot(df["step"], df["HIN_to_ENG_chrF"], label="HIN→ENG chrF")
plt.xlabel("Training Step")
plt.ylabel("chrF")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "chrf_vs_steps.png")
plt.close()

print("📈 Saved plots:")
print(" • bleu_vs_steps.png")
print(" • chrf_vs_steps.png")

print("\n✅ ALL CHECKPOINTS EVALUATED (LoRA MERGED)")
