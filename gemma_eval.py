how is this eval script: import torch
import json
import pandas as pd
import numpy as np
import unicodedata
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sacrebleu

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

OUTPUT_DIR = "final_eval_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = "./gemma3-4b-train/final_merged"
DATASET_NAME = "ai4bharat/Pralekha"
EVAL_CONFIG = "test"
MAX_TGT_LEN = 512

# ===============================
# LOAD MODEL (SINGLE GPU)
# ===============================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16
).to("cuda")

model.eval()
# ===============================
# DATA
# ===============================
dataset = load_dataset(DATASET_NAME, EVAL_CONFIG, split="eng_hin")

# ===============================
# EVALUATION
# ===============================
results = []
metrics = {"ENG_to_HIN": {"preds": [], "refs": []},
           "HIN_to_ENG": {"preds": [], "refs": []}}

for sample in tqdm(dataset):
    pairs = [
        ("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", sample["src_txt"], sample["tgt_txt"]),
        ("HIN_to_ENG", "Translate to ENGLISH:", sample["tgt_txt"], sample["src_txt"]),
    ]

    for mode, instr, src, ref in pairs:
        prompt = f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_TGT_LEN,
                temperature=0.1,
                do_sample=False,
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

# ============================================================
# >>> FIX START: LID + JSON
# ============================================================
import unicodedata

def is_devanagari(text):
    for ch in text:
        if "DEVANAGARI" in unicodedata.name(ch, ""):
            return True
    return False

lid_correct = []
for r in results:
    if r["mode"] == "ENG_to_HIN":
        lid_correct.append(is_devanagari(r["prediction"]))
    else:
        lid_correct.append(not is_devanagari(r["prediction"]))

true_lid_acc = np.mean(lid_correct)

with open("final_eval_strict.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

df = pd.DataFrame(results)
# >>> FIX END
# ============================================================

def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

e2h_bleu, e2h_chrf = calc_metrics(metrics["ENG_to_HIN"]["preds"], metrics["ENG_to_HIN"]["refs"])
h2e_bleu, h2e_chrf = calc_metrics(metrics["HIN_to_ENG"]["preds"], metrics["HIN_to_ENG"]["refs"])

print("\n" + "="*50)
print("STRICT PAPER-STANDARD METRICS")
print(f"ENG → HIN | BLEU: {e2h_bleu} | chrF: {e2h_chrf}")
print(f"HIN → ENG | BLEU: {h2e_bleu} | chrF: {h2e_chrf}")
print("="*50)
print(f"Strict Script Accuracy: {true_lid_acc:.2%}")
print("="*50)

# ============================================================
# TOP-10 QUALITATIVE (SRC / REF / PRED)
# ============================================================
print("\n TOP-10 DETAILED SAMPLES\n")
for i in range(min(10, len(df))):
    r = df.iloc[i]
    print(f"[{i+1}] {r['mode']}")
    print("SRC :", r["source"])
    print("REF :", r["reference"])
    print("PRED:", r["prediction"])
    print("-" * 80)

# ============================================================
# EXPORT JSONL + ZIP
# ============================================================
out_dir = Path("exports_jsonl")
out_dir.mkdir(exist_ok=True)

eng_path = out_dir / "eng_to_hin_src_ref_pred.jsonl"
hin_path = out_dir / "hin_to_eng_src_ref_pred.jsonl"

with open(eng_path, "w", encoding="utf-8") as fe, open(hin_path, "w", encoding="utf-8") as fh:
    for r in results:
        line = json.dumps(
            {"src": r["source"], "ref": r["reference"], "pred": r["prediction"]},
            ensure_ascii=False
        )
        if r["mode"] == "ENG_to_HIN":
            fe.write(line + "\n")
        else:
            fh.write(line + "\n")

import shutil
shutil.make_archive("translation_outputs", "zip", out_dir)

print("\n JSONL + ZIP CREATED")
print("translation_outputs.zip")
# ============================================================
# CREATE CLEAN JSONL FILES (SRC / REF / PRED ONLY)
# ============================================================

import json
from pathlib import Path

# Load your final evaluation JSON
with open("final_eval_strict.json", "r", encoding="utf-8") as f:
    records = json.load(f)

out_dir = Path("exports_jsonl")
out_dir.mkdir(exist_ok=True)

eng_hin_path = out_dir / "eng_to_hin_src_ref_pred.jsonl"
hin_eng_path = out_dir / "hin_to_eng_src_ref_pred.jsonl"

eng_hin_count = 0
hin_eng_count = 0

with open(eng_hin_path, "w", encoding="utf-8") as f_eng, \
     open(hin_eng_path, "w", encoding="utf-8") as f_hin:

    for r in records:
        line = {
            "src": r["source"],
            "ref": r["reference"],
            "pred": r["prediction"]
        }

        if r["mode"] == "ENG_to_HIN":
            f_eng.write(json.dumps(line, ensure_ascii=False) + "\n")
            eng_hin_count += 1

        elif r["mode"] == "HIN_to_ENG":
            f_hin.write(json.dumps(line, ensure_ascii=False) + "\n")
            hin_eng_count += 1

print(f"ENG→HIN JSONL records: {eng_hin_count}")
print(f"HIN→ENG JSONL records: {hin_eng_count}")
print(f"Files saved in: {out_dir.resolve()}")
# ============================================================
# FINAL ANALYSIS CELL: LOSS CURVE + METRICS REPORT
# ============================================================

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sacrebleu
import unicodedata



# ------------------------------------------------------------
# 2. METRICS + SCRIPT (LID) ACCURACY
# ------------------------------------------------------------
def is_devanagari(text):
    if not text:
        return False
    for ch in text:
        try:
            if "DEVANAGARI" in unicodedata.name(ch):
                return True
        except ValueError:
            continue
    return False

summary_rows = []
directions = ["ENG_to_HIN", "HIN_to_ENG"]

for direction in directions:
    subset = [r for r in results if r["mode"] == direction]

    if not subset:
        continue

    preds = [r["prediction"] for r in subset]
    refs  = [r["reference"] for r in subset]

    bleu_score = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf_score = sacrebleu.corpus_chrf(preds, [refs]).score

    lid_hits = []
    for p in preds:
        has_dev = is_devanagari(p)
        correct = has_dev if direction == "ENG_to_HIN" else not has_dev
        lid_hits.append(correct)

    script_acc = np.mean(lid_hits) * 100 if lid_hits else 0.0

    summary_rows.append({
        "Direction": direction.replace("_", " "),
        "BLEU": round(bleu_score, 2),
        "chrF": round(chrf_score, 2),
        "Script Accuracy (%)": round(script_acc, 2),
        "Total Samples": len(subset)
    })

# ------------------------------------------------------------
# 3. SAVE METRICS REPORT TO EXCEL
# ------------------------------------------------------------
metrics_df = pd.DataFrame(summary_rows)

excel_path = os.path.join(OUTPUT_DIR, "final_translation_metrics.xlsx")
metrics_df.to_excel(excel_path, index=False)

print("\n" + "═" * 70)
print(f"📊 Final metrics report saved to: {excel_path}")
print("═" * 70)
print(metrics_df.to_string(index=False))
print("═" * 70)
