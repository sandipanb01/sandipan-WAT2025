# ============================================================
# STRICT BASELINE EVALUATION (NO TRAINING, NO LoRA)
# Gemma-3-270M-IT — Paper-Compliant
# ============================================================

import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import sacrebleu

# --- Dependencies Guard ---
def install_and_import(package):
    import subprocess, sys
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_and_import("langdetect")
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42

# Reproducibility (matches your script)
set_seed(42)

# ============================================================
# 1. CONFIGURATION & STRICT FILTERING
# ============================================================

MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "./gemma3-baseline-eval"
EVAL_SAMPLES = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

def strict_filter(example):
    """
    Prevents copy-overlap between src and tgt
    (anti-cheating sanity filter)
    """
    sim = SequenceMatcher(
        None,
        example["src_txt"].lower(),
        example["tgt_txt"].lower()
    ).ratio()
    return sim < 0.65

# NOTE: Using dataset split exactly as you did
raw_dataset = load_dataset(DATASET_NAME, "train", split="eng_hin")
filtered_dataset = raw_dataset.filter(strict_filter)

test_set = filtered_dataset.shuffle(seed=99).select(
    range(min(EVAL_SAMPLES, len(filtered_dataset)))
)

# ============================================================
# 2. MODEL & TOKENIZER (BASELINE ONLY)
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="auto"
)
model.eval()

# ============================================================
# 3. STRICT BASELINE EVALUATION
# ============================================================

results = []
metrics = {
    "ENG_to_HIN": {"preds": [], "refs": []},
    "HIN_to_ENG": {"preds": [], "refs": []},
}

print(f"📝 Evaluating {len(test_set)} baseline samples...")

for sample in tqdm(test_set):
    pairs = [
        ("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", sample["src_txt"], sample["tgt_txt"]),
        ("HIN_to_ENG", "Translate to ENGLISH:", sample["tgt_txt"], sample["src_txt"]),
    ]

    for mode, instr, src, ref in pairs:
        prompt = (
            f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,        # greedy decoding
                temperature=0.1,
                repetition_penalty=1.1
            )

        pred = tokenizer.decode(
            output[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        try:
            lid = detect(pred)
        except:
            lid = "unknown"

        results.append({
            "mode": mode,
            "source": src,
            "reference": ref,
            "prediction": pred,
            "lid": lid
        })

        metrics[mode]["preds"].append(pred)
        metrics[mode]["refs"].append(ref)

# ============================================================
# 4. BLEU + chrF
# ============================================================

def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

e2h_bleu, e2h_chrf = calc_metrics(
    metrics["ENG_to_HIN"]["preds"],
    metrics["ENG_to_HIN"]["refs"]
)

h2e_bleu, h2e_chrf = calc_metrics(
    metrics["HIN_to_ENG"]["preds"],
    metrics["HIN_to_ENG"]["refs"]
)

# ============================================================
# 5. SCRIPT-BASED LID CHECK
# ============================================================

import unicodedata

def is_hindi_script(text):
    for ch in text:
        if "DEVANAGARI" in unicodedata.name(ch, ""):
            return True
    return False

df = pd.DataFrame(results)

script_accs = []
for _, row in df.iterrows():
    if row["mode"] == "ENG_to_HIN":
        script_accs.append(is_hindi_script(row["prediction"]))
    else:
        script_accs.append(not is_hindi_script(row["prediction"]))

lid_accuracy = np.mean(script_accs)

# ============================================================
# 6. SAVE OUTPUTS
# ============================================================

df.to_csv(f"{OUTPUT_DIR}/baseline_eval.csv", index=False, encoding="utf-8-sig")

with open(f"{OUTPUT_DIR}/baseline_eval.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "="*55)
print("🔒 STRICT BASELINE METRICS (NO TRAINING)")
print("="*55)
print(f"ENG → HIN | BLEU: {e2h_bleu} | chrF: {e2h_chrf}")
print(f"HIN → ENG | BLEU: {h2e_bleu} | chrF: {h2e_chrf}")
print("-"*55)
print(f"LID Script Accuracy: {lid_accuracy:.2%}")
print("="*55)

# Qualitative sanity sample
sample = df.iloc[0]
print("\n[SAMPLE OUTPUT]")
print("SRC :", sample["source"][:80])
print("PRED:", sample["prediction"][:80])
# ============================================================
# 🔍 TOP-10 QUALITATIVE REVIEWS (SRC / REF / PRED)
# ============================================================

import pandas as pd

df = pd.read_csv(f"{OUTPUT_DIR}/baseline_eval.csv")

print("\n" + "="*70)
print("🔎 TOP 10 QUALITATIVE SAMPLES — ENG → HIN")
print("="*70)

e2h = df[df["mode"] == "ENG_to_HIN"].head(10)

for i, row in enumerate(e2h.itertuples(), 1):
    print(f"\n[{i}]")
    print("SRC :", row.source[:200])
    print("REF :", row.reference[:200])
    print("PRED:", row.prediction[:200])
    print("-"*70)

print("\n" + "="*70)
print("🔎 TOP 10 QUALITATIVE SAMPLES — HIN → ENG")
print("="*70)

h2e = df[df["mode"] == "HIN_to_ENG"].head(10)

for i, row in enumerate(h2e.itertuples(), 1):
    print(f"\n[{i}]")
    print("SRC :", row.source[:200])
    print("REF :", row.reference[:200])
    print("PRED:", row.prediction[:200])
    print("-"*70)
# ============================================================
# 📄 CREATE JSONL FILES (SRC / REF / PRED ONLY)
# ============================================================

import json
from pathlib import Path

out_dir = Path(f"{OUTPUT_DIR}/jsonl_exports")
out_dir.mkdir(exist_ok=True)

eng_hin_path = out_dir / "eng_to_hin_src_ref_pred.jsonl"
hin_eng_path = out_dir / "hin_to_eng_src_ref_pred.jsonl"

eng_hin_count = 0
hin_eng_count = 0

with open(eng_hin_path, "w", encoding="utf-8") as f_eng, \
     open(hin_eng_path, "w", encoding="utf-8") as f_hin:

    for row in df.itertuples():
        record = {
            "src": row.source,
            "ref": row.reference,
            "pred": row.prediction
        }

        if row.mode == "ENG_to_HIN":
            f_eng.write(json.dumps(record, ensure_ascii=False) + "\n")
            eng_hin_count += 1

        elif row.mode == "HIN_to_ENG":
            f_hin.write(json.dumps(record, ensure_ascii=False) + "\n")
            hin_eng_count += 1

print(f"✅ ENG→HIN JSONL records: {eng_hin_count}")
print(f"✅ HIN→ENG JSONL records: {hin_eng_count}")
print(f"📂 Saved in: {out_dir.resolve()}")
# ============================================================
# 📊 SAVE FINAL METRICS AS CSV + COLAB DOWNLOAD
# ============================================================

import pandas as pd
from pathlib import Path

scores_path = Path(f"{OUTPUT_DIR}/baseline_scores.csv")

scores_df = pd.DataFrame([
    {
        "model": MODEL_ID,
        "eval_samples": EVAL_SAMPLES,
        "direction": "ENG_to_HIN",
        "BLEU": e2h_bleu,
        "chrF": e2h_chrf,
        "LID_script_accuracy": lid_accuracy
    },
    {
        "model": MODEL_ID,
        "eval_samples": EVAL_SAMPLES,
        "direction": "HIN_to_ENG",
        "BLEU": h2e_bleu,
        "chrF": h2e_chrf,
        "LID_script_accuracy": lid_accuracy
    }
])

scores_df.to_csv(scores_path, index=False)

print(f"✅ Scores CSV saved at: {scores_path.resolve()}")

# --- Colab download ---
try:
    from google.colab import files
    files.download(str(scores_path))
except ImportError:
    print("⚠️ google.colab not available — CSV saved locally only.")

# ============================================================
# ZIP & DOWNLOAD JSONL FILES (OPTIONAL AND FOR COLAB ONLY)
# ============================================================

import shutil
from pathlib import Path

# Directory where JSONL files were saved
jsonl_dir = Path(f"{OUTPUT_DIR}/jsonl_exports")
zip_path = Path(f"{OUTPUT_DIR}/jsonl_exports.zip")

# Create zip archive
shutil.make_archive(
    base_name=str(zip_path).replace(".zip", ""),
    format="zip",
    root_dir=jsonl_dir
)

print(f"📦 ZIP created at: {zip_path.resolve()}")

# --- Colab download ---
try:
    from google.colab import files
    files.download(str(zip_path))
except ImportError:
    print("⚠️ google.colab not available — ZIP saved locally only.")
