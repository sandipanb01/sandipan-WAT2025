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
import unicodedata
import sys

# ============================================================
# 0. SEED & ENV
# ============================================================
set_seed(42)
torch.set_grad_enabled(False)

# ============================================================
# 1. CONFIG
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
EVAL_CONFIG = "test"

EVAL_SAMPLES = None        # None = full official test set
MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400

OUTPUT_DIR = "baseline_eval_outputs"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ============================================================
# 2. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 3. FILTERS (STRICT, SAME AS TRAINING)
# ============================================================
def strict_filter(example):
    sim = SequenceMatcher(
        None,
        example["src_txt"].lower(),
        example["tgt_txt"].lower()
    ).ratio()
    return sim < 0.65

def length_filter(example):
    src_len = len(tokenizer(example["src_txt"], add_special_tokens=True)["input_ids"])
    tgt_len = len(tokenizer(example["tgt_txt"], add_special_tokens=True)["input_ids"])
    return (src_len <= MAX_SRC_LEN) and (tgt_len <= MAX_TGT_LEN)

# ============================================================
# 4. LOAD OFFICIAL TEST DATA
# ============================================================
eval_dataset = load_dataset(DATASET_NAME, EVAL_CONFIG, split="eng_hin")
eval_dataset = eval_dataset.filter(strict_filter)
eval_dataset = eval_dataset.filter(length_filter)

if EVAL_SAMPLES is not None:
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(EVAL_SAMPLES))

print(f"✅ Evaluation samples: {len(eval_dataset)}")

# ============================================================
# 5. LOAD BASE MODEL (NO TRAINING)
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# ============================================================
# 6. EVALUATION LOOP (STRICT)
# ============================================================
results = []
metrics = {
    "ENG_to_HIN": {"preds": [], "refs": []},
    "HIN_to_ENG": {"preds": [], "refs": []}
}

def is_devanagari(text):
    for ch in text:
        if "DEVANAGARI" in unicodedata.name(ch, ""):
            return True
    return False

lid_correct = []

for sample in tqdm(eval_dataset, desc="Evaluating"):
    pairs = [
        ("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", sample["src_txt"], sample["tgt_txt"]),
        ("HIN_to_ENG", "Translate to ENGLISH:", sample["tgt_txt"], sample["src_txt"]),
    ]

    for mode, instr, src, ref in pairs:
        prompt = (
            "<start_of_turn>user\n"
            f"{instr}\n{src}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_TGT_LEN,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1
            )

        pred_tokens = output[0][inputs.input_ids.shape[-1]:]
        pred = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

        results.append({
            "mode": mode,
            "source": src,
            "reference": ref,
            "prediction": pred
        })

        metrics[mode]["preds"].append(pred)
        metrics[mode]["refs"].append(ref)

        if mode == "ENG_to_HIN":
            lid_correct.append(is_devanagari(pred))
        else:
            lid_correct.append(not is_devanagari(pred))

# ============================================================
# 7. METRICS
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

true_lid_acc = np.mean(lid_correct)

print("\n" + "=" * 50)
print("STRICT ZERO-SHOT BASELINE (NO TRAINING)")
print(f"ENG → HIN | BLEU: {e2h_bleu} | chrF: {e2h_chrf}")
print(f"HIN → ENG | BLEU: {h2e_bleu} | chrF: {h2e_chrf}")
print(f"Strict Script Accuracy: {true_lid_acc:.2%}")
print("=" * 50)

# ============================================================
# 8. SAVE JSON + JSONL
# ============================================================
with open(f"{OUTPUT_DIR}/baseline_eval.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

df = pd.DataFrame(results)

out_dir = Path(f"{OUTPUT_DIR}/jsonl")
out_dir.mkdir(exist_ok=True)

eng_path = out_dir / "eng_to_hin.jsonl"
hin_path = out_dir / "hin_to_eng.jsonl"

with open(eng_path, "w", encoding="utf-8") as fe, \
     open(hin_path, "w", encoding="utf-8") as fh:
    for r in results:
        line = json.dumps(
            {"src": r["source"], "ref": r["reference"], "pred": r["prediction"]},
            ensure_ascii=False
        )
        if r["mode"] == "ENG_to_HIN":
            fe.write(line + "\n")
        else:
            fh.write(line + "\n")

print(f"📁 Outputs saved to: {OUTPUT_DIR}")

# ============================================================
# 9. QUALITATIVE CHECK
# ============================================================
print("\n🧪 TOP-10 SAMPLES\n")
for i in range(min(10, len(df))):
    r = df.iloc[i]
    print(f"[{i+1}] {r['mode']}")
    print("SRC :", r["source"])
    print("REF :", r["reference"])
    print("PRED:", r["prediction"])
    print("-" * 80)
# ADDED ON: # ============================================================
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

print("\n✅ JSONL + ZIP CREATED")
print("📦 translation_outputs.zip")
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

print(f"✅ ENG→HIN JSONL records: {eng_hin_count}")
print(f"✅ HIN→ENG JSONL records: {hin_eng_count}")
print(f"📂 Files saved in: {out_dir.resolve()}")
# ============================================================
# ZIP & DOWNLOAD JSONL FILES (COLAB) #OPTIONAL, MIGHT NOT WORK WITH VS CODE
# ============================================================

import shutil
from google.colab import files

zip_name = "translation_jsonl_outputs"
shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir="exports_jsonl"
)

files.download(f"{zip_name}.zip")
# ============================================================
# UNIVERSAL VISUAL CHECK (Works in Colab & VS Code)
# ============================================================
import pandas as pd
import sys

# 1. Prepare the data
visual_df = df[['mode', 'source', 'reference', 'prediction']].head(10)
visual_df.columns = ['Direction', 'Source Text', 'Ground Truth (Ref)', 'Model Prediction (Pred)']

# 2. Environment-Specific Display
is_colab = 'google.colab' in sys.modules

if is_colab:
    from google.colab import data_table
    data_table.enable_dataframe_formatter()
    print("✨ COLAB MODE: Interactive Table Enabled")
    display(visual_df)
else:
    # VS Code / Terminal Mode
    print("✨ VS CODE MODE: Printing Summary Table")
    # We use to_string() to ensure the terminal doesn't cut off the middle columns
    print(visual_df.to_string(index=False, max_colwidth=50))

# 3. Universal Detailed Text View (Best for comparing scripts/characters)
print("\n" + "═" * 80)
print("📝 DETAILED SAMPLES (Top 10)")
print("═" * 80)

for idx, row in visual_df.iterrows():
    print(f"📍 SAMPLE #{idx+1} | {row['Direction']}")
    print(f"   [SRC]: {row['Source Text']}")
    print(f"   [REF]: {row['Ground Truth (Ref)']}")
    print(f"   [PRED]: {row['Model Prediction (Pred)']}")
    print("-" * 80)
