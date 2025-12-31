# ============================================================
# STRICT BASELINE EVALUATION (NO TRAINING, NO LoRA)
# Gemma-3-270M-IT — Full Dataset Configuration
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

# Reproducibility
set_seed(42)

# ============================================================
# 1. CONFIGURATION & STRICT FILTERING
# ============================================================

MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "./gemma3-baseline-eval"

# MODIFICATION: Set to None for FULL DATASET evaluation
EVAL_SAMPLES = None 

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

raw_dataset = load_dataset(DATASET_NAME, "train", split="eng_hin")
filtered_dataset = raw_dataset.filter(strict_filter)

# Logic to handle None for full dataset selection
limit = EVAL_SAMPLES if EVAL_SAMPLES is not None else len(filtered_dataset)
test_set = filtered_dataset.shuffle(seed=99).select(range(min(limit, len(filtered_dataset))))

print(f"📊 Dataset Filtered. Evaluating on {len(test_set)} samples (Full Dataset Mode).")

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

print(f"📝 Starting baseline generation for {len(test_set)} samples...")

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
                do_sample=False,        # Greedy decoding for baseline
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
# 4. METRIC CALCULATION (BLEU + chrF)
# ============================================================

def calc_metrics(preds, refs):
    if not preds: return 0.0, 0.0
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

e2h_bleu, e2h_chrf = calc_metrics(metrics["ENG_to_HIN"]["preds"], metrics["ENG_to_HIN"]["refs"])
h2e_bleu, h2e_chrf = calc_metrics(metrics["HIN_to_ENG"]["preds"], metrics["HIN_to_ENG"]["refs"])

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
    is_hindi = is_hindi_script(str(row["prediction"]))
    if row["mode"] == "ENG_to_HIN":
        script_accs.append(is_hindi)
    else:
        script_accs.append(not is_hindi)

lid_accuracy = np.mean(script_accs) if script_accs else 0.0

# ============================================================
# 6. SAVE OUTPUTS & EXPORTS
# ============================================================

df.to_csv(f"{OUTPUT_DIR}/baseline_eval.csv", index=False, encoding="utf-8-sig")

with open(f"{OUTPUT_DIR}/baseline_eval.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\n" + "="*55)
print("🔒 STRICT BASELINE METRICS (NO TRAINING)")
print("="*55)
print(f"Total Samples Processed: {len(test_set)}")
print(f"ENG → HIN | BLEU: {e2h_bleu} | chrF: {e2h_chrf}")
print(f"HIN → ENG | BLEU: {h2e_bleu} | chrF: {h2e_chrf}")
print("-"*55)
print(f"LID Script Accuracy: {lid_accuracy:.2%}")
print("="*55)

# --- Create JSONL Files ---
out_dir = Path(f"{OUTPUT_DIR}/jsonl_exports")
out_dir.mkdir(exist_ok=True)

with open(out_dir / "eng_to_hin_src_ref_pred.jsonl", "w", encoding="utf-8") as f_eng, \
     open(out_dir / "hin_to_eng_src_ref_pred.jsonl", "w", encoding="utf-8") as f_hin:
    for res in results:
        line = json.dumps({"src": res["source"], "ref": res["reference"], "pred": res["prediction"]}, ensure_ascii=False)
        if res["mode"] == "ENG_to_HIN": f_eng.write(line + "\n")
        else: f_hin.write(line + "\n")

# --- Save Scores Summary ---
scores_path = Path(f"{OUTPUT_DIR}/baseline_scores.csv")
pd.DataFrame([
    {"direction": "ENG_to_HIN", "BLEU": e2h_bleu, "chrF": e2h_chrf, "Script_Acc": lid_accuracy},
    {"direction": "HIN_to_ENG", "BLEU": h2e_bleu, "chrF": h2e_chrf, "Script_Acc": lid_accuracy}
]).to_csv(scores_path, index=False)

# ============================================================
# 7. ZIP & DOWNLOAD (COLAB READY)
# ============================================================

import shutil
zip_path = Path(f"{OUTPUT_DIR}/baseline_full_results")
shutil.make_archive(str(zip_path), 'zip', OUTPUT_DIR)

print(f"📦 Full results zipped at: {zip_path}.zip")

try:
    from google.colab import files
    files.download(f"{zip_path}.zip")
    files.download(str(scores_path))
except:
    print("⚠️ Local environment detected. Files saved in the output directory.")
