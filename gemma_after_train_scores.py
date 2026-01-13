import json
import sacrebleu
import numpy as np
import unicodedata
import pandas as pd  # Added for Excel export
from pathlib import Path

# --- Helper Functions ---

def is_devanagari(text):
    """Checks if a string contains Devanagari characters."""
    if not isinstance(text, str): return False
    for ch in text:
        try:
            if "DEVANAGARI" in unicodedata.name(ch, ""):
                return True
        except ValueError:
            continue
    return False

def load_jsonl(file_path):
    """Loads a JSONL file into a list of dictionaries, skipping malformed lines."""
    data = []
    if not Path(file_path).exists():
        print(f"⚠️ Warning: File {file_path} not found.")
        return data
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Skipping malformed JSON line in {file_path}: {e}")
    return data

def calculate_metrics(records, direction):
    """Calculates BLEU, chrF, and LID accuracy for a given direction."""
    if not records:
        return None

    preds = [r["pred"] for r in records if "pred" in r]
    refs = [r["ref"] for r in records if "ref" in r]

    if not preds or not refs:
        return None

    # Calculate BLEU and chrF using sacrebleu
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    # Script Accuracy (LID)
    lid_results = []
    for p in preds:
        has_devanagari = is_devanagari(p)
        if direction == "ENG_to_HIN":
            # Correct if prediction contains Devanagari
            lid_results.append(has_devanagari)
        else:
            # Correct if prediction does NOT contain Devanagari
            lid_results.append(not has_devanagari)

    lid_acc = np.mean(lid_results) * 100 if lid_results else 0

    return {
        "Direction": direction.replace("_", " "),
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2),
        "LID_Accuracy (%)": round(lid_acc, 2),
        "Sample_Count": len(records)
    }

# --- Main Execution ---

# File paths
e2h_file = "eng_to_hin_src_ref_pred.jsonl"
h2e_file = "hin_to_eng_src_ref_pred.jsonl"

# Load data
e2h_data = load_jsonl(e2h_file)
h2e_data = load_jsonl(h2e_file)

# Compute metrics
e2h_metrics = calculate_metrics(e2h_data, "ENG_to_HIN")
h2e_metrics = calculate_metrics(h2e_data, "HIN_to_ENG")

# 1. Prepare data for Display and Excel
all_results = []
if e2h_metrics: all_results.append(e2h_metrics)
if h2e_metrics: all_results.append(h2e_metrics)

# 2. Display Results in Console
print("="*75)
print(f"{'Direction':<15} | {'BLEU':<8} | {'chrF':<8} | {'LID Acc (%)':<12} | {'Samples'}")
print("-"*75)

for res in all_results:
    print(f"{res['Direction']:<15} | {res['BLEU']:<8} | {res['chrF']:<8} | {res['LID_Accuracy (%)']:<12} | {res['Sample_Count']}")

print("="*75)

# 3. SAVE TO EXCEL
if all_results:
    output_excel = "translation_metrics_summary.xlsx"
    df = pd.DataFrame(all_results)
    
    # Optional: Reorder columns for a cleaner look
    cols = ['Direction', 'BLEU', 'chrF', 'LID_Accuracy (%)', 'Sample_Count']
    df = df[cols]
    
    df.to_excel(output_excel, index=False)
    print(f"\n✅ Metrics successfully saved to: {output_excel}")
else:
    print("\n❌ No data found to save.")
