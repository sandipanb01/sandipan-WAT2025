import json
import sacrebleu
import numpy as np
import unicodedata
from pathlib import Path

# --- Helper Functions ---

def is_devanagari(text):
    """Checks if a string contains Devanagari characters."""
    for ch in text:
        if "DEVANAGARI" in unicodedata.name(ch, ""):
            return True
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
                print(f"⚠️ Warning: Skipping malformed JSON line in {file_path}: {e} - Line: {line.strip()[:100]}...")
    return data

def calculate_metrics(records, direction):
    """Calculates BLEU, chrF, and LID accuracy for a given direction."""
    if not records:
        return None

    preds = [r["pred"] for r in records]
    refs = [r["ref"] for r in records]

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
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2),
        "LID_Accuracy": round(lid_acc, 2)
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

# Display Results
print("="*60)
print(f"{'Direction':<15} | {'BLEU':<8} | {'chrF':<8} | {'LID Acc (%)':<12}")
print("-"*60)

if e2h_metrics:
    print(f"{'ENG → HIN':<15} | {e2h_metrics['BLEU']:<8} | {e2h_metrics['chrF']:<8} | {e2h_metrics['LID_Accuracy']:<12}")
else:
    print(f"{'ENG → HIN':<15} | No data found.")

if h2e_metrics:
    print(f"{'HIN → ENG':<15} | {h2e_metrics['BLEU']:<8} | {h2e_metrics['chrF']:<8} | {h2e_metrics['LID_Accuracy']:<12}")
else:
    print(f"{'HIN → ENG':<15} | No data found.")

print("="*60)
