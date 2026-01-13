import json
import sacrebleu
import numpy as np
import unicodedata
import pandas as pd
from pathlib import Path

# --- Configuration ---
INPUT_FILE = "baseline_eval.json"

# --- Helper Functions ---

def is_devanagari(text):
    """Checks if a string contains any Devanagari characters."""
    if not text: return False
    for ch in text:
        try:
            if "DEVANAGARI" in unicodedata.name(ch, ""):
                return True
        except ValueError:
            continue
    return False

def calculate_metrics(records, direction):
    """Calculates BLEU, chrF, and LID accuracy for a subset of records."""
    if not records:
        return None

    # Baseline predictions often include preambles like "Here is the translation:"
    # We evaluate them as-is to see the raw performance.
    preds = [r["prediction"] for r in records]
    refs = [r["reference"] for r in records]

    # Calculate BLEU and chrF
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    # Script Accuracy (LID)
    lid_results = []
    for p in preds:
        has_devanagari = is_devanagari(p)
        if direction == "ENG_to_HIN":
            # Pass if prediction contains Devanagari
            lid_results.append(has_devanagari)
        else:
            # Pass if prediction does NOT contain Devanagari (English/Latin)
            lid_results.append(not has_devanagari)
    
    lid_acc = np.mean(lid_results) * 100 if lid_results else 0
    
    return {
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2),
        "LID_Accuracy": round(lid_acc, 2),
        "Count": len(records)
    }

# --- Main Execution ---

if not Path(INPUT_FILE).exists():
    print(f"❌ Error: File '{INPUT_FILE}' not found in the current directory.")
else:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Split by mode
    e2h_records = [r for r in data if r["mode"] == "ENG_to_HIN"]
    h2e_records = [r for r in data if r["mode"] == "HIN_to_ENG"]

    # Compute metrics
    e2h_metrics = calculate_metrics(e2h_records, "ENG_to_HIN")
    h2e_metrics = calculate_metrics(h2e_records, "HIN_to_ENG")

    # Display Results
    print("\n" + "="*65)
    print(f"{'Direction':<15} | {'BLEU':<8} | {'chrF':<8} | {'LID Acc (%)':<12} | {'Samples'}")
    print("-"*65)

    if e2h_metrics:
        print(f"{'ENG → HIN':<15} | {e2h_metrics['BLEU']:<8} | {e2h_metrics['chrF']:<8} | {e2h_metrics['LID_Accuracy']:<12} | {e2h_metrics['Count']}")
    else:
        print(f"{'ENG → HIN':<15} | No data found.")

    if h2e_metrics:
        print(f"{'HIN → ENG':<15} | {h2e_metrics['BLEU']:<8} | {h2e_metrics['chrF']:<8} | {h2e_metrics['LID_Accuracy']:<12} | {h2e_metrics['Count']}")
    else:
        print(f"{'HIN → ENG':<15} | No data found.")

    print("="*65)
    
    # Optional: Brief Analysis
    avg_bleu = np.mean([m['BLEU'] for m in [e2h_metrics, h2e_metrics] if m])
    print(f"\n💡 Baseline Average BLEU: {avg_bleu:.2f}")
    if any(m['LID_Accuracy'] < 80 for m in [e2h_metrics, h2e_metrics] if m):
        print("⚠️ Note: Low LID Accuracy suggests the model is responding in the wrong language or mixing scripts.")
