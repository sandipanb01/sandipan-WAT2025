# First install required packages #EVALUATION CODE#
!pip install sacrebleu>=2.3.0
!pip install pandas openpyxl

# Then run the evaluation
from pathlib import Path
import json
import pandas as pd
from sacrebleu.metrics import CHRF
import os

def _extract(line: str) -> str:
    try:
        obj = json.loads(line) if line.strip().startswith(("[", "{", "\"")) else [line]
        return obj[0] if isinstance(obj, list) else obj.get("translation", "")
    except json.JSONDecodeError:
        return line.strip()
    except Exception:
        return line.strip()

def _load(path: Path) -> List[str]:
    texts = []
    if path.suffix.lower() == '.csv':
        try:
            df = pd.read_csv(path)
            if 'pred_txt' in df.columns:
                texts = df['pred_txt'].astype(str).tolist()
            else:
                print(f"Error: CSV file {path} does not contain a 'pred_txt' column.")
        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"Error reading CSV file {path}: {e}")
    elif path.suffix.lower() == '.jsonl':
        try:
            with path.open(encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        texts.append(_extract(ln))
        except FileNotFoundError:
            print(f"File not found: {path}")
        except Exception as e:
            print(f"Error reading JSONL file {path}: {e}")
    else:
        print(f"Unsupported file format for {path}. Please use .csv or .jsonl")
    return texts

# ---------------------------------------------------------------------------
# EVALUATE ALL LANGUAGE PAIRS
# ---------------------------------------------------------------------------

LANGUAGE_PAIRS = [
    # English to Indic languages
    ("eng", "ben"), ("eng", "guj"), ("eng", "hin"), ("eng", "kan"), 
    ("eng", "mal"), ("eng", "mar"), ("eng", "ori"), ("eng", "pan"),
    ("eng", "tam"), ("eng", "tel"), ("eng", "urd"),
    
    # Indic languages to English (reverse direction)
    ("ben", "eng"), ("guj", "eng"), ("hin", "eng"), ("kan", "eng"),
    ("mal", "eng"), ("mar", "eng"), ("ori", "eng"), ("pan", "eng"),
    ("tam", "eng"), ("tel", "eng"), ("urd", "eng")
]

LANG_NAMES = {
    "eng": "English", "ben": "Bengali", "guj": "Gujarati", "hin": "Hindi",
    "kan": "Kannada", "mal": "Malayalam", "mar": "Marathi", "ori": "Odia",
    "pan": "Punjabi", "tam": "Tamil", "tel": "Telugu", "urd": "Urdu"
}

def evaluate_all_languages():
    """Evaluate all language pairs and save to Excel & CSV"""
    SUBSET = "dev"
    MODEL_TYPE = "it"
    STRATEGY = "sentence"
    N = 10
    
    print("Starting evaluation for ALL language pairs...")
    
    # Create a DataFrame to store all results
    results_df = pd.DataFrame(columns=['Language Pair', 'Direction', 'ChrF Score', 'Model Type', 'Samples', 'File Name'])
    
    for src_lang, tgt_lang in LANGUAGE_PAIRS:
        print(f"\n{'='*60}")
        print(f"Evaluating: {LANG_NAMES[src_lang]} → {LANG_NAMES[tgt_lang]}")
        print(f"{'='*60}")
        
        # Construct file paths
        hyp_file = f"gemma3_1b_{MODEL_TYPE}_{STRATEGY}_{src_lang}_to_{tgt_lang}_{SUBSET}_{N}.csv"
        ref_file = f"/tmp/pralekha_data/{SUBSET}/{src_lang}_{tgt_lang}/doc.{tgt_lang}.jsonl"
        
        print(f"Hypothesis: {hyp_file}")
        print(f"Reference: {ref_file}")
        
        # Load files
        refs = _load(Path(ref_file))
        hyps = _load(Path(hyp_file))
        
        print(f"References: {len(refs)}, Hypotheses: {len(hyps)}")
        
        if not refs or not hyps:
            print("Error loading files. Skipping...")
            continue
            
        if len(refs) != len(hyps):
            print(f"Warning: Length mismatch! refs={len(refs)} vs hyps={len(hyps)}")
            min_len = min(len(refs), len(hyps))
            refs = refs[:min_len]
            hyps = hyps[:min_len]
            print(f"Using first {min_len} samples for evaluation")
        
        # Compute ChrF score
        try:
            score = CHRF().corpus_score(hyps, [refs]).score
            print(f"ChrF Score: {score:.4f}")
            
            # Add to results DataFrame
            new_row = pd.DataFrame({
                'Language Pair': [f"{src_lang}-{tgt_lang}"],
                'Direction': [f"{LANG_NAMES[src_lang]} → {LANG_NAMES[tgt_lang]}"],
                'ChrF Score': [score],
                'Model Type': [MODEL_TYPE],
                'Samples': [len(hyps)],
                'File Name': [hyp_file]
            })
            
            results_df = pd.concat([results_df, new_row], ignore_index=True)
            
        except Exception as e:
            print(f"Error computing ChrF score: {e}")
            continue
    
    return results_df

# ---------------------------------------------------------------------------
# RUN EVALUATION AND SAVE RESULTS
# ---------------------------------------------------------------------------

# Evaluate all language pairs
results_df = evaluate_all_languages()

# Save to CSV file
csv_file = "translation_evaluation_results.csv"
results_df.to_csv(csv_file, index=False)
print(f"\n✓ Results saved to CSV: {csv_file}")

# Save to Excel file
excel_file = "translation_evaluation_results.xlsx"
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    # Main results sheet
    results_df.to_excel(writer, sheet_name='Evaluation Results', index=False)
    
    # Summary statistics sheet
    summary_df = results_df.groupby('Direction')['ChrF Score'].agg(['mean', 'std', 'count']).round(4)
    summary_df.to_excel(writer, sheet_name='Summary Statistics')
    
    # Pivot table sheet
    pivot_df = results_df.pivot_table(
        values='ChrF Score', 
        index='Language Pair', 
        columns='Model Type', 
        aggfunc='mean'
    ).round(4)
    pivot_df.to_excel(writer, sheet_name='Pivot Table')

print(f"✓ Results saved to Excel: {excel_file}")

# Print summary
print(f"\n{'='*60}")
print("EVALUATION SUMMARY:")
print(f"{'='*60}")
print(results_df[['Direction', 'ChrF Score', 'Samples']].to_string(index=False))

print(f"\n{'='*60}")
print("✓ Evaluation completed! Files created:")
print(f"1. {csv_file} (CSV format)")
print(f"2. {excel_file} (Excel format with multiple sheets)")
print(f"{'='*60}")

# Show the files in current directory
print("\nFiles in current directory:")
!ls -la *.csv *.xlsx 2>/dev/null || echo "No CSV or Excel files found"
