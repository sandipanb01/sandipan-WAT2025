#!/usr/bin/env python   #EVALUATION FOR ALL LANGUAGE PAIR#
# -*- coding: utf-8 -*-
"""
eval_chrf.py  –  Compute ChrF for WAT JSONL outputs for all language pairs.
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Dict
import pandas as pd

try:
    from sacrebleu.metrics import CHRF
except ImportError:
    sys.exit("Please `pip install sacrebleu>=2.3` first.")

# ---------------------------------------------------------------------------
def _extract(line: str) -> str:
    obj = json.loads(line) if line.strip().startswith(("[", "{", "\"")) else [line]
    return obj[0] if isinstance(obj, list) else obj.get("translation", "")

def _load(path: Path) -> List[str]:
    with path.open(encoding="utf-8") as f:
        return [_extract(ln).strip() for ln in f if ln.strip()]

def evaluate_all_pairs(ref_dir: Path, hyp_dir: Path, output_file: Path):
    """Evaluate all language pairs and save results to Excel/CSV."""

    LANGUAGE_PAIRS = [
        "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal",
        "eng_mar", "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
    ]

    results = []

    for pair in LANGUAGE_PAIRS:
        src_lang, tgt_lang = pair.split("_")

        # Reference file
        ref_file = ref_dir / pair / f"doc.{tgt_lang}.jsonl"

        # Hypothesis file (assuming naming convention)
        hyp_file = hyp_dir / f"translations_{src_lang}_{tgt_lang}_dev.csv"

        if not ref_file.exists():
            print(f"Warning: Reference file not found: {ref_file}")
            continue

        if not hyp_file.exists():
            print(f"Warning: Hypothesis file not found: {hyp_file}")
            continue

        try:
            # Load references
            refs = _load(ref_file)

            # Load hypotheses from CSV
            df = pd.read_csv(hyp_file)
            hyps = df["pred_txt"].tolist()

            if len(refs) != len(hyps):
                print(f"Warning: Length mismatch for {pair}: refs={len(refs)} vs hyps={len(hyps)}")
                # Truncate or pad the shorter list to match the length of the longer list
                min_len = min(len(refs), len(hyps))
                refs = refs[:min_len]
                hyps = hyps[:min_len]
                print(f"Adjusted lengths to {min_len} for {pair}")


            # Calculate ChrF score
            score = CHRF().corpus_score(hyps, [refs]).score

            results.append({
                "language_pair": pair,
                "chrf_score": score,
                "num_samples": len(refs)
            })

            print(f"{pair}: ChrF = {score:.4f} (n={len(refs)})")

        except Exception as e:
            print(f"Error processing {pair}: {e}")
            continue

    # Save results
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_excel(output_file.with_suffix(".xlsx"), index=False)
        df_results.to_csv(output_file.with_suffix(".csv"), index=False)
        print(f"✓ Results saved to {output_file.with_suffix('.xlsx')} and {output_file.with_suffix('.csv')}")
    else:
        print("No results to save")

# ---------------------------------------------------------------------------
def main() -> None:
    # Check if running in Colab and provide default args
    if 'google.colab' in sys.modules:
        # Define default values for Colab environment
        default_ref_dir = Path("./pralekha_data/dev") # Example reference directory
        default_hyp_dir = Path(".") # Example hypothesis directory (assuming CSVs are in the current directory)
        default_output_file = Path("./evaluation_results") # Example output file path

        # Create a dummy parser to get default values
        p = argparse.ArgumentParser()
        p.add_argument("--ref_dir", type=Path, default=default_ref_dir)
        p.add_argument("--hyp_dir", type=Path, default=default_hyp_dir)
        p.add_argument("--output_file", type=Path, default=default_output_file)
        args = p.parse_args([]) # Parse empty args to get defaults

        ref_dir = args.ref_dir
        hyp_dir = args.hyp_dir
        output_file = args.output_file
    else:
        # Use argparse for command-line execution
        ap = argparse.ArgumentParser()
        ap.add_argument("--ref_dir", required=True, type=Path, help="Directory containing reference JSONL files")
        ap.add_argument("--hyp_dir", required=True, type=Path, help="Directory containing hypothesis files")
        ap.add_argument("--output_file", required=True, type=Path, help="Output file path (without extension)")
        args = ap.parse_args()

        ref_dir = args.ref_dir
        hyp_dir = args.hyp_dir
        output_file = args.output_file


    evaluate_all_pairs(ref_dir, hyp_dir, output_file)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
