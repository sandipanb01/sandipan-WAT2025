#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate WAT2025 IndicDoc outputs with ChrF and BLEU.
- Loops over all 11 pairs
- Collects scores
- Saves to TSV, CSV, and Excel
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List
import pandas as pd

try:
    from sacrebleu.metrics import CHRF, BLEU
except ImportError:
    sys.exit("Please `pip install sacrebleu>=2.3` first.")

# ---------------------------------------------------------------------------
PAIRS = [
    "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal",
    "eng_mar", "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
]

# ---------------------------------------------------------------------------
def _extract(line: str) -> str:
    obj = json.loads(line) if line.strip().startswith(("[", "{", "\"")) else [line]
    return obj[0] if isinstance(obj, list) else obj.get("translation", "")

def _load(path: Path) -> List[str]:
    with path.open(encoding="utf-8") as f:
        return [_extract(ln).strip() for ln in f if ln.strip()]

# ---------------------------------------------------------------------------
def evaluate_all(data_root: Path, output_root: Path, out_prefix: str):
    scores = []
    chrf_metric = CHRF()
    bleu_metric = BLEU()

    for pair in PAIRS:
        src, tgt = pair.split("_")
        ref_file = data_root / "dev" / pair / f"doc.{tgt}.jsonl"
        hyp_file = output_root / f"{pair}.gemma.jsonl"

        if not ref_file.exists() or not hyp_file.exists():
            print(f"[!] Skipping {pair} (missing files)")
            continue

        refs = _load(ref_file)
        hyps = _load(hyp_file)

        if len(refs) != len(hyps):
            print(f"[!] Length mismatch for {pair}: refs={len(refs)} vs hyps={len(hyps)}")
            continue

        chrf = chrf_metric.corpus_score(hyps, [refs]).score
        bleu = bleu_metric.corpus_score(hyps, [refs]).score

        scores.append((pair, chrf, bleu))
        print(f"{pair}\tChrF={chrf:.4f}\tBLEU={bleu:.2f}")

    # Save results
    df = pd.DataFrame(scores, columns=["pair", "chrf", "bleu"])
    out_tsv = output_root / f"{out_prefix}.tsv"
    out_csv = output_root / f"{out_prefix}.csv"
    out_xlsx = output_root / f"{out_prefix}.xlsx"

    df.to_csv(out_tsv, sep="\t", index=False)
    df.to_csv(out_csv, index=False)
    df.to_excel(out_xlsx, index=False)

    print(f"\n✓ Results saved: {out_tsv}, {out_csv}, {out_xlsx}")

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data", help="Root directory of Pralekha dataset")
    ap.add_argument("--output_root", default="./outputs", help="Directory containing system outputs")
    ap.add_argument("--out_prefix", default="scores", help="Prefix for results file")
    args = ap.parse_args()

    evaluate_all(Path(args.data_root), Path(args.output_root), args.out_prefix)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
