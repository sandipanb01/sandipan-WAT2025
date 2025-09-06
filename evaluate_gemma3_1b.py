#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_chrf.py  –  Compute ChrF for WAT JSONL outputs.

Example
-------
# print to console only
python eval_chrf.py refs.jsonl hyps.jsonl

# also append to a TSV file
python eval_chrf.py refs.jsonl hyps.jsonl --out scores.tsv
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List

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

# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="Reference JSONL")
    ap.add_argument("--hyp", help="System output JSONL")
    ap.add_argument("--out", help="Optional file to append the score (TSV)")
    args = ap.parse_args()

    refs = _load(Path(args.ref))
    hyps = _load(Path(args.hyp))
    if len(refs) != len(hyps):
        sys.exit(f"[ERROR] Length mismatch: refs={len(refs)} vs hyps={len(hyps)}")

    score = CHRF().corpus_score(hyps, [refs]).score
    result_line = f"{Path(args.hyp).name}\t{score:.4f}"

    print(result_line)

    # optional file output
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(result_line + "\n")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
