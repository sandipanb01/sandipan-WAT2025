#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate few-shot prompts for Gemma 3 models.
Output: doc.{src}_2_{tgt}.{k}.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List

LANG_LABELS = {
    "eng": "English",   "ben": "Bengali",   "guj": "Gujarati",
    "hin": "Hindi",     "kan": "Kannada",   "mal": "Malayalam",
    "mar": "Marathi",   "ori": "Odia",      "pan": "Punjabi",
    "tam": "Tamil",     "tel": "Telugu",    "urd": "Urdu",
}

# ───────────────────────── helper funcs ─────────────────────────
def _extract_text(line: str) -> str:
    obj = json.loads(line)
    if isinstance(obj, list):
        return str(obj[0])
    if isinstance(obj, str):
        return obj
    raise ValueError("Line must be string or single-element list.")

def _load_texts(path: Path) -> List[str]:
    with path.open(encoding="utf-8") as f:
        return [_extract_text(ln).strip() for ln in f if ln.strip()]

def _build_block(src, tgt, src_lbl, tgt_lbl):
    return f"<start_of_turn>user\nTranslate this {src_lbl} text to {tgt_lbl}:\n\n{src}<end_of_turn>\n<start_of_turn>model\n{tgt}<end_of_turn>"

def _build_final_prompt(src, src_lbl, tgt_lbl):
    return f"<start_of_turn>user\nTranslate this {src_lbl} text to {tgt_lbl}:\n\n{src}<end_of_turn>\n<start_of_turn>model\n"

def _default_out_path(test_path: Path, src: str, tgt: str, k: int) -> Path:
    return test_path.parent / f"doc.{src}_2_{tgt}.{k}.jsonl"

def _word_len(txt: str) -> int:
    return len(txt.split())

# ─────────────────────────── main ───────────────────────────────
def main() -> None:
    # Define parameters directly for Colab environment
    test_file = "./pralekha_data/test/eng_hin/doc.eng.jsonl"  # Example test file
    example_src_file = "./pralekha_data/dev/eng_hin/doc.eng.jsonl" # Example example source file
    example_tgt_file = "./pralekha_data/dev/eng_hin/doc.hin.jsonl" # Example example target file
    few_shot = 5  # Example number of few-shot examples
    src_lang = "eng" # Example source language
    tgt_lang = "hin" # Example target language
    output_file = None # Optional output file
    src_label = None # Optional source label
    tgt_label = None # Optional target label


    if few_shot < 0:
        sys.exit("few_shot must be ≥ 0")

    src_lbl = src_label or LANG_LABELS.get(src_lang, src_lang)
    tgt_lbl = tgt_label or LANG_LABELS.get(tgt_lang, tgt_lang)

    test_path   = Path(test_file)
    ex_src_path = Path(example_src_file)
    ex_tgt_path = Path(example_tgt_file)

    out_path = (
        Path(output_file)
        if output_file
        else _default_out_path(test_path, src_lang, tgt_lang, few_shot)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- load all example pairs ---------------------------------------------
    ex_src = _load_texts(ex_src_path)
    ex_tgt = _load_texts(ex_tgt_path)
    if len(ex_src) != len(ex_tgt):
        sys.exit("[!] Example source / target length mismatch")

    # ---- pick first k whose English side is 101-200 words --------------------
    selected = []
    for s, t in zip(ex_src, ex_tgt):
        eng_side = s if src_lang == "eng" else t if tgt_lang == "eng" else None
        if eng_side is None:
            # neither side is English – should not happen for this task
            continue
        wlen = _word_len(eng_side)
        if 100 < wlen <= 200:
            selected.append((s, t))
            if len(selected) == few_shot:
                break

    if len(selected) < few_shot:
        print(f"[!] Only found {len(selected)} examples satisfying 101-200 words (needed {few_shot})")

    # Build system prompt
    system_prompt = f"<start_of_turn>system\nYou are a professional translator. Translate the text accurately from {src_lbl} to {tgt_lbl}, preserving meaning, tone, and formatting.<end_of_turn>\n"

    # Build example blocks
    example_blocks = [system_prompt]
    for s, t in selected:
        example_blocks.append(_build_block(s, t, src_lbl, tgt_lbl))

    # ---- build prompts -------------------------------------------------------
    with test_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        tests = 0
        for raw in fin:
            if not raw.strip():
                continue
            test_src = _extract_text(raw).strip()
            final_prompt = _build_final_prompt(test_src, src_lbl, tgt_lbl)
            full_prompt = "\n".join(example_blocks + [final_prompt])
            fout.write(json.dumps([full_prompt], ensure_ascii=False) + "\n")
            tests += 1

    print(f"✓ {out_path} written. examples={len(selected)}, tests={tests}")

if __name__ == "__main__":
    main()
