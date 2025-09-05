#!/usr/bin/env python                    #Gen_prompt for all lang pairs#
# -*- coding: utf-8 -*-
"""
Generate few-shot prompts for ALL language pairs (English side 101-200 words only).
Output: doc.{src}_2_{tgt}.{k}.jsonl for each language pair
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

# All language pairs to process
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
    return f"{src_lbl}: {src}\n\n{tgt_lbl}: {tgt}"

def _build_final_block(src, src_lbl, tgt_lbl):
    return f"{src_lbl}: {src}\n\n{tgt_lbl}:"

def _default_out_path(test_path: Path, src: str, tgt: str, k: int) -> Path:
    return test_path.parent / f"doc.{src}_2_{tgt}.{k}.jsonl"

def _word_len(txt: str) -> int:
    return len(txt.split())

def generate_prompts_for_pair(src_lang: str, tgt_lang: str, few_shot: int = 3, subset: str = "dev"):
    """Generate prompts for a specific language pair"""
    
    src_lbl = LANG_LABELS.get(src_lang, src_lang)
    tgt_lbl = LANG_LABELS.get(tgt_lang, tgt_lang)
    
    # Construct file paths for this language pair
    base_dir = "/tmp/pralekha_data"
    test_file = f"{base_dir}/test/{src_lang}_{tgt_lang}/doc.{src_lang}.jsonl"
    example_src_file = f"{base_dir}/{subset}/{src_lang}_{tgt_lang}/doc.{src_lang}.jsonl"
    example_tgt_file = f"{base_dir}/{subset}/{src_lang}_{tgt_lang}/doc.{tgt_lang}.jsonl"

    test_path = Path(test_file)
    ex_src_path = Path(example_src_file)
    ex_tgt_path = Path(example_tgt_file)

    # Check if files exist
    if not test_path.exists():
        print(f"⚠ Test file not found: {test_file}")
        return
    if not ex_src_path.exists() or not ex_tgt_path.exists():
        print(f"⚠ Example files not found for {src_lang}_{tgt_lang}")
        return

    out_path = _default_out_path(test_path, src_lang, tgt_lang, few_shot)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load all example pairs
    ex_src = _load_texts(ex_src_path)
    ex_tgt = _load_texts(ex_tgt_path)
    if len(ex_src) != len(ex_tgt):
        print(f"[!] Example source/target length mismatch for {src_lang}_{tgt_lang}")
        return

    # Pick first k examples where English side is 101-200 words
    selected = []
    for s, t in zip(ex_src, ex_tgt):
        # For English→Indic: English is source, for Indic→English: English is target
        eng_side = s if src_lang == "eng" else t if tgt_lang == "eng" else None
        if eng_side is None:
            continue
        wlen = _word_len(eng_side)
        if 100 < wlen <= 200:
            selected.append((s, t))
            if len(selected) == few_shot:
                break

    if len(selected) < few_shot:
        print(f"[!] Only found {len(selected)} examples satisfying 101-200 words for {src_lang}_{tgt_lang} (needed {few_shot})")

    example_blocks = [_build_block(s, t, src_lbl, tgt_lbl) for s, t in selected]

    # Build prompts
    with test_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        tests = 0
        for raw in fin:
            if not raw.strip():
                continue
            test_src = _extract_text(raw).strip()
            parts = example_blocks + [_build_final_block(test_src, src_lbl, tgt_lbl)]
            fout.write(json.dumps(["\n\n".join(parts)], ensure_ascii=False) + "\n")
            tests += 1

    print(f"✓ {out_path} written. examples={len(selected)}, tests={tests}")

# ─────────────────────────── main ───────────────────────────────
def main() -> None:
    few_shot = 3
    subset = "dev"  # Use "dev" set for few-shot examples

    print(f"Generating prompts for ALL language pairs (few_shot={few_shot})...")
    
    # Process all language pairs
    for src_lang, tgt_lang in LANGUAGE_PAIRS:
        print(f"\n{'='*50}")
        print(f"Processing: {LANG_LABELS[src_lang]} → {LANG_LABELS[tgt_lang]}")
        print(f"{'='*50}")
        
        generate_prompts_for_pair(src_lang, tgt_lang, few_shot, subset)

    print(f"\n{'='*50}")
    print("✓ Prompt generation completed for ALL language pairs!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
