#!/usr/bin/env python
# gene_prompt_gemma3_1b.py

import argparse
import json
import sys
import os
from pathlib import Path

from transformers import AutoTokenizer

LANG_LABELS = {
    "eng": "English", "ben": "Bengali", "guj": "Gujarati",
    "hin": "Hindi", "kan": "Kannada", "mal": "Malayalam",
    "mar": "Marathi", "ori": "Odia", "pan": "Punjabi",
    "tam": "Tamil", "tel": "Telugu", "urd": "Urdu",
}

def _extract_text(line: str) -> str:
    obj = json.loads(line)
    if isinstance(obj, list):
        return str(obj[0])
    if isinstance(obj, str):
        return obj
    raise ValueError("Line must be string or single-element list.")

def _load_texts(path: Path):
    with path.open(encoding="utf-8") as f:
        return [_extract_text(ln).strip() for ln in f if ln.strip()]

def _word_len(txt: str) -> int:
    return len(txt.split())

def build_blocks(ex_src, ex_tgt, src_lbl, tgt_lbl, few_shot, chat, tokenizer=None):
    selected = []
    for s, t in zip(ex_src, ex_tgt):
        eng = s if src_lbl == "English" else t if tgt_lbl == "English" else None
        if eng and 100 < _word_len(eng) <= 200:
            selected.append((s, t))
            if len(selected) == few_shot:
                break
    return selected

def format_prompt(src, tgt, src_lbl, tgt_lbl, chat, tokenizer):
    if not chat:
        return f"{src_lbl}: {src}\n\n{tgt_lbl}: {tgt}"
    # chat template style
    messages = [
        {"role": "user", "content": f"{src_lbl}: {src}\n\n{tgt_lbl}: {tgt}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_file", required=True)
    ap.add_argument("--example_src_file", required=True)
    ap.add_argument("--example_tgt_file", required=True)
    ap.add_argument("--few_shot", type=int, required=True)
    ap.add_argument("--src_lang", required=True)
    ap.add_argument("--tgt_lang", required=True)
    ap.add_argument("--output_file")
    ap.add_argument("--src_label")
    ap.add_argument("--tgt_label")
    ap.add_argument("--chat_template", action="store_true",
                    help="Use chat template with tokenizer.apply_chat_template")
    ap.add_argument("--model", help="Required if --chat_template to load tokenizer")
    args = ap.parse_args()

    if args.few_shot < 0:
        sys.exit("few_shot must be ≥ 0")

    src_lbl = args.src_label or LANG_LABELS.get(args.src_lang, args.src_lang)
    tgt_lbl = args.tgt_label or LANG_LABELS.get(args.tgt_lang, args.tgt_lang)

    test_path = Path(args.test_file)
    ex_src = _load_texts(Path(args.example_src_file))
    ex_tgt = _load_texts(Path(args.example_tgt_file))
    if len(ex_src) != len(ex_tgt):
        sys.exit("[!] Example source / target length mismatch")

    example_pairs = build_blocks(ex_src, ex_tgt, src_lbl, tgt_lbl, args.few_shot, args.chat_template, None)

    tokenizer = None
    if args.chat_template:
        if not args.model:
            sys.exit("Need --model for chat_template mode")
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    with open(args.output_file or f"doc.{args.src_lang}_2_{args.tgt_lang}.{args.few_shot}.jsonl", "w", encoding="utf-8") as fout:
        for raw in test_path.open(encoding="utf-8"):
            if not raw.strip(): continue
            test_src = _extract_text(raw).strip()

            parts = []
            for s, t in example_pairs:
                parts.append(format_prompt(s, t, src_lbl, tgt_lbl, args.chat_template, tokenizer))

            final = format_prompt(test_src, "", src_lbl, tgt_lbl, args.chat_template, tokenizer)
            parts.append(final)

            if args.chat_template:
                fout.write(json.dumps(parts, ensure_ascii=False) + "\n")
            else:
                fout.write(json.dumps(["\n\n".join(parts)], ensure_ascii=False) + "\n")

    print("✓ Prompt file written.")

if __name__ == "__main__":
    main()
