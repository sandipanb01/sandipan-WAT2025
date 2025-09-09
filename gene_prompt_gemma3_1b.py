#!/usr/bin/env python
# gene_prompt_gemma3_1b.py

import argparse
import json
import sys
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


def build_examples(src_texts, tgt_texts, src_label, tgt_label, few_shot):
    selected = []
    for s, t in zip(src_texts, tgt_texts):
        eng_side = s if src_label == "English" else t if tgt_label == "English" else None
        if eng_side and 100 < _word_len(eng_side) <= 200:
            selected.append((s, t))
            if len(selected) == few_shot:
                break
    return selected


def build_chat_prompt(examples, test_src, src_label, tgt_label, tokenizer):
    """
    Builds a chat prompt with a system instruction + few-shot examples + test input,
    then applies Hugging Face’s chat template.
    """
    messages = [
        {
            "role": "system",
            "content": f"You are a translation assistant. Translate from {src_label} to {tgt_label}. "
                       f"Generate only the translation, nothing else."
        }
    ]

    # Add few-shot examples
    for s, t in examples:
        messages.append({"role": "user", "content": f"{src_label}: {s}"})
        messages.append({"role": "assistant", "content": f"{tgt_label}: {t}"})

    # Final turn: test input for generation
    messages.append({"role": "user", "content": f"{src_label}: {test_src}"})

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  # signals model to generate the assistant response
    )
    return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--example_src_file", required=True)
    parser.add_argument("--example_tgt_file", required=True)
    parser.add_argument("--few_shot", type=int, required=True)
    parser.add_argument("--src_lang", required=True)
    parser.add_argument("--tgt_lang", required=True)
    parser.add_argument("--output_file")
    parser.add_argument("--src_label")
    parser.add_argument("--tgt_label")
    parser.add_argument("--chat_template", action="store_true",
                        help="Enable chat formatting using tokenizer.apply_chat_template")
    parser.add_argument("--model", help="Required to load tokenizer when using chat_template")
    args = parser.parse_args()

    if args.few_shot < 0:
        sys.exit("few_shot must be ≥ 0")

    src_label = args.src_label or LANG_LABELS.get(args.src_lang, args.src_lang)
    tgt_label = args.tgt_label or LANG_LABELS.get(args.tgt_lang, args.tgt_lang)

    ex_src = _load_texts(Path(args.example_src_file))
    ex_tgt = _load_texts(Path(args.example_tgt_file))
    if len(ex_src) != len(ex_tgt):
        sys.exit("[!] Example source/target length mismatch")

    examples = build_examples(ex_src, ex_tgt, src_label, tgt_label, args.few_shot)

    tokenizer = None
    if args.chat_template:
        if not args.model:
            sys.exit("Error: --model argument required for chat_template mode")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    output_path = args.output_file or f"doc.{args.src_lang}_2_{args.tgt_lang}.{args.few_shot}.jsonl"
    with open(output_path, "w", encoding="utf-8") as fout:
        for test_line in Path(args.test_file).open(encoding="utf-8"):
            if not test_line.strip():
                continue
            test_src = _extract_text(test_line).strip()

            if args.chat_template:
                prompt = build_chat_prompt(examples, test_src, src_label, tgt_label, tokenizer)
                fout.write(json.dumps(prompt, ensure_ascii=False) + "\n")
            else:
                parts = []
                for s, t in examples:
                    parts.append(f"{src_label}: {s}\n\n{tgt_label}: {t}")
                parts.append(f"{src_label}: {test_src}\n\n{tgt_label}: ")
                fout.write(json.dumps(["\n\n".join(parts)], ensure_ascii=False) + "\n")

    print(f"✓ Prompt file written to: {output_path}")


if __name__ == "__main__":
    main()
