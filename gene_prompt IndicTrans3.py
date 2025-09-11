#!/usr/bin/env python
# gene_prompt_indictrans3.py
"""
Generate prompts for IndicTrans3 doc-level translation using Pralekha JSONL data.
Supports naive prompts or Hugging Face chat template.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer

# IndicTrans3 supported languages
LANG_LABELS = {
    "eng": "English", "ben": "Bengali", "guj": "Gujarati",
    "hin": "Hindi", "kan": "Kannada", "mal": "Malayalam",
    "mar": "Marathi", "ori": "Odia", "pan": "Punjabi",
    "tam": "Tamil", "tel": "Telugu", "urd": "Urdu",
}

def load_jsonl(path: Path) -> List[str]:
    """Load JSONL file where each line is a string."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def build_chat_prompt(examples: List[Tuple[str, str]],
                      test_src: str,
                      src_label: str,
                      tgt_label: str,
                      tokenizer,
                      system_prompt: str = None) -> str:
    """Build prompt using HF chat template with few-shot examples."""
    example_block = "\n\n".join(f"{src_label}: {s}\n{tgt_label}: {t}" for s, t in examples)
    sys_msg = system_prompt or f"You are a translation assistant. Translate from {src_label} to {tgt_label} only."
    if example_block:
        sys_msg += f"\n\nHere are some examples:\n\n{example_block}\n\nFollow this style."

    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"{src_label}: {test_src}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def build_naive_prompt(examples: List[Tuple[str, str]],
                       test_src: str,
                       src_label: str,
                       tgt_label: str) -> List[str]:
    """Build simple few-shot prompt without chat template."""
    lines = [f"{src_label}: {s}\n{tgt_label}: {t}" for s, t in examples]
    lines.append(f"{src_label}: {test_src}\n{tgt_label}: ")
    return ["\n\n".join(lines)]

def save_jsonl(lines: List, path: Path) -> None:
    """Save list of items to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in lines:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def main(args=None):
    parser = argparse.ArgumentParser(description="Generate doc-level prompts for IndicTrans3")
    parser.add_argument("--out_root", required=True, help="Root directory of downloaded Pralekha JSONL files")
    parser.add_argument("--split", default="dev", help="Dataset split: dev, test, or train")
    parser.add_argument("--pair", required=True, help="Language pair in the format src_tgt (e.g., eng_hin)")
    parser.add_argument("--few_shot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--chat_template", action="store_true", help="Use HF chat template")
    parser.add_argument("--model", default="ai4bharat/IndicTrans3-beta", help="Model for tokenizer")
    parser.add_argument("--system_prompt", help="Custom system instruction")
    parser.add_argument("--output_file", help="Output JSONL prompt file")
    args = parser.parse_args(args)

    src_lang, tgt_lang = args.pair.split("_")
    src_label = LANG_LABELS.get(src_lang, src_lang)
    tgt_label = LANG_LABELS.get(tgt_lang, tgt_lang)

    # Locate the downloaded JSONL files
    base_path = Path(args.out_root) / args.split / args.pair
    src_file = base_path / f"doc.{src_lang}.jsonl"
    tgt_file = base_path / f"doc.{tgt_lang}.jsonl"

    # Load all documents
    src_docs = load_jsonl(src_file)
    tgt_docs = load_jsonl(tgt_file)
    if len(src_docs) != len(tgt_docs):
        raise ValueError(f"Source/target doc length mismatch: {len(src_docs)} != {len(tgt_docs)}")

    # Select few-shot examples
    examples = list(zip(src_docs, tgt_docs))[:args.few_shot]

    # Load tokenizer if using chat template
    tokenizer = None
    if args.chat_template:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Generate prompts for all documents
    prompts = []
    for doc in src_docs:
        if args.chat_template:
            prompt = build_chat_prompt(examples, doc, src_label, tgt_label, tokenizer, args.system_prompt)
        else:
            prompt = build_naive_prompt(examples, doc, src_label, tgt_label)
        prompts.append(prompt)

    # Save output
    output_path = Path(args.output_file or f"{base_path}/doc.{src_lang}_2_{tgt_lang}.{args.few_shot}.prompts.jsonl")
    save_jsonl(prompts, output_path)
    print(f"✓ Prompts saved to {output_path} ({len(src_docs)} docs)")

if __name__ == "__main__":
    # Example of how to run in Colab by passing arguments as a list
    # Replace placeholder values with actual file paths and language codes
    main(args=[
        '--out_root', '/content/pralekha_data',  # Replace with the root directory of your downloaded data
        '--pair', 'eng_hin',  # Replace with your desired language pair (e.g., eng_ben)
        '--split', 'dev',  # Dataset split (e.g., dev, test, train)
        '--few_shot', '0',  # Number of few-shot examples (change if needed)
        # Add '--chat_template' if you want to use the chat template
        # Add '--output_file', '/content/your_output.jsonl' to specify output file
        # Add '--model', 'your_model_name' to specify a different model
        # Add '--system_prompt', 'your_system_prompt' for a custom system prompt
    ])
