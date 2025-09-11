#!/usr/bin/env python
# inference_indictrans3.py
"""
Run inference for IndicTrans3 using generated prompts.
Saves output translations to JSONL.
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

def load_prompts(path):
    """Load prompts from JSONL."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Input prompt file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    return lines

def save_jsonl(lines, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

def build_generator(model_name, max_new_tokens=4096, sampling=True, temperature=0.7, top_p=0.9):
    """Initialize vLLM generator and tokenizer."""
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature if sampling else 0.0,
        top_p=top_p if sampling else 1.0,
        stop=["\n\n"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def generate(batch_prompts):
        outputs = llm.generate(batch_prompts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    return generate, tokenizer

def main(args=None): # Modified to accept args
    parser = argparse.ArgumentParser(description="IndicTrans3 inference")
    parser.add_argument("--input_file", required=True, help="JSONL file of prompts")
    parser.add_argument("--output_file", required=True, help="Output JSONL file of translations")
    parser.add_argument("--model", default="ai4bharat/IndicTrans3-beta", help="HF model name")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--sampling", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(args.input_file)
    print(f"Loaded {len(prompts)} prompts.")

    gen_fn, _ = build_generator(args.model, args.max_new_tokens, args.sampling)

    translations = []
    for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_outputs = gen_fn(batch_prompts)
        translations.extend(batch_outputs)

    save_jsonl(translations, args.output_file)
    print(f"✓ Translations saved to {args.output_file} ({len(translations)} docs)")

if __name__ == "__main__":
    # Example of how to run in Colab by passing arguments as a list
    # Replace placeholder values with actual file paths and language codes
    main(args=[
        '--input_file', '/content/pralekha_data/dev/eng_hin/doc.eng_2_hin.0.prompts.jsonl', # Replace with your input prompt file
        '--output_file', '/content/pralekha_data/dev/eng_hin/doc.eng_2_hin.translations.jsonl', # Replace with your desired output file
        '--model', 'ai4bharat/IndicTrans3-beta', # Model to use
        '--max_new_tokens', '4096',
        '--batch_size', '2',
        '--sampling'
    ])
