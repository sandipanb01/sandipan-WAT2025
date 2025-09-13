#!/usr/bin/env python
# inference_indictrans3.py

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# -------------------------------
# Utility: Build chat-style prompt
# -------------------------------
def build_prompt(src_text: str, src_lang: str, tgt_lang: str):
    """Builds chat-style prompts for IndicTrans3 using HF chat template."""
    # System instruction
    system_msg = (
        f"You are a translation model. Translate the following text "
        f"from {src_lang} to {tgt_lang}."
    )

    # Chat message format
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": src_text},
    ]
    return messages


# -------------------------------
# Load prompts from file
# -------------------------------
def load_prompts(path: Path):
    """Load prompts from JSONL file. Each line is a dict with src_text, src_lang, tgt_lang."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            prompts.append(entry)
    return prompts


# -------------------------------
# Save translations
# -------------------------------
def save_outputs(outputs, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for out in outputs:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


# -------------------------------
# Main inference function
# -------------------------------
def run_inference(model_name, input_file, output_file, max_new_tokens=4096):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Init vLLM
    llm = LLM(model=model_name, trust_remote_code=True)

    # Sampling params
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
    )

    # Load prompts
    data = load_prompts(Path(input_file))

    # Format prompts using chat template
    formatted_prompts = []
    for item in data:
        src_text = item["src_text"]
        src_lang = item["src_lang"]
        tgt_lang = item["tgt_lang"]

        messages = build_prompt(src_text, src_lang, tgt_lang)
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(
            {"id": item.get("id", None), "prompt": formatted_prompt}
        )

    # Run inference
    texts = [x["prompt"] for x in formatted_prompts]
    outputs = llm.generate(texts, sampling_params)

    # Collect results
    results = []
    for inp, out in zip(formatted_prompts, outputs):
        translation = out.outputs[0].text.strip()
        results.append(
            {
                "id": inp["id"],
                "prompt": inp["prompt"],
                "translation": translation,
            }
        )

    # Save
    save_outputs(results, Path(output_file))
    print(f"✅ Saved {len(results)} translations to {output_file}")


# -------------------------------
# CLI entrypoint
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path or name of IndicTrans3 model")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to JSONL prompts file")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Where to save translations")
    parser.add_argument("--max_new_tokens", type=int, default=4096,
                        help="Max tokens for generation")
    args = parser.parse_args()

    run_inference(args.model_name, args.input_file, args.output_file, args.max_new_tokens)


if __name__ == "__main__":
    main()
