#!/usr/bin/env python
# inference_all_pairs_vllm.py
import argparse
import json
import sys
import os
from tqdm import tqdm
from pathlib import Path
import torch
from transformers import AutoTokenizer

def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF repo path or local dir; e.g. google/gemma-3-1b-it")
    p.add_argument("--max_new_tokens", type=int, default=4096)  # Increased for document-level translation
    p.add_argument("--sampling", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--model_type", choices=["pretrained", "instruction_tuned"], default="instruction_tuned",
                   help="Type of model: 'pretrained' for base models, 'instruction_tuned' for chat models")
    p.add_argument("--test_root", required=True,
                   help="Root folder for Pralekha test set, e.g. /content/pralekha_data/test")
    p.add_argument("--out_root", required=True,
                   help="Root folder for saving outputs, e.g. /content/pralekha_data/vllm_outputs")
    return p.parse_args(args=args)

def load_prompts(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: 
            continue
        obj = json.loads(line) if line[0] in "{[" else {"prompt": line}
        if isinstance(obj, list):
            yield obj[0]
        else:
            yield obj["prompt"]

def build_generator(args):
    from vllm import LLM, SamplingParams

    print(f"Initializing vLLM for model: {args.model} ({args.model_type})")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.sampling else 0.0,
        top_p=args.top_p if args.sampling else 1.0,
        stop=["\n\n"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def generate(prompts, tokenizer):
        if args.model_type == "instruction_tuned":
            formatted = []
            for prompt_text in prompts:
                messages = [{"role": "user", "content": prompt_text}]
                template_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                formatted.append(template_text)
            outputs = llm.generate(formatted, sampling_params)
        else:
            outputs = llm.generate(prompts, sampling_params)

        return [{"generated_text": out.outputs[0].text.strip()} for out in outputs]

    return generate, tokenizer

def run_inference(args, lang_pair, direction):
    # Input file path
    test_file = os.path.join(args.test_root, lang_pair, f"doc.{direction}.jsonl")
    if not os.path.exists(test_file):
        print(f"Skipping {direction}: {test_file} not found")
        return

    # Output file path
    os.makedirs(os.path.join(args.out_root, lang_pair), exist_ok=True)
    out_file = os.path.join(
        args.out_root, lang_pair,
        f"doc.{direction}.vllm.{Path(args.model).name}.{args.model_type}.jsonl"
    )

    prompts = list(load_prompts(test_file))
    print(f"[{direction}] Loaded {len(prompts)} prompts.")

    gen_fn, tokenizer = build_generator(args)

    with open(out_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc=f"Generating {direction}"):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_outputs = gen_fn(batch_prompts, tokenizer)

            for output in batch_outputs:
                generation = output["generated_text"]
                fout.write(json.dumps([generation], ensure_ascii=False) + "\n")
            fout.flush()

    print(f"[{direction}] Saved outputs to {out_file}")

def main(args=None):
    args = parse_args(args=args)

    # All Pralekha doc-level language pairs
    lang_pairs = {
        "eng_ben": ["eng_2_ben", "ben_2_eng"],
        "eng_guj": ["eng_2_guj", "guj_2_eng"],
        "eng_hin": ["eng_2_hin", "hin_2_eng"],
        "eng_mar": ["eng_2_mar", "mar_2_eng"],
        "eng_mal": ["eng_2_mal", "mal_2_eng"],
        "eng_ory": ["eng_2_ory", "ory_2_eng"],
        "eng_pan": ["eng_2_pan", "pan_2_eng"],
        "eng_tam": ["eng_2_tam", "tam_2_eng"],
        "eng_tel": ["eng_2_tel", "tel_2_eng"],
        "eng_urd": ["eng_2_urd", "urd_2_eng"],
    }

    for lang_pair, directions in lang_pairs.items():
        for direction in directions:
            run_inference(args, lang_pair, direction)

if __name__ == "__main__":
    # Example: instruction-tuned
    colab_args_instruct = [
        "--model", "google/gemma-3-1b-it",
        "--model_type", "instruction_tuned",
        "--max_new_tokens", "4096",
        "--batch_size", "4",
        "--test_root", "/content/pralekha_data/test",
        "--out_root", "/content/pralekha_data/vllm_outputs"
    ]

    # Example: pretrained
    colab_args_pretrained = [
        "--model", "google/gemma-3-1b-pt",
        "--model_type", "pretrained",
        "--max_new_tokens", "4096",
        "--batch_size", "4",
        "--test_root", "/content/pralekha_data/test",
        "--out_root", "/content/pralekha_data/vllm_outputs"
    ]

    print("=== Running inference for Gemma 3-1B instruction-tuned model ===")
    main(args=colab_args_instruct)

    print("=== Running inference for Gemma 3-1B pretrained model ===")
    main(args=colab_args_pretrained)
