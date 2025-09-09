#!/usr/bin/env python
# inference_vllm.py
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
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--model", required=True,
                   help="HF repo path or local dir; e.g. google/gemma-3-1b-it")
    p.add_argument("--max_new_tokens", type=int, default=4096)  # Increased for document-level translation
    p.add_argument("--sampling", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--model_type", choices=["pretrained", "instruction_tuned"], default="instruction_tuned",
                   help="Type of model: 'pretrained' for base models, 'instruction_tuned' for chat models")
    return p.parse_args(args=args)

def load_prompts(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        obj = json.loads(line) if line[0] in "{[" else {"prompt": line}
        if isinstance(obj, list): yield obj[0]
        else: yield obj["prompt"]

def build_generator(args):
    # Import vLLM here to control error handling
    from vllm import LLM, SamplingParams

    # Initialize vLLM engine with safe settings for problematic environments
    print(f"Initializing vLLM for model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,  # Use single GPU tensor parallelism for stability
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,  # Be conservative with memory
    )

    # Configure sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.sampling else 0.0,
        top_p=args.top_p if args.sampling else 1.0,
        stop=["\n\n"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Define a generator function that uses vLLM for batched inference
    def generate(prompts, tokenizer):
        if args.model_type == "instruction_tuned":
            # Apply chat template for instruction-tuned models
            chat_formatted_prompts = []
            for each_prompt in prompts:
                # Check if prompt already has chat template markers (Gemma format)
                if "<start_of_turn>" in each_prompt:
                    # Prompt already formatted with Gemma template
                    chat_formatted_prompts.append(each_prompt)
                else:
                    # Apply standard chat template
                    messages = [
                        {"role": "user", "content": each_prompt}
                    ]
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    chat_formatted_prompts.append(text)
            outputs = llm.generate(chat_formatted_prompts, sampling_params)
        else:
            # Use naive prompts for pre-trained models
            outputs = llm.generate(prompts, sampling_params)

        return [
            {"generated_text": output.outputs[0].text.strip()}
            for output in outputs
        ]

    return generate, tokenizer

def main(args=None):
    args = parse_args(args=args)
    prompts = list(load_prompts(args.input_file))
    print(f"Loaded {len(prompts)} prompts. Model type: {args.model_type}")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Initialize vLLM generator
    gen_fn, tokenizer = build_generator(args)
    print(f"Starting batched inference with vLLM (batch size: {args.batch_size})...")

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_outputs = gen_fn(batch_prompts, tokenizer)

            for output in batch_outputs:
                generation = output["generated_text"]
                fout.write(json.dumps([generation], ensure_ascii=False) + "\n")
            fout.flush()

    print("vLLM inference completed successfully.")

if __name__ == "__main__":
    # Define arguments programmatically for Colab execution
    # For Gemma 3-1B instruction-tuned model
    colab_args_instruct = [
        "--input_file", "/content/pralekha_data/test/eng_hin/doc.eng_2_hin.5.jsonl",
        "--output_file", "/content/pralekha_data/test/eng_hin/doc.eng_2_hin.5.vllm.gemma3-1b-it.jsonl",
        "--model", "google/gemma-3-1b-it",
        "--batch_size", "4",
        "--max_new_tokens", "4096",
        "--model_type", "instruction_tuned"
    ]

    # For Gemma 3-1B pre-trained model
    # colab_args_pretrained = [
    #     "--input_file", "/content/pralekha_data/test/eng_hin/doc.eng_2_hin.5.jsonl",
    #     "--output_file", "/content/pralekha_data/test/eng_hin/doc.eng_2_hin.5.vllm.gemma3-1b-pt.jsonl",
    #     "--model", "google/gemma-3-1b-pt",
    #     "--batch_size", "4",
    #     "--max_new_tokens", "4096",
    #     "--model_type", "pretrained"
    # ]

    # Run inference for instruction-tuned model
    print("Running inference for Gemma 3-1B instruction-tuned model...")
    main(args=colab_args_instruct)

    # Run inference for pre-trained model
    # print("Running inference for Gemma 3-1B pre-trained model...")
    # main(args=colab_args_pretrained)
