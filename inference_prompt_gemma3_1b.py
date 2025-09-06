#!/usr/bin/env python    #INFERENCE PROMPT FOR ALL LANGUAGE PAIRS#
# inference_vllm.py
import argparse
import json
import sys
import os
from tqdm import tqdm
from pathlib import Path
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--model", default="google/gemma-3-1b-pt",
                   help="HF repo path or local dir; using google/gemma-3-1b-pt")
    p.add_argument("--max_new_tokens", type=int, default=4096)
    p.add_argument("--sampling", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()

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

    # Initialize vLLM engine
    print(f"Initializing vLLM for model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="float16",
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.sampling else 0.0,
        top_p=args.top_p if args.sampling else 1.0,
        stop=["<end_of_turn>", "\n\n"],
    )
    
    # Define a generator function that uses vLLM for batched inference
    def generate(prompts):
        outputs = llm.generate(prompts, sampling_params)
        return [
            {"generated_text": prompt + output.outputs[0].text}
            for prompt, output in zip(prompts, outputs)
        ]
    
    return generate

def main():
    args = parse_args()
    prompts = list(load_prompts(args.input_file))
    print(f"Loaded {len(prompts)} prompts.")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Initialize vLLM generator
    gen_fn = build_generator(args)
    print(f"Starting batched inference with vLLM (batch size: {args.batch_size})...")
    
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_outputs = gen_fn(batch_prompts)
            
            for output, prompt in zip(batch_outputs, batch_prompts):
                generation = output["generated_text"].replace(prompt, "").strip()
                # Clean up any remaining special tokens
                generation = generation.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
                fout.write(json.dumps([generation], ensure_ascii=False) + "\n")
            fout.flush()

            # Report memory usage for monitoring
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                used_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"Max GPU memory used: {used_memory:.2f} MB")
    
    print("vLLM inference completed successfully.")

if __name__ == "__main__":
    main()
