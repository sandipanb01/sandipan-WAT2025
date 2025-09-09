#!/usr/bin/env python
# inference_vllm_chat_multiturn.py
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--model", required=True,
                   help="HF repo path or local dir; e.g. meta-llama/Llama-3.1-8B")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--sampling", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=4)
    return p.parse_args()

def load_prompts(path):
    """
    Expects each line to be either:
      - a string (single user prompt)
      - a JSON object like {"messages":[{"role":"user","content":"..."}, ...]}
    """
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "messages" in obj:
                yield obj["messages"]
            else:
                yield [{"role": "user", "content": line}]
        except json.JSONDecodeError:
            yield [{"role": "user", "content": line}]

def build_generator(args):
    from vllm import LLM, SamplingParams
    
    print(f"Initializing vLLM for model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="bfloat16",
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

    def generate(batch_messages, tokenizer):
        # batch_messages is a list of lists of {"role":..., "content":...} messages
        chat_prompts = [
            tokenizer.apply_chat_template(
                messages=messages,
                add_generation_prompt=True,
                tokenize=False
            )
            for messages in batch_messages
        ]
        outputs = llm.generate(chat_prompts, sampling_params)
        return [
            {"generated_text": output.outputs[0].text}
            for output in outputs
        ]

    return generate, tokenizer

def main():
    args = parse_args()
    prompts = list(load_prompts(args.input_file))
    print(f"Loaded {len(prompts)} prompts.")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    gen_fn, tokenizer = build_generator(args)
    print(f"Starting batched inference with vLLM (batch size: {args.batch_size})...")
    
    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_outputs = gen_fn(batch_prompts, tokenizer)
            
            for output in batch_outputs:
                generation = output["generated_text"].strip()
                fout.write(json.dumps([generation], ensure_ascii=False) + "\n")
            fout.flush()

    print("vLLM multi-turn chat inference completed successfully.")

if __name__ == "__main__":
    main()
