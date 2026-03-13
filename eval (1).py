import json
import os
import re
import argparse
from tqdm import tqdm
import torch
import sacrebleu

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names
from peft import AutoPeftModelForCausalLM

def free_gpu():
    import gc
    
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

def load_wat_dataset_train(tokenizer=None):
    logging.info(f"Loading Machine Translation dataset")

    dataset = load_dataset("ai4bharat/Pralekha", data_dir="train")

    def format_example(example):
        src_lang = example["src_lang"]
        tgt_lang = example["tgt_lang"]

        source_lang = "English"
        target_lang = ""

        if tgt_lang == "ben":
            target_lang = "Bengali"
        elif tgt_lang == "guj":
            target_lang = "Gujarati"
        elif tgt_lang == "hin":
            target_lang = "Hindi"
        elif tgt_lang == "kan":
            target_lang = "Kannada"
        elif tgt_lang == "mal":
            target_lang = "Malayalam"
        elif tgt_lang == "mar":
            target_lang = "Marathi"
        elif tgt_lang == "ori":
            target_lang = "Odiya"
        elif tgt_lang == "pan":
            target_lang = "Punjabi"
        elif tgt_lang == "tam":
            target_lang = "Tamil"
        elif tgt_lang == "tel":
            target_lang = "Telugu"
        elif tgt_lang == "urd":
            target_lang = "Urdu"

        messages = {
            "prompt": [
                {
                    "role": "user",
                    "content": f"Translate the following sentence from English to {target_lang}.\n\n"
                    f"English: {example['src_txt']}",
                }
            ],
            "completion": [{"role": "assistant", "content": example["tgt_txt"]}],
        }

        return messages

    dataset = dataset.filter(
        lambda x: x["src_txt"] != x["tgt_txt"],
        num_proc=32,
    )["train"]
    dataset = dataset.map(format_example)
    logging.info(dataset)

    dev_dataset = load_dataset("ai4bharat/Pralekha", data_dir="dev")
    dev_dataset = dev_dataset.filter(
        lambda x: x["src_txt"] != x["tgt_txt"],
        num_proc=32,
    )["train"]
    dev_dataset = dev_dataset.map(format_example)

    return dataset, dev_dataset


def build_prompt_wat(example, tokenizer):
    src = example["src_txt"]
    ref = example["tgt_txt"]
    tgt_lang = example["tgt_lang"]

    target_lang = ""

    if tgt_lang == "ben":
        target_lang = "Bengali"
    elif tgt_lang == "guj":
        target_lang = "Gujarati"
    elif tgt_lang == "hin":
        target_lang = "Hindi"
    elif tgt_lang == "kan":
        target_lang = "Kannada"
    elif tgt_lang == "mal":
        target_lang = "Malayalam"
    elif tgt_lang == "mar":
        target_lang = "Marathi"
    elif tgt_lang == "ori":
        target_lang = "Odiya"
    elif tgt_lang == "pan":
        target_lang = "Punjabi"
    elif tgt_lang == "tam":
        target_lang = "Tamil"
    elif tgt_lang == "tel":
        target_lang = "Telugu"
    elif tgt_lang == "urd":
        target_lang = "Urdu"

    prompt = (
        f"Translate the following text from English to {target_lang}:\n"
        f"English: {src}\n"
        f"{target_lang}: "
    )

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokens = tokenizer(
        prompt,
        truncation=True,
        padding=False,
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "reference": ref,
    }

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model"
    )

    parser.add_argument(
    "--model_name",
    type=str,
    default="ibm-iitr-mt-research/gemma-3-1b-it_wat_sft")

    parser.add_argument("--dataset", type=str, default="wat")

    parser.add_argument("--output_dir", type=str, default="./output")

    parser.add_argument("--use_lora", action="store_true")

    parser.add_argument("--bf16", action="store_true")

    return parser.parse_args()

def evaluate_wat(model, tokenizer, dataset, batch_size=4):
    predictions = []
    references = []

    print("Running inference...")

    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]

        padded = tokenizer.pad(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
            padding=True,
            return_tensors="pt",
        )

        input_ids = padded["input_ids"].to(model.device)
        attention_mask = padded["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=4096,
                do_sample=False,
                use_cache=True,
            )

        new_tokens = outputs[:, input_ids.shape[1]:]

        decoded = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )

        predictions.extend(decoded)

        refs = dataset[i:i+batch_size]["reference"]
        references.extend(refs)

    return references, predictions


def main():
    args = parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    DATASET_NAME = "ai4bharat/Pralekha"
    EVAL_SPLIT = "test"

    # ===== CHANGE: evaluate ALL eng_* splits instead of eng_hin =====
    all_configs = get_dataset_config_names(DATASET_NAME)
    test_splits = [c for c in all_configs if c.startswith("eng_")]

    results = {}

    for split_name in test_splits:
        dataset = load_dataset(DATASET_NAME, EVAL_SPLIT, split=split_name)
        dataset = dataset.map(build_prompt_wat,
                              fn_kwargs={"tokenizer": tokenizer})

        references, predictions = evaluate_wat(model, tokenizer, dataset)

        bleu, chrf = calc_metrics(predictions, references)

        results[split_name] = {"BLEU": bleu, "CHRF": chrf}

    avg_bleu = sum(r["BLEU"] for r in results.values()) / len(results)
    avg_chrf = sum(r["CHRF"] for r in results.values()) / len(results)

    results["average"] = {
        "BLEU": round(avg_bleu, 2),
        "CHRF": round(avg_chrf, 2),
    }

    results_path = (
        f"{args.output_dir}/wat_-1_{args.model_name.split('/')[-1]}.json"
    )

    with open(results_path, "w") as outfile:
        json.dump(results, outfile, indent=2)

    # ===== checkpoint loop remains structurally identical =====

    base_dir = args.output_dir

    for name in sorted(os.listdir(base_dir)):
        free_gpu()

        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint-"):
            match = re.match(r"checkpoint-(\d+)", name)
            ckpt_num = int(match.group(1))

            if args.use_lora:
                model = AutoPeftModelForCausalLM.from_pretrained(
                    path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )
                model = model.merge_and_unload()
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                )

            model.eval()

            tokenizer = AutoTokenizer.from_pretrained(path)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            results = {}

            for split_name in test_splits:
                dataset = load_dataset(DATASET_NAME, EVAL_SPLIT,
                                       split=split_name)

                dataset = dataset.map(build_prompt_wat,
                                      fn_kwargs={"tokenizer": tokenizer})

                references, predictions = evaluate_wat(model, tokenizer, dataset)

                bleu, chrf = calc_metrics(predictions, references)

                results[split_name] = {"BLEU": bleu, "CHRF": chrf}

            avg_bleu = sum(r["BLEU"] for r in results.values()) / len(results)
            avg_chrf = sum(r["CHRF"] for r in results.values()) / len(results)

            results["average"] = {
                "BLEU": round(avg_bleu, 2),
                "CHRF": round(avg_chrf, 2),
            }

            results_path = (
                f"{args.output_dir}/wat_{ckpt_num}.json"
            )

            with open(results_path, "w") as outfile:
                json.dump(results, outfile, indent=2)

if __name__ == "__main__":
    main()