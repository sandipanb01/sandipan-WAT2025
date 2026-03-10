import json
import os
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


LANG_MAP = {
    "ben": "Bengali",
    "guj": "Gujarati",
    "hin": "Hindi",
    "kan": "Kannada",
    "mal": "Malayalam",
    "mar": "Marathi",
    "ori": "Odiya",
    "pan": "Punjabi",
    "tam": "Tamil",
    "tel": "Telugu",
    "urd": "Urdu",
}


def build_prompt_wat(example, tokenizer):

    src = example["src_txt"]
    tgt_lang = example["tgt_lang"]

    target_lang = LANG_MAP.get(tgt_lang, tgt_lang)

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
        "reference": example["tgt_txt"],
    }


def evaluate_wat(model, tokenizer, dataset, batch_size=4):

    predictions = []
    references = []

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


def evaluate_model(model_path, output_dir, dtype, use_lora):

    print(f"\nEvaluating {model_path}")

    if use_lora:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype
        )
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=dtype
        )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    DATASET_NAME = "ai4bharat/Pralekha"

    configs = get_dataset_config_names(DATASET_NAME)

    configs = [c for c in configs if c.startswith("eng_")]

    results = {}

    bleu_all = []
    chrf_all = []

    for config in configs:

        print(f"Running {config}")

        dataset = load_dataset(
            DATASET_NAME,
            config,
            split="test"
        )

        dataset = dataset.map(
            build_prompt_wat,
            fn_kwargs={"tokenizer": tokenizer}
        )

        references, predictions = evaluate_wat(model, tokenizer, dataset)

        bleu, chrf = calc_metrics(predictions, references)

        results[config] = {
            "BLEU": bleu,
            "CHRF": chrf
        }

        bleu_all.append(bleu)
        chrf_all.append(chrf)

    results["avg"] = {
        "BLEU": round(sum(bleu_all) / len(bleu_all), 2),
        "CHRF": round(sum(chrf_all) / len(chrf_all), 2),
    }

    name = model_path.split("/")[-1]

    results_path = f"{output_dir}/wat_{name}.json"

    with open(results_path, "w") as outfile:
        json.dump(results, outfile, indent=2)

    print(f"Saved results → {results_path}")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="ibm-iitr-mt-research/gemma-3-1b-it_wat_sft",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
    )

    parser.add_argument(
        "--bf16",
        action="store_true",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch.float32
    if args.fp16:
        dtype = torch.float16
    if args.bf16:
        dtype = torch.bfloat16

    checkpoints = [
        "",
        "checkpoint-24438",
        "checkpoint-48876",
        "checkpoint-73314"
    ]

    for ckpt in checkpoints:

        free_gpu()

        if ckpt == "":
            model_path = args.model_name
        else:
            model_path = f"{args.model_name}/{ckpt}"

        evaluate_model(
            model_path,
            args.output_dir,
            dtype,
            args.use_lora
        )


if __name__ == "__main__":
    main()
