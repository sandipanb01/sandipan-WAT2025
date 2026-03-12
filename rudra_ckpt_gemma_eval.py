# ===============================
# INSTALL DEPENDENCIES
# ===============================

!pip install -q huggingface_hub


# ===============================
# IMPORTS
# ===============================

import json
import os
import re
from tqdm import tqdm
import torch
import sacrebleu

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM
from huggingface_hub import snapshot_download


# ===============================
# SETTINGS
# ===============================

REPO_ID = "ibm-iitr-mt-research/gemma-3-1b-it_wat_sft"

OUTPUT_DIR = "./output"

USE_LORA = True
FP16 = True

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# DOWNLOAD MODEL FROM HF
# ===============================

print("Downloading model from HuggingFace...")

ckpt_root = snapshot_download(REPO_ID)

print("Model downloaded to:")
print(ckpt_root)


# ===============================
# UTILITIES
# ===============================

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


# ===============================
# PROMPT BUILDER
# ===============================

def build_prompt_wat(example, tokenizer):

    ref = example["tgt_txt"]
    tgt_lang = example["tgt_lang"]

    lang_map = {
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
        "urd": "Urdu"
    }

    target_lang = lang_map[tgt_lang]

    # EXACT STRUCTURE FROM ADVISOR
    messages = {
        "prompt": [
            {
                "role": "user",
                "content": f"Translate the following sentence from English to {target_lang}.\n\n"
                f"English: {example['src_txt']}",
            }
        ],
        "completion": [
            {"role": "assistant", "content": example["tgt_txt"]}
        ],
    }

    # For inference we only use the prompt
    prompt = tokenizer.apply_chat_template(
        messages["prompt"],
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


# ===============================
# EVALUATION
# ===============================

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


# ===============================
# LANGUAGE SPLITS
# ===============================

language_splits = [
    "eng_ben",
    "eng_guj",
    "eng_hin",
    "eng_kan",
    "eng_mal",
    "eng_mar",
    "eng_ori",
    "eng_pan",
    "eng_tam",
    "eng_tel",
    "eng_urd"
]


# ===============================
# FIND CHECKPOINTS
# ===============================

base_dir = ckpt_root

checkpoints = []

for name in sorted(os.listdir(base_dir)):

    if os.path.isdir(os.path.join(base_dir, name)) and name.startswith("checkpoint-"):
        checkpoints.append(name)

print("\nFound checkpoints:")
for ck in checkpoints:
    print(ck)


# ===============================
# EVALUATE CHECKPOINTS
# ===============================

for name in checkpoints:

    free_gpu()

    path = os.path.join(base_dir, name)

    match = re.match(r"checkpoint-(\d+)", name)
    ckpt_num = int(match.group(1))

    print("\n==============================")
    print(f"Evaluating checkpoint {ckpt_num}")
    print("==============================")

    if USE_LORA:

        print("Loading LoRA model")

        model = AutoPeftModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.float16 if FP16 else torch.float32,
        )

        model = model.merge_and_unload()

    else:

        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="auto",
            torch_dtype=torch.float16 if FP16 else torch.float32,
        )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    # ===============================
    # LOOP LANGUAGES
    # ===============================

    for split in language_splits:

        print(f"\nEvaluating {split}")

        dataset = load_dataset(
            "ai4bharat/Pralekha",
            "test",
            split=split
        )

        dataset = dataset.map(
            build_prompt_wat,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=32
        )

        references, predictions = evaluate_wat(
            model,
            tokenizer,
            dataset
        )

        bleu, chrf = calc_metrics(predictions, references)

        results = {
            "checkpoint": ckpt_num,
            "split": split,
            "BLEU": bleu,
            "CHRF": chrf
        }

        results_path = f"{OUTPUT_DIR}/{split}_ckpt_{ckpt_num}.json"

        with open(results_path, "w") as f:
            json.dump(results, f)

        print("Saved:", results_path)


print("\n✅ Evaluation complete.")
print("Results stored in:", OUTPUT_DIR)
