# ======================================================
# ✅ Universal Fine-tuning + Evaluation for HF instruct/causal LM
# (Streaming, LoRA, Fast Evaluation, Metrics, JSONL + ZIP)
# ✅ STRICTLY MODIFIED FOR 2 FULL EPOCHS
# ======================================================

import os, json, zipfile, torch, random
from pathlib import Path
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, apply_chat_template
from tqdm import tqdm
import sacrebleu
from trl import apply_chat_template
import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from functools import partial
from datasets import load_dataset
from tqdm import tqdm
import sacrebleu
import json
import zipfile
from pathlib import Path
import matplotlib.pyplot as plt   # 🔹 ADDED

# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./universal_output_best")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 512

BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 2 #NUMBER OF EPOCHS
EVAL_BATCH_SIZE = 8

FULL_DATASET = False
MAX_COLAB_SAMPLES = 100

DIRECTIONS = ["eng_hin", "hin_eng"]

# 🔹 TOKEN STATISTICS CONTAINERS
INPUT_TOKEN_LENS = []
OUTPUT_TOKEN_LENS = []

# ------------------------------ MODEL PREP
def prepare_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    return model, tokenizer

# ------------------------------ STREAM + TOKENIZE
def stream_examples_list(max_samples=None, tokenizer=None):
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"
    split = get_dataset_split_names(dataset_name, config_name)[0]

    def build_prompt(example):
        prompt = (
            f"Translate this {example['src_lang']} text to "
            f"{example['tgt_lang']}:\n{example['src_txt']}"
        )
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": example["tgt_txt"]}
            ]
        }

    dataset = load_dataset(
        dataset_name,
        name=config_name,
        split=split,
        streaming=True
    )

    if not FULL_DATASET and max_samples is not None:
        dataset = dataset.take(max_samples)

    dataset = dataset.map(build_prompt)
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer}
    )

    return dataset

# ------------------------------ TRAINING
def train_model(max_samples=None):
    model, tokenizer = prepare_model()
    dataset = stream_examples_list(max_samples, tokenizer)

    peft_config = LoraConfig(
        r=256,
        lora_alpha=16,
        target_modules="all-linear"
    )

    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        completion_only_loss=True,
        packing=False,
        max_seq_length=MAX_SEQ_LEN
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    return model, tokenizer

# ============================== EVALUATION ==============================

def build_eval_prompt_messages(example, src_lang, tgt_lang):
    user_prompt = (
        f"Translate this {src_lang} text to {tgt_lang}:\n{example['src_txt']}"
    )
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": ""}
    ]

class EvalDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, src_lang, tgt_lang):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __iter__(self):
        for ex in self.dataset:
            if self.src_lang == "eng":
                src_text, ref_text = ex["src_txt"], ex["tgt_txt"]
            else:
                src_text, ref_text = ex["tgt_txt"], ex["src_txt"]

            messages = build_eval_prompt_messages(
                {"src_txt": src_text}, self.src_lang, self.tgt_lang
            )

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )

            yield {
                "input_ids": torch.tensor(input_ids),
                "reference": ref_text.strip()
            }

def eval_collate_fn(batch, tokenizer):
    input_ids = [x["input_ids"] for x in batch]
    refs = [x["reference"] for x in batch]

    enc = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    )

    return enc["input_ids"], enc["attention_mask"], refs

# 🔹 MODIFIED: token statistics added
def generate_batch(model, tokenizer, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    preds = []
    for i in range(len(outputs)):
        prompt_len = attention_mask[i].sum().item()
        gen_ids = outputs[i][prompt_len:]

        INPUT_TOKEN_LENS.append(prompt_len)
        OUTPUT_TOKEN_LENS.append(len(gen_ids))

        preds.append(
            tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        )
    return preds

def load_pralekha_split():
    return load_dataset(
        "ai4bharat/Pralekha",
        name="train",
        split="eng_hin",
        streaming=True
    )

def evaluate_direction(model, tokenizer, src, tgt, max_samples=100):
    dataset = load_pralekha_split()
    eval_ds = EvalDataset(dataset, tokenizer, src, tgt)

    loader = DataLoader(
        eval_ds,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=partial(eval_collate_fn, tokenizer=tokenizer),
        num_workers=0
    )

    preds, refs, seen = [], [], 0
    for input_ids, attn, batch_refs in tqdm(loader, desc=f"{src}→{tgt}"):
        batch_preds = generate_batch(model, tokenizer, input_ids, attn)
        preds.extend(batch_preds)
        refs.extend(batch_refs)
        seen += len(batch_refs)
        if seen >= max_samples:
            break

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.metrics.CHRF(word_order=0).corpus_score(preds, [refs]).score
    print(f"{src}→{tgt} | BLEU={bleu:.2f} | chrF={chrf:.3f}")
    return preds, refs, bleu, chrf

# ============================== MAIN ==============================

if __name__ == "__main__":
    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES
    model, tokenizer = train_model(max_samples=max_samples)

    results = {}
    for split in DIRECTIONS:
        src, tgt = split.split("_")
        preds, refs, bleu, chrf = evaluate_direction(
            model, tokenizer, src, tgt, max_samples=max_samples
        )
        results[split] = {"BLEU": bleu, "chrF": chrf}

    print("\n✅ Final Results (ENG↔HIN):")
    for split, scores in results.items():
        print(f"{split}: BLEU={scores['BLEU']:.2f}, chrF={scores['chrF']:.3f}")

# ============================== JSONL EXPORT ==============================

directions = ["eng_hin", "hin_eng"]
max_samples_export = 100
batch_size = 8

jsonl_files = []

for split in directions:
    src, tgt = split.split("_")

    raw_ds = load_pralekha_split()
    eval_ds = EvalDataset(raw_ds, tokenizer, src, tgt)

    collate = partial(eval_collate_fn, tokenizer=tokenizer)
    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0
    )

    save_path = OUTPUT_DIR / f"{split}_pred_refs.jsonl"
    processed = 0

    with open(save_path, "w", encoding="utf-8") as f:
        for input_ids, attention_mask, refs in loader:
            preds = generate_batch(model, tokenizer, input_ids, attention_mask)

            for p, r in zip(preds, refs):
                f.write(json.dumps(
                    {"prediction": p, "reference": r},
                    ensure_ascii=False
                ) + "\n")

            processed += len(refs)
            if processed >= max_samples_export:
                break

    jsonl_files.append(save_path)
    print(f"Saved {processed} examples to {save_path}")

# ------------------ ZIP -----------------------------------------
zip_path = OUTPUT_DIR / "pred_refs_eng_hin.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for f in jsonl_files:
        zipf.write(f, arcname=f.name)

print(f"ZIP saved at: {zip_path}")

# ============================== TOKEN HISTOGRAMS ==============================

def plot_hist(data, title, xlabel, path):
    plt.figure()
    plt.hist(data, bins=50)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

if INPUT_TOKEN_LENS and OUTPUT_TOKEN_LENS:
    plot_hist(
        INPUT_TOKEN_LENS,
        "Input Prompt Token Length Distribution",
        "Tokens",
        OUTPUT_DIR / "input_token_histogram.png"
    )
    plot_hist(
        OUTPUT_TOKEN_LENS,
        "Generated Output Token Length Distribution",
        "Tokens",
        OUTPUT_DIR / "output_token_histogram.png"
    )
    print("📊 Token length histograms saved.")
