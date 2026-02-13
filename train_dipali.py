# ============================================================
# 0. IMPORTS
# ============================================================
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from difflib import SequenceMatcher
from datasets import load_dataset, Value
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ============================================================
# 1. REPRODUCIBILITY (STRICT)
# ============================================================
set_seed(42)

# ============================================================
# 2. CONFIG
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"

OUTPUT_DIR = Path("./gemma3_outputs")
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRAIN_SAMPLES = None   # set int for debugging
MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_SEQ_LEN = MAX_SRC_LEN + MAX_TGT_LEN

# ============================================================
# 3. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 4. STRICT DATA FILTERING (ANTI-CHEATING)
# ============================================================
def strict_filter(example):
    s = str(example["src_txt"] or "").lower()
    t = str(example["tgt_txt"] or "").lower()
    sim = SequenceMatcher(None, s, t).ratio()
    return sim < 0.65

def length_filter(example):
    if not isinstance(example["src_txt"], str) or not isinstance(example["tgt_txt"], str):
        return False

    src_len = len(tokenizer(example["src_txt"], truncation=False)["input_ids"])
    tgt_len = len(tokenizer(example["tgt_txt"], truncation=False)["input_ids"])
    return (src_len <= MAX_SRC_LEN) and (tgt_len <= MAX_TGT_LEN)

def clean_utf8(example):
    s = example["src_txt"]
    t = example["tgt_txt"]
    example["src_txt"] = s.decode("utf-8", errors="ignore") if s is not None else ""
    example["tgt_txt"] = t.decode("utf-8", errors="ignore") if t is not None else ""
    return example

# ============================================================
# 5. LOAD OFFICIAL TRAIN + DEV SPLITS
# ============================================================
print("Loading Pralekha train & dev splits...")

train_raw = load_dataset(DATASET_NAME, "train", split="eng_hin")
dev_raw   = load_dataset(DATASET_NAME, "dev",   split="eng_hin")

def preprocess(ds, desc):
    print(f"\nProcessing {desc} split...")

    ds = ds.cast_column("src_txt", Value("binary"))
    ds = ds.cast_column("tgt_txt", Value("binary"))

    ds = ds.map(clean_utf8, num_proc=32, desc="UTF-8 cleaning")

    ds = ds.cast_column("src_txt", Value("string"))
    ds = ds.cast_column("tgt_txt", Value("string"))

    ds = ds.filter(lambda x: x["src_txt"].strip(), num_proc=32, desc="Remove empty src")
    ds = ds.filter(lambda x: x["tgt_txt"].strip(), num_proc=32, desc="Remove empty tgt")

    ds = ds.filter(strict_filter, num_proc=32, desc="Strict similarity filter")
    ds = ds.filter(length_filter, desc="Length filter")

    return ds

train_raw = preprocess(train_raw, "TRAIN")
dev_raw   = preprocess(dev_raw,   "DEV")

if MAX_TRAIN_SAMPLES is not None:
    train_raw = train_raw.shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))

print(f"\nFinal sizes → Train: {len(train_raw)} | Dev: {len(dev_raw)}")

# ============================================================
# 6. BIDIRECTIONAL PROMPT FORMAT (STRICT PARITY)
# ============================================================
def format_fn(batch):
    prompts, completions = [], []

    for i in range(len(batch["src_txt"])):
        if i % 2 == 0:
            instr, src, tgt = "Translate to HINDI DEVANAGARI:", batch["src_txt"][i], batch["tgt_txt"][i]
        else:
            instr, src, tgt = "Translate to ENGLISH:", batch["tgt_txt"][i], batch["src_txt"][i]

        prompts.append(
            f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
        )
        completions.append(f"{tgt}<end_of_turn>")

    return {"prompt": prompts, "completion": completions}

train_ds = train_raw.map(format_fn, batched=True, remove_columns=train_raw.column_names)
dev_ds   = dev_raw.map(format_fn,   batched=True, remove_columns=dev_raw.column_names)

# ============================================================
# 7. MODEL + LoRA
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
    bias="none"
)

# ============================================================
# 8. TRAINER (OFFICIAL DEV SET)
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    peft_config=peft_config,
    args=SFTConfig(
        output_dir=str(CKPT_DIR),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=400,

        bf16=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=10,

        max_length=MAX_SEQ_LEN,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,

        completion_only_loss=True,
        packing=False,
        report_to="none",
        ddp_find_unused_parameters=False
    )
)

trainer.train()

# ============================================================
# 9. LOSS CURVES
# ============================================================
logs = trainer.state.log_history
train_loss = [(x["step"], x["loss"]) for x in logs if "loss" in x]
val_loss   = [(x["step"], x["eval_loss"]) for x in logs if "eval_loss" in x]

plt.figure()
plt.plot(*zip(*train_loss), label="Train Loss")
plt.plot(*zip(*val_loss), label="Dev Loss")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training vs Dev Loss")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png")
plt.close()

# ============================================================
# 10. FINAL MERGED MODEL SAVE
# ============================================================
merged_model = trainer.model.merge_and_unload()
merged_model = merged_model.to("cpu").eval()

FINAL_MODEL_DIR = OUTPUT_DIR / "final_merged"
FINAL_MODEL_DIR.mkdir(exist_ok=True)

merged_model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print("\n✅ TRAINING COMPLETE (OFFICIAL DEV SET USED)")
print(f"📁 Checkpoints: {CKPT_DIR}")
print(f"📦 Final model: {FINAL_MODEL_DIR}")
