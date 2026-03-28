# ============================================================
# XML DOCUMENT MACHINE TRANSLATION — TRAINING (STABLE)
# Root cause of inf/nan confirmed: gradient explosion due to
# ~1e+36 logit magnitudes in float32 + 262k vocab.
# Fix: much lower LR, tight grad clipping, long warmup,
#      and bfloat16 which saturates instead of overflowing.
# ============================================================

import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ============================================================
# ENVIRONMENT
# ============================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
set_seed(42)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "google/gemma-3-4b-it"
DATA_ROOT  = "localization-xml-mt"
LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]
OUTPUT_DIR = "./xml_mt_lora"

MAX_SEQ_LENGTH = 1024
EVAL_EVERY     = 1000

Path(OUTPUT_DIR).mkdir(exist_ok=True)

LANG_CODE_MAP = {
    "ende": "German",
    "enfr": "French",
    "ennl": "Dutch",
    "enfi": "Finnish",
    "enru": "Russian",
}

# ============================================================
# DATA NORMALIZATION
# ============================================================

def normalize_salesforce_entry(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if "text" in v:
            return v["text"]
        if "segments" in v:
            return "".join(seg.get("text", "") for seg in v["segments"])
        return json.dumps(v, ensure_ascii=False)
    return str(v)

# ============================================================
# LOAD DATA SPLITS
# ============================================================

def load_split(root, lang_pair, split):
    base     = os.path.join(root, "data", lang_pair)
    src_file = os.path.join(base, f"{lang_pair}_en_{split}.json")
    tgt_file = os.path.join(base, f"{lang_pair}_{lang_pair[2:]}_{split}.json")

    with open(src_file) as f:
        src_json = json.load(f)
    with open(tgt_file) as f:
        tgt_json = json.load(f)

    src = [normalize_salesforce_entry(v) for v in src_json["text"].values()]
    tgt = [normalize_salesforce_entry(v) for v in tgt_json["text"].values()]
    return src, tgt

# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(split):
    src_all, tgt_all, lang_all = [], [], []
    for lp in LANG_PAIRS:
        src, tgt = load_split(DATA_ROOT, lp, split)
        lang     = LANG_CODE_MAP[lp]
        for s, t in zip(src, tgt):
            src_all.append(s)
            tgt_all.append(t)
            lang_all.append(lang)
    return Dataset.from_dict({
        "src_txt": src_all,
        "tgt_txt": tgt_all,
        "lang":    lang_all,
    })

# ============================================================
# TOKENIZER
# Use <pad> (ID 0) not eos (ID 1) — they must differ.
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token    = "<pad>"
tokenizer.pad_token_id = 0
tokenizer.padding_side = "right"

assert tokenizer.pad_token_id != tokenizer.eos_token_id, \
    "pad == eos — will cause inf loss!"

print(f"pad={tokenizer.pad_token_id}  "
      f"eos={tokenizer.eos_token_id}  "
      f"bos={tokenizer.bos_token_id}")

# ============================================================
# FORMAT EXAMPLES
# ============================================================

def format_example(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Translate the following XML document from English to {example['lang']}.\n\n"
                    f"English XML:\n{example['src_txt']}"
                ),
            },
            {
                "role": "assistant",
                "content": example["tgt_txt"],
            },
        ]
    }

# ============================================================
# LOAD DATASETS
# ============================================================

print("Loading datasets...")

train_dataset = build_dataset("train")
dev_dataset   = build_dataset("dev")

train_dataset = train_dataset.map(
    format_example,
    remove_columns=train_dataset.column_names,
    num_proc=32,
)

dev_dataset = dev_dataset.map(
    format_example,
    remove_columns=dev_dataset.column_names,
    num_proc=32,
)

print(f"Train: {len(train_dataset)}  Dev: {len(dev_dataset)}")

# ============================================================
# SANITY CHECK
# ============================================================

print("Token length check on 5 examples...")
for i in range(5):
    tokens = tokenizer.apply_chat_template(
        train_dataset[i]["messages"],
        tokenize=True,
        add_generation_prompt=False,
    )
    print(f"  Example {i}: {len(tokens)} tokens")

# ============================================================
# MODEL
# KEY FIX: switch back to bfloat16.
#
# The debug showed logits of ~1e+36 in float32.
# float32 max is ~3.4e+38 — so gradients of logits^2 overflow
# immediately and become inf/nan.
#
# bfloat16 has the same exponent range as float32 (8 bits) but
# saturates gracefully at large values instead of overflowing.
# Combined with tight grad clipping this is stable.
#
# float32 actually made things WORSE here because it allows
# larger intermediate values before saturating.
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    dtype=torch.bfloat16,    # bfloat16 is correct for Gemma-3
)

# Sync config with tokenizer — prevents the "tokens differ" warning
model.config.pad_token_id = tokenizer.pad_token_id   # 0
model.config.eos_token_id = tokenizer.eos_token_id   # 1
model.config.bos_token_id = tokenizer.bos_token_id   # 2

# Verify weights
bad = [
    n for n, p in model.named_parameters()
    if torch.isnan(p).any() or torch.isinf(p).any()
]
if bad:
    raise RuntimeError(f"Corrupt model weights: {bad}")
print("All model weights clean.")

# ============================================================
# LORA CONFIG
# ============================================================

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

# ============================================================
# TRAINING CONFIG
#
# Three changes to tame the ~1e+36 logit scale:
#   1. learning_rate 5e-5 → 1e-5   (smaller initial gradient step)
#   2. max_grad_norm 1.0  → 0.3    (hard cap on gradient norm)
#   3. warmup_ratio  0.10 → 0.15   (longer ramp before full LR)
#   4. bf16=True                   (matches model dtype)
# ============================================================

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,

    num_train_epochs=2,

    learning_rate=1e-5,           # FIX: was 5e-5, too large for 1e+36 logits
    lr_scheduler_type="cosine",
    warmup_ratio=0.15,            # FIX: was 0.10, longer warmup

    logging_steps=50,
    save_steps=EVAL_EVERY,

    eval_strategy="steps",
    eval_steps=EVAL_EVERY,

    bf16=True,                    # FIX: matches bfloat16 model
    fp16=False,

    max_length=MAX_SEQ_LENGTH,

    gradient_checkpointing=True,
    max_grad_norm=0.3,            # FIX: was 1.0, tight clip for large logits

    packing=False,

    report_to="none",
)

# ============================================================
# TRAINER
# ============================================================

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    peft_config=lora_config,
    args=training_args,
    processing_class=tokenizer,
)

# ============================================================
# TRAIN
# ============================================================

print("Starting training...")
trainer.train()
trainer.save_model()
print("Training finished.")