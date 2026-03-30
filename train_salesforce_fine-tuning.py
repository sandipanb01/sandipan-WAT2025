# ============================================================
# XML DOCUMENT MACHINE TRANSLATION — TRAINING
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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "google/gemma-3-4b-it"

DATA_ROOT = "localization-xml-mt"

LANG_PAIRS = ["ende", "enfr", "ennl", "enfi", "enru"]

OUTPUT_DIR = "./xml_mt_lora"

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 512
EVAL_EVERY = 1000

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ============================================================
# LANGUAGE MAP
# ============================================================

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

    base = os.path.join(root, "data", lang_pair)

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

    src_all = []
    tgt_all = []
    lang_all = []

    for lp in LANG_PAIRS:

        src, tgt = load_split(DATA_ROOT, lp, split)
        lang = LANG_CODE_MAP[lp]

        for s, t in zip(src, tgt):

            src_all.append(s)
            tgt_all.append(t)
            lang_all.append(lang)

    return Dataset.from_dict({
        "src_txt": src_all,
        "tgt_txt": tgt_all,
        "lang": lang_all,
    })

# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# FORMAT EXAMPLES 
# ============================================================

def format_example(example):

    target_lang = example["lang"]

    messages = {
        "prompt": [
            {
                "role": "user",
                "content":
                f"Translate the following XML document from English to {target_lang}.\n\n"
                f"English XML:\n{example['src_txt']}",
            }
        ],
        "completion": [
            {
                "role": "assistant",
                "content": example["tgt_txt"]
            }
        ],
    }

    return messages

# ============================================================
# LOAD DATASETS
# ============================================================

print("Loading datasets...")

train_dataset = build_dataset("train")
dev_dataset = build_dataset("dev")

# ============================================================
# REMOVE EMPTY EXAMPLES 
# ============================================================

def filter_empty(example):
    return (
        example["src_txt"] is not None and
        example["tgt_txt"] is not None and
        len(example["src_txt"].strip()) > 0 and
        len(example["tgt_txt"].strip()) > 0
    )

# ============================================================
# FORMAT DATASET
# ============================================================
train_dataset = train_dataset.filter(filter_empty)
dev_dataset = dev_dataset.filter(filter_empty)
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

# ============================================================
# MODEL
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)
model.config.use_cache = False

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
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

# ============================================================
# TRAINING CONFIG
# ============================================================

training_args = SFTConfig(

    output_dir=OUTPUT_DIR,
    
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,

    num_train_epochs=2,

    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    logging_steps=500,
    save_steps=EVAL_EVERY,

    bf16=True,

    max_length=MAX_SEQ_LENGTH,

    gradient_checkpointing=True,
    max_grad_norm=1.0,

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
