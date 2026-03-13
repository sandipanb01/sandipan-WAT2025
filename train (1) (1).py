# ============================================================
# TRAIN SCRIPT — STRICT (ADVISOR-ALIGNED) — 2-GPU READY
# ============================================================

import os
import json

import torch
torch.backends.cuda.matmul.allow_tf32 = True
from pathlib import Path
from difflib import SequenceMatcher

from datasets import load_dataset, Dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================
set_seed(42)

# ============================================================
# 2. CONFIG
# ============================================================
MODEL_ID     = "google/gemma-3-4b-it"
DATASET_NAME = "ai4bharat/pralekha"

SRC_LANG = "eng"
TGT_LANG = "hin"

OUTPUT_DIR      = Path("./gemma3_outputs")
CKPT_DIR        = OUTPUT_DIR / "checkpoints"
FINAL_MODEL_DIR = OUTPUT_DIR / "final_merged"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(exist_ok=True)

MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_SEQ_LEN = MAX_SRC_LEN + MAX_TGT_LEN

# Formatted dataset cache paths
FINAL_TRAIN_DS = Path("final_train_ds")
FINAL_DEV_DS   = Path("final_dev_ds")

# Raw cleaned dataset cache paths
DATA_CACHE_DIR = Path("./cached_datasets")
TRAIN_CACHE    = DATA_CACHE_DIR / "train_clean"
DEV_CACHE      = DATA_CACHE_DIR / "dev_clean"

# ============================================================
# 3. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 4. FILTERS (kept for reference, used during raw load)
# ============================================================
def strict_filter(example):
    s = str(example["src_txt"] or "").lower()
    t = str(example["tgt_txt"] or "").lower()
    return SequenceMatcher(None, s, t).ratio() < 0.65

def length_filter(example):
    src_len = len(tokenizer(example["src_txt"], truncation=False)["input_ids"])
    tgt_len = len(tokenizer(example["tgt_txt"], truncation=False)["input_ids"])
    return src_len <= MAX_SRC_LEN and tgt_len <= MAX_TGT_LEN

# ============================================================
# 5. RAW DATA LOADER
# ============================================================
import re

def sanitize(text):
    # Encode to bytes as latin-1 (accepts anything), then decode only valid utf-8
    # This strips all non-utf-8 bytes cleanly
    raw = text.encode("latin-1", "replace")
    clean = raw.decode("utf-8", "ignore")
    # Remove null bytes and other control chars (keep newlines/tabs)
    clean = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', clean)
    return clean.strip()

def load_split(split_name):
    stream_ds = load_dataset(
        DATASET_NAME,
        data_dir=split_name,
        split="train",
        streaming=True
    )

    clean_examples = []
    for example in stream_ds:
        try:
            if (
                example["src_lang"] == SRC_LANG
                and example["tgt_lang"] == TGT_LANG
            ):
                src = example["src_txt"]
                tgt = example["tgt_txt"]

                if isinstance(src, str) and isinstance(tgt, str):
                    src = sanitize(src)
                    tgt = sanitize(tgt)

                    if src and tgt:
                        clean_examples.append({
                            "src_lang": SRC_LANG,
                            "tgt_lang": TGT_LANG,
                            "src_txt":  src,
                            "tgt_txt":  tgt,
                        })
        except Exception:
            continue

    print(f"Loaded {len(clean_examples)} clean examples for {split_name}")
    return Dataset.from_list(clean_examples)

# ============================================================
# 6. BIDIRECTIONAL PROMPT FORMATTER
# ============================================================
def format_fn(batch):
    prompts     = []
    completions = []

    for i, (src, tgt) in enumerate(zip(batch["src_txt"], batch["tgt_txt"])):
        try:
            if i % 2 == 0:
                instr       = "Translate to HINDI DEVANAGARI:"
                input_text  = src
                output_text = tgt
            else:
                instr       = "Translate to ENGLISH:"
                input_text  = tgt
                output_text = src

            prompt = (
                f"<start_of_turn>user\n{instr}\n"
                f"{input_text}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            completion = f"{output_text}<end_of_turn>"

            prompts.append(prompt)
            completions.append(completion)

        except Exception:
            continue

    return {"prompt": prompts, "completion": completions}

# ============================================================
# 7. LOAD OR BUILD DATASETS  (smart caching — no redundant work)
# ============================================================

# --- If final formatted datasets already exist, load them directly ---
if FINAL_TRAIN_DS.exists() and FINAL_DEV_DS.exists():
    print("✅ Loading pre-formatted datasets from disk (skipping all preprocessing)...")
    train_ds = load_from_disk(str(FINAL_TRAIN_DS))
    dev_ds   = load_from_disk(str(FINAL_DEV_DS))

else:
    # --- Load or build raw cleaned datasets ---
    if TRAIN_CACHE.exists() and DEV_CACHE.exists():
        print("Loading cleaned raw datasets from disk...")
        train_raw = load_from_disk(str(TRAIN_CACHE))
        dev_raw   = load_from_disk(str(DEV_CACHE))
    else:
        print("Building cleaned datasets (first time only)...")
        train_raw = load_split("train")
        dev_raw   = load_split("dev")
        DATA_CACHE_DIR.mkdir(exist_ok=True)
        train_raw.save_to_disk(str(TRAIN_CACHE))
        dev_raw.save_to_disk(str(DEV_CACHE))

    print(f"Train: {len(train_raw)} | Dev: {len(dev_raw)}")

    # --- Format ---
    print("Formatting datasets...")
    train_ds = train_raw.map(format_fn, batched=True, remove_columns=train_raw.column_names)
    dev_ds   = dev_raw.map(format_fn,   batched=True, remove_columns=dev_raw.column_names)

    # --- Save formatted datasets for future runs ---
    train_ds.save_to_disk(str(FINAL_TRAIN_DS))
    dev_ds.save_to_disk(str(FINAL_DEV_DS))
    print("✅ Formatted datasets saved to disk.")

print(f"Train examples: {len(train_ds)} | Dev examples: {len(dev_ds)}")

# ============================================================
# 8. MODEL
# ============================================================
print("Clearing CUDA cache before model load...")
torch.cuda.empty_cache()
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,   # ✅ fixed: was "dtype"
)

# ============================================================
# 9. LoRA CONFIG
# ============================================================
peft_config = LoraConfig(
    r=32,
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
# 10. TRAINER
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
        eval_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=10,
        max_length=MAX_SEQ_LEN,
        completion_only_loss=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        packing=False,
        report_to="none",
        ddp_find_unused_parameters= True,   # ✅ required for DDP + gradient checkpointing
    )
)

# ============================================================
# 11. TRAIN
# ============================================================
print("🚀 Starting training...")
trainer.train(resume_from_checkpoint=True)

# ============================================================
# 12. MERGE AND SAVE FINAL MODEL
# ============================================================
# Only rank 0 should save in DDP
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank == 0:
    print("Merging and saving final model...")
    merged_model = trainer.model.merge_and_unload().eval()
    merged_model.save_pretrained(FINAL_MODEL_DIR)
    tokenizer.save_pretrained(FINAL_MODEL_DIR)
    print(f"✅ TRAINING COMPLETE — FINAL MODEL SAVED to {FINAL_MODEL_DIR}")