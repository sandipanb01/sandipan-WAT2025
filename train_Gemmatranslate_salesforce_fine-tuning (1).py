# ============================================================
# TRANSLATEGEMMA XML MT TRAINING (PAPER-ALIGNED)
# Based on:
# - TranslateGemma (2026)
# - AMTA 2024 XML MT
# ============================================================

import os
import json
import torch
import torch.distributed as dist
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────────────────────
# ENV
# ─────────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

set_seed(42)
local_rank = int(os.environ.get("LOCAL_RANK", 0))

def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL_NAME  = "google/translategemma-4b-it"
DATA_ROOT   = "localization-xml-mt"
LANG_PAIRS  = ["ende", "enfr", "ennl", "enfi", "enru"]
OUTPUT_DIR  = "./xml_mt_translategemma_lora"

MAX_SEQ_LEN = 1024
SAVE_EVERY  = 500

Path(OUTPUT_DIR).mkdir(exist_ok=True)

LANG_CODE_MAP = {
    "ende": "de", "enfr": "fr", "ennl": "nl",
    "enfi": "fi", "enru": "ru"
}

# ─────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

tokenizer.pad_token = "<pad>"
tokenizer.pad_token_id = 0
tokenizer.padding_side = "right"


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
def normalize(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if "text" in v:
            return v["text"]
        if "segments" in v:
            return "".join(s.get("text", "") for s in v["segments"])
    return str(v)


def load_split(lp, split):
    base = os.path.join(DATA_ROOT, "data", lp)

    with open(f"{base}/{lp}_en_{split}.json") as f:
        src_json = json.load(f)

    with open(f"{base}/{lp}_{lp[2:]}_{split}.json") as f:
        tgt_json = json.load(f)

    src = [normalize(v) for v in src_json["text"].values()]
    tgt = [normalize(v) for v in tgt_json["text"].values()]

    return src, tgt


# ─────────────────────────────────────────────────────────────
# BUILD DATASET (IMPORTANT)
# ─────────────────────────────────────────────────────────────
def build_dataset(split):
    texts = []

    for lp in LANG_PAIRS:
        src_list, tgt_list = load_split(lp, split)
        code = LANG_CODE_MAP[lp]

        for src, tgt in zip(src_list, tgt_list):

            text = (
                f"<xml_translate>\n"
                f"<source_lang=en>\n"
                f"<target_lang={code}>\n"
                f"<input>\n{src}\n</input>\n"
                f"<output>\n{tgt}\n</output>"
            )

            texts.append(text)

    return Dataset.from_dict({"text": texts})


train_ds = build_dataset("train")
dev_ds   = build_dataset("dev")


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map={"": local_rank},
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

model.config.pad_token_id = tokenizer.pad_token_id


# ─────────────────────────────────────────────────────────────
# LORA
# ─────────────────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
)


# ─────────────────────────────────────────────────────────────
# TRAINING CONFIG (FIXED DDP)
# ─────────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-5,
    bf16=True,
    max_length=MAX_SEQ_LEN,
    dataset_text_field="text",
    logging_steps=50,
    save_steps=SAVE_EVERY,

    # 🔥 CRITICAL FIX
    ddp_find_unused_parameters=True,

    gradient_checkpointing=True,
    report_to="none",
)


# ─────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    peft_config=lora_config,
    args=training_args,
    processing_class=tokenizer,
)


# ─────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────
if is_main():
    print("Starting training...")

trainer.train()

if is_main():
    trainer.save_model()