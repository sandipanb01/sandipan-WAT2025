# ============================================================
# TRAIN SCRIPT — SIMPLE (ADVISOR STYLE)
# ============================================================

import torch
from pathlib import Path

from datasets import load_dataset
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
MODEL_ID = "google/gemma-3-4b-it"
DATASET_NAME = "ai4bharat/pralekha"

SRC_LANG = "eng"
TGT_LANG = "hin"

OUTPUT_DIR = Path("./gemma3_outputs")
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
FINAL_MODEL_DIR = OUTPUT_DIR / "final_merged"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
FINAL_MODEL_DIR.mkdir(exist_ok=True)

MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_SEQ_LEN = MAX_SRC_LEN + MAX_TGT_LEN

# ============================================================
# 3. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 4. LOAD DATA (ADVISOR WAY — SIMPLE)
# ============================================================
def load_split(split_name):
    ds = load_dataset(
        DATASET_NAME,
        data_dir=split_name,
        split="train"
    )

    ds = ds.filter(
        lambda x: (
            x["src_lang"] == SRC_LANG and
            x["tgt_lang"] == TGT_LANG and
            x["src_txt"].strip() and
            x["tgt_txt"].strip()
        ),
        num_proc=4
    )
    return ds

print("Loading datasets...")
train_raw = load_split("train")
dev_raw   = load_split("dev")

print(f"Train: {len(train_raw)} | Dev: {len(dev_raw)}")

# ============================================================
# 5. BIDIRECTIONAL PROMPTING (UNCHANGED)
# ============================================================
def format_fn(batch):
    prompts, completions = [], []
    for i in range(len(batch["src_txt"])):
        if i % 2 == 0:
            instr, src, tgt = (
                "Translate to HINDI DEVANAGARI:",
                batch["src_txt"][i],
                batch["tgt_txt"][i],
            )
        else:
            instr, src, tgt = (
                "Translate to ENGLISH:",
                batch["tgt_txt"][i],
                batch["src_txt"][i],
            )

        prompts.append(
            f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
        )
        completions.append(f"{tgt}<end_of_turn>")

    return {"prompt": prompts, "completion": completions}

train_ds = train_raw.map(format_fn, batched=True, remove_columns=train_raw.column_names)
dev_ds   = dev_raw.map(format_fn,   batched=True, remove_columns=dev_raw.column_names)

# ============================================================
# 6. MODEL + LoRA
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
    bias="none",
)

# ============================================================
# 7. TRAINING
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
    ),
)

trainer.train()

# ============================================================
# 8. MERGE FINAL MODEL
# ============================================================
model = trainer.model.merge_and_unload().eval()
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

print("✅ TRAINING COMPLETE — FINAL MODEL SAVED")
