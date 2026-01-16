import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
from datasets import (
    load_dataset,
    get_dataset_config_names,
    concatenate_datasets
)
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import sacrebleu
import unicodedata
import matplotlib.pyplot as plt

# ============================================================
# SEED
# ============================================================
set_seed(42)

# ============================================================
# 1. CONFIGURATION
# ============================================================
MODEL_ID = "google/gemma-3-4b-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "./gemma3-4b-strict-bidirectional"

MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400

# Dataset caps (Set to None for full data)
MAX_TRAIN_SAMPLES = None   # Set to None 
EVAL_SAMPLES = None        # Set to None

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# STRICT FILTERS
# ============================================================
def strict_filter(example):
    return (
        SequenceMatcher(
            None,
            example["src_txt"].lower(),
            example["tgt_txt"].lower()
        ).ratio() < 0.65
    )

def length_filter(example):
    return (
        len(tokenizer(example["src_txt"], truncation=False)["input_ids"]) <= MAX_SRC_LEN
        and
        len(tokenizer(example["tgt_txt"], truncation=False)["input_ids"]) <= MAX_TGT_LEN
    )

# ============================================================
# LOAD ALL PRALEKHA LANGUAGE PAIRS (CORRECTLY)
# ============================================================
LANG_PAIRS = get_dataset_config_names(DATASET_NAME)

def load_all_pairs(split):
    all_sets = []
    for lp in LANG_PAIRS:
        ds = load_dataset(DATASET_NAME, lp, split=split)
        ds = ds.add_column("lang_pair", [lp] * len(ds))
        all_sets.append(ds)
    return concatenate_datasets(all_sets)

raw_train = load_all_pairs("train")
raw_test  = load_all_pairs("test")

# Apply strict filtering
train_set = raw_train.filter(strict_filter).filter(length_filter)
test_set  = raw_test.filter(length_filter)

# Proper sample caps
if MAX_TRAIN_SAMPLES is not None:
    train_set = train_set.select(range(min(MAX_TRAIN_SAMPLES, len(train_set))))

if EVAL_SAMPLES is not None:
    test_set = test_set.select(range(min(EVAL_SAMPLES, len(test_set))))

print(f"Loaded {len(LANG_PAIRS)} language pairs")
print(f"Train samples: {len(train_set)} | Test samples: {len(test_set)}")

# ============================================================
# MODEL + LoRA
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj","k_proj","v_proj",
        "o_proj","gate_proj","up_proj","down_proj"
    ],
    task_type="CAUSAL_LM",
    bias="none"
)

# ============================================================
# BIDIRECTIONAL PROMPTS
# ============================================================
def formatting_prompts_func(example):
    prompts, completions = [], []

    for i in range(len(example["src_txt"])):
        if i % 2 == 0:
            instr, src, tgt = (
                "Translate to INDIC LANGUAGE:",
                example["src_txt"][i],
                example["tgt_txt"][i]
            )
        else:
            instr, src, tgt = (
                "Translate to ENGLISH:",
                example["tgt_txt"][i],
                example["src_txt"][i]
            )

        prompts.append(
            f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
        )
        completions.append(f"{tgt}<end_of_turn>")

    return {"prompt": prompts, "completion": completions}

train_set = train_set.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=train_set.column_names
)

# ============================================================
# TRAINING
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_set,
    peft_config=peft_config,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        #max_length=4800,
        max_length=2048,
        learning_rate=5e-5,
        num_train_epochs=2,
        logging_steps=10,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        completion_only_loss=True,
        save_strategy="no",
        gradient_checkpointing=True,
        report_to="none"
    ),
) #mx_length=MAX_SRC_LENGTH + MAX+TGT_LENGTH

trainer.train()

# ============================================================
# SAVE MODEL
# ============================================================
model = trainer.model.merge_and_unload()
model.eval()
model.save_pretrained(f"{OUTPUT_DIR}/final_merged")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_merged")

# ============================================================
# TRAINING LOSS PLOT
# ============================================================
losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
plt.figure()
plt.plot(losses)
plt.xlabel("Logging Step")
plt.ylabel("Training Loss")
plt.title("Gemma-3-4B Training Loss")
plt.savefig(f"{OUTPUT_DIR}/training_loss.jpg")
plt.close()

# ============================================================
# SCRIPT-LEVEL INDIC LID (ALL SCRIPTS)
# ============================================================
INDIC_SCRIPTS = [
    "DEVANAGARI","BENGALI","GURMUKHI","GUJARATI",
    "ORIYA","TAMIL","TELUGU","KANNADA","MALAYALAM","SINHALA"
]

def is_indic_script(text):
    for ch in text:
        try:
            if any(s in unicodedata.name(ch) for s in INDIC_SCRIPTS):
                return True
        except ValueError:
            continue
    return False

# ============================================================
# EVALUATION
# ============================================================
results = []

for sample in tqdm(test_set):
    pairs = [
        ("ENG_to_INDIC", "Translate to INDIC LANGUAGE:", sample["src_txt"], sample["tgt_txt"]),
        ("INDIC_to_ENG", "Translate to ENGLISH:", sample["tgt_txt"], sample["src_txt"]),
    ]

    for mode, instr, src, ref in pairs:
        prompt = f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1
            ) #max_new_tokens=MAX_TGT_LENGTH

        pred = tokenizer.decode(
            output[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        results.append({
            "lang_pair": sample["lang_pair"],
            "mode": mode,
            "source": src,
            "reference": ref,
            "prediction": pred
        })

# ============================================================
# LID ACCURACY (PER LANGUAGE × DIRECTION)
# ============================================================
lid_rows = []

for lp in LANG_PAIRS:
    for direction in ["ENG_to_INDIC", "INDIC_to_ENG"]:
        subset = [r for r in results if r["lang_pair"] == lp and r["mode"] == direction]
        if not subset:
            continue

        correct = []
        for r in subset:
            pred_is_indic = is_indic_script(r["prediction"])
            correct.append(pred_is_indic if direction == "ENG_to_INDIC" else not pred_is_indic)

        lid_rows.append({
            "Language Pair": lp,
            "Direction": direction,
            "LID Accuracy (%)": round(np.mean(correct) * 100, 2),
            "Samples": len(subset)
        })

lid_df = pd.DataFrame(lid_rows)
lid_path = f"{OUTPUT_DIR}/final_lid_accuracy.xlsx"
lid_df.to_excel(lid_path, index=False)

print(f"\n📊 LID accuracy saved to: {lid_path}")

# ============================================================
# EXPORT JSONL
# ============================================================
out_dir = Path("exports_jsonl")
out_dir.mkdir(exist_ok=True)

with open(out_dir / "translations_all_langpairs.jsonl", "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("✅ JSONL export complete")
