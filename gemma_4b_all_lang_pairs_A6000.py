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

# ============================================================
# TRAIN / EVAL SPLITS & MAX SAMPLES
# ============================================================
TRAIN_CONFIG = "train"
EVAL_CONFIG  = "test"

MAX_TRAIN_SAMPLES = 10  # None = use full data
EVAL_SAMPLES      = 10

MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400

# ----------------------------
# Load tokenizer early
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ----------------------------
# STRICT FILTERS
# ----------------------------
def strict_filter(example):
    sim = SequenceMatcher(
        None,
        example["src_txt"].lower(),
        example["tgt_txt"].lower()
    ).ratio()
    return sim < 0.65

def length_filter(example):
    src_len = len(tokenizer(example["src_txt"], add_special_tokens=True, truncation=False)["input_ids"])
    tgt_len = len(tokenizer(example["tgt_txt"], add_special_tokens=True, truncation=False)["input_ids"])
    return (src_len <= MAX_SRC_LEN) and (tgt_len <= MAX_TGT_LEN)

# ============================================================
# LANGUAGE PAIRS
# ============================================================
LANG_PAIRS = ['eng_hin', 'eng_ben', 'eng_guj', 'eng_kan', 'eng_mal',
              'eng_mar', 'eng_ori', 'eng_pan', 'eng_tam', 'eng_tel', 'eng_urd']

# ============================================================
# LOAD TRAIN DATA (ALL LANGUAGE PAIRS)
# ============================================================
train_datasets = []

for lp in LANG_PAIRS:
    raw_ds = load_dataset(DATASET_NAME, TRAIN_CONFIG, split=lp)
    raw_ds = raw_ds.add_column("lang_pair", [lp]*len(raw_ds))
    
    # Strict filtering
    filtered_ds = raw_ds.filter(strict_filter).filter(length_filter)
    
    # Apply MAX_TRAIN_SAMPLES cap
    t_limit = MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES is not None else len(filtered_ds)
    train_set_lp = filtered_ds.shuffle(seed=42).select(range(t_limit))
    
    train_datasets.append(train_set_lp)

# Concatenate all language pairs
train_set = concatenate_datasets(train_datasets)

# ============================================================
# LOAD TEST DATA (ALL LANGUAGE PAIRS)
# ============================================================
test_datasets = []

for lp in LANG_PAIRS:
    raw_ds = load_dataset(DATASET_NAME, EVAL_CONFIG, split=lp)
    raw_ds = raw_ds.add_column("lang_pair", [lp]*len(raw_ds))
    
    # Only length filter for test
    filtered_ds = raw_ds.filter(length_filter)
    
    # Apply EVAL_SAMPLES cap
    e_limit = EVAL_SAMPLES if EVAL_SAMPLES is not None else len(filtered_ds)
    test_set_lp = filtered_ds.shuffle(seed=99).select(range(e_limit))
    
    test_datasets.append(test_set_lp)

# Concatenate all language pairs
test_set = concatenate_datasets(test_datasets)

print(f"✅ Loaded {len(LANG_PAIRS)} language pairs")
print(f"Train samples: {len(train_set)} | Test samples: {len(test_set)}")

# ============================================================
# MODEL + LoRA
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
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
        max_length=1024,
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
