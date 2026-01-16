import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
import sacrebleu

# --- Dependencies Guard ---
def install_and_import(package):
    import subprocess, sys
    try: __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_and_import("langdetect")
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42
set_seed(42)

# ============================================================
# 1. CONFIGURATION
# ============================================================
MODEL_ID = "google/gemma-3-4b-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "./gemma3-4b-strict-bidirectional"

TRAIN_CONFIG = "train"
EVAL_CONFIG  = "test"

MAX_TRAIN_SAMPLES = None
EVAL_SAMPLES = None

MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400

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
    sim = SequenceMatcher(
        None,
        example["src_txt"].lower(),
        example["tgt_txt"].lower()
    ).ratio()
    return sim < 0.65

def length_filter(example):
    src_len = len(tokenizer(example["src_txt"], truncation=False)["input_ids"])
    tgt_len = len(tokenizer(example["tgt_txt"], truncation=False)["input_ids"])
    return src_len <= MAX_SRC_LEN and tgt_len <= MAX_TGT_LEN

# ============================================================
# LOAD ALL PRALEKHA LANGUAGE PAIRS (CORRECTLY)
# ============================================================
LANG_PAIRS = get_dataset_config_names(DATASET_NAME)

def load_all_pairs(split):
    datasets = []
    for lp in LANG_PAIRS:
        ds = load_dataset(DATASET_NAME, lp, split=split)
        ds = ds.add_column("lang_pair", [lp] * len(ds))
        datasets.append(ds)
    return concatenate_datasets(datasets)

raw_train = load_all_pairs("train")
raw_test  = load_all_pairs("test")

train_set = raw_train.filter(strict_filter).filter(length_filter)
test_set  = raw_test.filter(length_filter)

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
    r=64,
    lora_alpha=128,
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
            instr, src, tgt = "Translate to INDIC LANGUAGE:", example["src_txt"][i], example["tgt_txt"][i]
        else:
            instr, src, tgt = "Translate to ENGLISH:", example["tgt_txt"][i], example["src_txt"][i]

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
)# max_length=MAX_SRC_LENGTH+MAX_TGT_LENGTH

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
import matplotlib.pyplot as plt

losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
plt.figure()
plt.plot(losses)
plt.xlabel("Logging Step")
plt.ylabel("Training Loss")
plt.title("Gemma-3-4B Training Loss")
plt.savefig(f"{OUTPUT_DIR}/training_loss.jpg")
plt.close()

# ============================================================
# EVALUATION
# ============================================================
results = []
metrics = {"ENG_to_INDIC": {"preds": [], "refs": []},
           "INDIC_to_ENG": {"preds": [], "refs": []}}

import unicodedata

def is_indic(text):
    for ch in text:
        try:
            if "DEVANAGARI" in unicodedata.name(ch, ""):
                return True
        except ValueError:
            continue
    return False

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
                #max_new_tokens=MAX_TGT_LEN,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1
         ) # max_new_tokens=MAX_TGT_LEN

        pred = tokenizer.decode(
            output[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        results.append({
            "mode": mode,
            "source": src,
            "reference": ref,
            "prediction": pred
        })

        metrics[mode]["preds"].append(pred)
        metrics[mode]["refs"].append(ref)

# ============================================================
# METRICS
# ============================================================
def calc(preds, refs):
    return (
        sacrebleu.corpus_bleu(preds, [refs]).score,
        sacrebleu.corpus_chrf(preds, [refs]).score
    )

e2i_bleu, e2i_chrf = calc(metrics["ENG_to_INDIC"]["preds"], metrics["ENG_to_INDIC"]["refs"])
i2e_bleu, i2e_chrf = calc(metrics["INDIC_to_ENG"]["preds"], metrics["INDIC_to_ENG"]["refs"])

lid_acc = np.mean([
    is_indic(r["prediction"]) if r["mode"]=="ENG_to_INDIC"
    else not is_indic(r["prediction"])
    for r in results
])

print("\n" + "="*60)
print(f"ENG → INDIC | BLEU {e2i_bleu:.2f} | chrF {e2i_chrf:.2f}")
print(f"INDIC → ENG | BLEU {i2e_bleu:.2f} | chrF {i2e_chrf:.2f}")
print(f"LID Accuracy: {lid_acc:.2%}")
print("="*60)

# ============================================================
# EXPORT JSONL
# ============================================================
out_dir = Path("exports_jsonl")
out_dir.mkdir(exist_ok=True)

eng_indic = out_dir / "eng_to_indic_src_ref_pred.jsonl"
indic_eng = out_dir / "indic_to_eng_src_ref_pred.jsonl"

with open(eng_indic,"w",encoding="utf-8") as fe, open(indic_eng,"w",encoding="utf-8") as fi:
    for r in results:
        line = json.dumps(
            {"src":r["source"],"ref":r["reference"],"pred":r["prediction"]},
            ensure_ascii=False
        )
        if r["mode"]=="ENG_to_INDIC":
            fe.write(line+"\n")
        else:
            fi.write(line+"\n")

print("JSONL files written successfully.")
