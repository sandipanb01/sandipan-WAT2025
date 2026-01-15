# ============================================================
# GEMMA-3-4B-IT MULTILINGUAL PRALEKHA FINETUNING (ACCELERATE)
# ============================================================

import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import sacrebleu
import unicodedata
import matplotlib.pyplot as plt

set_seed(42)

# ============================================================
# 1. CONFIG
# ============================================================
MODEL_ID = "google/gemma-3-4b-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "./gemma3-4b-multilingual"

MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_SEQ_LENGTH = 2048

# (SET None FOR FULL DATA)
TRAIN_SAMPLE_LIMIT = None   # Set None for full data
EVAL_SAMPLE_LIMIT  = None   # Set None for full data

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 2. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 3. STRICT FILTERS
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
# 4. LOAD ALL LANGUAGE PAIRS
# ============================================================
configs = get_dataset_config_names(DATASET_NAME)

train_sets, test_sets = [], []

for cfg in configs:
    train_ds = load_dataset(DATASET_NAME, cfg, split="train")
    test_ds  = load_dataset(DATASET_NAME, cfg, split="test")

    train_ds = train_ds.filter(strict_filter).filter(length_filter)
    test_ds  = test_ds.filter(length_filter)

    train_ds = train_ds.add_column("lang_pair", [cfg] * len(train_ds))
    test_ds  = test_ds.add_column("lang_pair", [cfg] * len(test_ds))

    train_sets.append(train_ds)
    test_sets.append(test_ds)

train_set = concatenate_datasets(train_sets).shuffle(seed=42)
test_set  = concatenate_datasets(test_sets).shuffle(seed=99)

# APPLY TOGGLES
if TRAIN_SAMPLE_LIMIT is not None:
    train_set = train_set.select(range(min(TRAIN_SAMPLE_LIMIT, len(train_set))))

if EVAL_SAMPLE_LIMIT is not None:
    test_set = test_set.select(range(min(EVAL_SAMPLE_LIMIT, len(test_set))))

print(f"Train samples: {len(train_set)}")
print(f"Eval  samples: {len(test_set)}")

# ============================================================
# 5. PROMPT FORMATTING (BIDIRECTIONAL, ALL LANGS)
# ============================================================
def formatting_prompts_func(example):
    prompts, completions = [], []

    for lp, src, tgt in zip(
        example["lang_pair"],
        example["src_txt"],
        example["tgt_txt"]
    ):
        src_lang, tgt_lang = lp.split("_")

        prompt = (
            f"<start_of_turn>user\n"
            f"Translate from {src_lang.upper()} to {tgt_lang.upper()}:\n"
            f"{src}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        completion = f"{tgt}<end_of_turn>"

        prompts.append(prompt)
        completions.append(completion)

    return {"prompt": prompts, "completion": completions}

train_set = train_set.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=train_set.column_names
)

# ============================================================
# 6. MODEL + LoRA
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16
)

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
    bias="none"
)

# ============================================================
# 7. TRAINING (ACCELERATE)
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_set,
    peft_config=peft_config,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_seq_length=MAX_SEQ_LENGTH,
        learning_rate=5e-5,
        num_train_epochs=2,
        logging_steps=10,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        completion_only_loss=True,
        gradient_checkpointing=True,
        save_strategy="no",
        report_to="none"
    ),
)

trainer.train()

# ============================================================
# 8. MERGE & SAVE
# ============================================================
model.eval()
model = trainer.model.merge_and_unload()
model.save_pretrained(f"{OUTPUT_DIR}/final_merged")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_merged")

# ============================================================
# 9. EVALUATION
# ============================================================
results = []

for sample in tqdm(test_set):
    src_lang, tgt_lang = sample["lang_pair"].split("_")

    prompt = (
        f"<start_of_turn>user\n"
        f"Translate from {src_lang.upper()} to {tgt_lang.upper()}:\n"
        f"{sample['src_txt']}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
            output = model.generate(
                **inputs,
                #max_new_tokens=MAX_TGT_LEN,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1
         ) # max_new_tokens=MAX_TGT_LEN

    pred = tokenizer.decode(
        output[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()

    results.append({
        "lang_pair": sample["lang_pair"],
        "source": sample["src_txt"],
        "reference": sample["tgt_txt"],
        "prediction": pred
    })
# ============================================================
# 10. EXPORT JSONL (ENG→INDIC / INDIC→ENG / ALL)
# ============================================================

export_dir = Path(OUTPUT_DIR) / "exports_jsonl"
export_dir.mkdir(parents=True, exist_ok=True)

eng_to_indic_path = export_dir / "eng_to_indic_src_ref_pred.jsonl"
indic_to_eng_path = export_dir / "indic_to_eng_src_ref_pred.jsonl"
all_pairs_path    = export_dir / "all_lang_pairs_src_ref_pred.jsonl"

eng_to_indic_count = 0
indic_to_eng_count = 0

with open(eng_to_indic_path, "w", encoding="utf-8") as f_eng, \
     open(indic_to_eng_path, "w", encoding="utf-8") as f_indic, \
     open(all_pairs_path, "w", encoding="utf-8") as f_all:

    for r in results:
        src_lang, tgt_lang = r["lang_pair"].split("_")

        record = {
            "src": r["source"],
            "ref": r["reference"],
            "pred": r["prediction"],
            "lang_pair": r["lang_pair"]
        }

        # Write ALL samples
        f_all.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Direction split
        if src_lang == "eng":
            f_eng.write(json.dumps(record, ensure_ascii=False) + "\n")
            eng_to_indic_count += 1
        else:
            f_indic.write(json.dumps(record, ensure_ascii=False) + "\n")
            indic_to_eng_count += 1

print("\n" + "=" * 60)
print("JSONL EXPORT SUMMARY")
print(f"ENG → INDIC samples   : {eng_to_indic_count}")
print(f"INDIC → ENG samples   : {indic_to_eng_count}")
print(f"ALL samples exported  : {len(results)}")
print(f"Saved to directory    : {export_dir.resolve()}")
print("=" * 60)

# ============================================================
# 11. METRICS
# ============================================================
summary = []

for lp in sorted(set(r["lang_pair"] for r in results)):
    subset = [r for r in results if r["lang_pair"] == lp]

    preds = [r["prediction"] for r in subset]
    refs  = [r["reference"] for r in subset]

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    summary.append({
        "Language Pair": lp,
        "BLEU": round(bleu, 2),
        "chrF": round(chrf, 2),
        "Samples": len(subset)
    })

df_metrics = pd.DataFrame(summary)
df_metrics.to_excel(f"{OUTPUT_DIR}/final_metrics.xlsx", index=False)

# ============================================================
# 12. LOSS CURVE
# ============================================================
steps, losses = [], []

for log in trainer.state.log_history:
    if "loss" in log and "step" in log:
        steps.append(log["step"])
        losses.append(log["loss"])

plt.figure(figsize=(8, 5))
plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid()
plt.savefig(f"{OUTPUT_DIR}/training_loss_curve.jpg", dpi=300)
plt.close()
