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
set_seed(42) # Yann LeCun Suggestion: Ensure strict reproducibility

# ============================================================
# 1. CONFIGURATION & STRICT FILTERING (Anti-Cheating)
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "./gemma3-strict-bidirectional"

# MODIFICATION: Set to None for FULL DATASET
MAX_TRAIN_SAMPLES = None 
EVAL_SAMPLES = None

def strict_filter(example):
    """
    Prevents 'Copy Learning' by removing samples where source 
    and target are too lexically similar.
    """
    sim = SequenceMatcher(None, example["src_txt"].lower(), example["tgt_txt"].lower()).ratio()
    return sim < 0.65 

raw_dataset = load_dataset(DATASET_NAME, "train", split="eng_hin")
filtered_dataset = raw_dataset.filter(strict_filter)

# Logic to handle None as full dataset length
t_limit = MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES is not None else len(filtered_dataset)
e_limit = EVAL_SAMPLES if EVAL_SAMPLES is not None else len(filtered_dataset)

train_set = filtered_dataset.shuffle(seed=42).select(range(t_limit))
test_set = filtered_dataset.shuffle(seed=99).select(range(e_limit))

# ============================================================
# 2. MODEL & LoRA CONFIG (Paper-Safe Standards)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="flash_attention_2"
)
model.config.use_cache = False

# High-rank LoRA for cross-script mapping (Devanagari vs Latin)
peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
    bias="none"
)

# ============================================================
# 3. BIDIRECTIONAL FORMATTING (Gemma-3 Technical Format)
# ============================================================
def formatting_prompts_func(example):
    texts = []
    for i in range(len(example["src_txt"])):
        # Balanced Bidirectional Logic
        if i % 2 == 0:
            instr, src, tgt = "Translate to HINDI DEVANAGARI:", example["src_txt"][i], example["tgt_txt"][i]
        else:
            instr, src, tgt = "Translate to ENGLISH:", example["tgt_txt"][i], example["src_txt"][i]
        
        texts.append(
            f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n"
            f"<start_of_turn>model\n{tgt}<end_of_turn>"
        )
    return {"text": texts}

dataset = train_set.map(formatting_prompts_func, batched=True, remove_columns=train_set.column_names)

# ============================================================
# 4. TRAINING EXECUTION
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=4096,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4, 
        num_train_epochs=2,
        logging_steps=10,
        completion_only_loss=True, # Loss calculated only on model response
        save_strategy="no",
        gradient_checkpointing=True,
        report_to="none"
    ),
)

print(f"🚀 Starting Full Dataset Training ({len(train_set)} samples)...")
trainer.train()

# MODIFICATION: Correctly access PEFT model and merge
print("Merging LoRA adapters into base weights...")
model = trainer.model.merge_and_unload() 
model.eval()

# Save the finalized model
model.save_pretrained(f"{OUTPUT_DIR}/final_merged")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_merged")

# ============================================================
# 5. STRICT EVALUATION & SCORE CALCULATION
# ============================================================
results = []
metrics = {"ENG_to_HIN": {"preds": [], "refs": []}, "HIN_to_ENG": {"preds": [], "refs": []}}

print(f"📝 Evaluating {len(test_set)} samples...")

for sample in tqdm(test_set):
    pairs = [
        ("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", sample["src_txt"], sample["tgt_txt"]),
        ("HIN_to_ENG", "Translate to ENGLISH:", sample["tgt_txt"], sample["src_txt"])
    ]
    
    for mode, instr, src, ref in pairs:
        prompt = f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=4096, 
                temperature=0.1, 
                do_sample=False,
                repetition_penalty=1.1 # Yann LeCun Recommendation: Prevent loops in small models
            )
        
        pred = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        try: lang = detect(pred)
        except: lang = "unknown"
        
        results.append({
            "mode": mode, "source": src, "reference": ref, "prediction": pred, "lid": lang
        })
        metrics[mode]["preds"].append(pred)
        metrics[mode]["refs"].append(ref)

# Metrics Function
def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

e2h_bleu, e2h_chrf = calc_metrics(metrics["ENG_to_HIN"]["preds"], metrics["ENG_to_HIN"]["refs"])
h2e_bleu, h2e_chrf = calc_metrics(metrics["HIN_to_ENG"]["preds"], metrics["HIN_to_ENG"]["refs"])

# ============================================================
# 6. FINAL STRICT EVALUATION (SEMANTIC + SCRIPT-BASED LID)
# ============================================================
import unicodedata

def is_hindi_script(text):
    # Strictly checks if the characters belong to the DEVANAGARI block
    for char in text:
        if 'DEVANAGARI' in unicodedata.name(char, ''):
            return True
    return False

df = pd.DataFrame(results)

# Calculate Script-Based LID Accuracy (Strict check for Devanagari/Latin)
script_accs = []
for idx, row in df.iterrows():
    has_hindi = is_hindi_script(row['prediction'])
    is_eng_mode = "ENG_to_HIN" in row['mode']
    # Match if (Mode is E2H and has Hindi) OR (Mode is H2E and has no Hindi)
    script_accs.append(has_hindi if is_eng_mode else not has_hindi)

true_lid_acc = np.mean(script_accs)

# Save Final Reports
df.to_csv("final_eval_strict.csv", index=False, encoding='utf-8-sig')
with open("final_eval_strict.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("\n" + "="*50)
print("STRICT PAPER-STANDARD METRICS")
print("-" * 50)
print(f"English -> Hindi | BLEU: {e2h_bleu} | chrF: {e2h_chrf}")
print(f"Hindi -> English | BLEU: {h2e_bleu} | chrF: {h2e_chrf}")
print("-" * 50)
print(f"Strict Script Accuracy: {true_lid_acc:.2%}")
print("="*50)

# Final Sanity Check Printout
sample = df.iloc[0]
print(f"\n[SAMPLE TEST - {sample['mode']}]")
print(f"Source: {sample['source'][:70]}")
print(f"Pred:   {sample['prediction'][:70]}")
# ============================================================
# 📊 POST-HOC STRICT EVAL CELL (SRC / PRED / REF)
# BLEU + chrF + LID Sanity Check
# ============================================================

import json
import sacrebleu
import pandas as pd
import numpy as np
from langdetect import detect
import unicodedata
from collections import Counter

# ---------------------------
# Load saved evaluation JSON
# ---------------------------
EVAL_JSON = "final_eval_strict.json"

with open(EVAL_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Loaded {len(df)} translation records")

# ---------------------------
# Helper: Script-based Hindi check
# ---------------------------
def is_hindi_script(text):
    for char in text:
        if 'DEVANAGARI' in unicodedata.name(char, ''):
            return True
    return False

# ---------------------------
# Split by direction
# ---------------------------
e2h = df[df["mode"] == "ENG_to_HIN"]
h2e = df[df["mode"] == "HIN_to_ENG"]

# ---------------------------
# BLEU + chrF
# ---------------------------
def compute_metrics(sub_df):
    preds = sub_df["prediction"].tolist()
    refs  = sub_df["reference"].tolist()
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

e2h_bleu, e2h_chrf = compute_metrics(e2h)
h2e_bleu, h2e_chrf = compute_metrics(h2e)

# ---------------------------
# LID Accuracy
# ---------------------------
lid_results = []

for _, row in df.iterrows():
    try:
        lid = detect(row["prediction"])
    except:
        lid = "unknown"

    if row["mode"] == "ENG_to_HIN":
        lid_results.append(is_hindi_script(row["prediction"]))
    else:
        lid_results.append(not is_hindi_script(row["prediction"]))

lid_accuracy = np.mean(lid_results)

# ---------------------------
# REPORT
# ---------------------------
print("\n" + "="*60)
print("📐 STRICT POST-HOC METRICS")
print("="*60)
print(f"ENG → HIN | BLEU: {e2h_bleu} | chrF: {e2h_chrf}")
print(f"HIN → ENG | BLEU: {h2e_bleu} | chrF: {h2e_chrf}")
print("-" * 60)
print(f"LID Script Accuracy: {lid_accuracy:.2%}")
print("="*60)

# ---------------------------
# Human Sanity Samples
# ---------------------------
print("\n🧪 QUALITATIVE SANITY CHECK (5 Samples)\n")

for i in range(5):
    row = df.iloc[i]
    print(f"[{i+1}] MODE: {row['mode']}")
    print("SRC :", row["source"][:120])
    print("REF :", row["reference"][:120])
    print("PRED:", row["prediction"][:120])
    print("-"*60)
# ============================================================
# CREATE CLEAN JSONL FILES (SRC / REF / PRED ONLY)
# ============================================================

import json
from pathlib import Path

# Load your final evaluation JSON
with open("final_eval_strict.json", "r", encoding="utf-8") as f:
    records = json.load(f)

out_dir = Path("exports_jsonl")
out_dir.mkdir(exist_ok=True)

eng_hin_path = out_dir / "eng_to_hin_src_ref_pred.jsonl"
hin_eng_path = out_dir / "hin_to_eng_src_ref_pred.jsonl"

eng_hin_count = 0
hin_eng_count = 0

with open(eng_hin_path, "w", encoding="utf-8") as f_eng, \
     open(hin_eng_path, "w", encoding="utf-8") as f_hin:

    for r in records:
        line = {
            "src": r["source"],
            "ref": r["reference"],
            "pred": r["prediction"]
        }

        if r["mode"] == "ENG_to_HIN":
            f_eng.write(json.dumps(line, ensure_ascii=False) + "\n")
            eng_hin_count += 1

        elif r["mode"] == "HIN_to_ENG":
            f_hin.write(json.dumps(line, ensure_ascii=False) + "\n")
            hin_eng_count += 1

print(f"✅ ENG→HIN JSONL records: {eng_hin_count}")
print(f"✅ HIN→ENG JSONL records: {hin_eng_count}")
print(f"📂 Files saved in: {out_dir.resolve()}")
# ============================================================
# ZIP & DOWNLOAD JSONL FILES (COLAB) #OPTIONAL, MIGHT NOT WORK WITH VS CODE
# ============================================================

import shutil
from google.colab import files

zip_name = "translation_jsonl_outputs"
shutil.make_archive(
    base_name=zip_name,
    format="zip",
    root_dir="exports_jsonl"
)

files.download(f"{zip_name}.zip")
