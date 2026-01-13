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
# 1. CONFIGURATION & STRICT FILTERING (Anti-Cheating)
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "./gemma3-strict-bidirectional"

TRAIN_CONFIG = "train"
EVAL_CONFIG  = "test"
#Set to None for full data
MAX_TRAIN_SAMPLES = None
EVAL_SAMPLES = None

MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400

# ----------------------------
# Load tokenizer EARLY
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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

# ----------------------------
# TRAIN DATA
# ----------------------------
raw_dataset = load_dataset(DATASET_NAME, TRAIN_CONFIG, split="eng_hin")
filtered_dataset = raw_dataset.filter(strict_filter)
filtered_dataset = filtered_dataset.filter(length_filter)

t_limit = MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES is not None else len(filtered_dataset)
train_set = filtered_dataset.shuffle(seed=42).select(range(t_limit))

# ----------------------------
# TEST DATA
# ----------------------------
eval_dataset = load_dataset(DATASET_NAME, EVAL_CONFIG, split="eng_hin")
eval_dataset = eval_dataset.filter(length_filter)

e_limit = EVAL_SAMPLES if EVAL_SAMPLES is not None else len(eval_dataset)
test_set = eval_dataset.shuffle(seed=99).select(range(e_limit))

# ============================================================
# 2. MODEL & LoRA CONFIG
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
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
    bias="none"
)

# ============================================================
# 3. BIDIRECTIONAL FORMATTING (PROMPT/COMPLETION SPLIT)
# ============================================================
def formatting_prompts_func(example):
    prompts = []
    completions = []
    for i in range(len(example["src_txt"])):
        if i % 2 == 0:
            instr, src, tgt = "Translate to HINDI DEVANAGARI:", example["src_txt"][i], example["tgt_txt"][i]
        else:
            instr, src, tgt = "Translate to ENGLISH:", example["tgt_txt"][i], example["src_txt"][i]

        prompts.append(f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n")
        completions.append(f"{tgt}<end_of_turn>")
        
    return {"prompt": prompts, "completion": completions}

dataset = train_set.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=train_set.column_names
)

# ============================================================
# 4. TRAINING
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        #max_length=4800,
        max_length=2048,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        completion_only_loss=True,
        save_strategy="no",
        gradient_checkpointing=True,
        report_to="none"
    ),
) # max_length=MAX_SRC_LENGTH+MAX_TGT_LENGTH

trainer.train()

model = trainer.model.merge_and_unload()
model.eval()

model.save_pretrained(f"{OUTPUT_DIR}/final_merged")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_merged")

# ============================================================
# 5. STRICT EVALUATION 
# ============================================================
results = []
metrics = {"ENG_to_HIN": {"preds": [], "refs": []}, "HIN_to_ENG": {"preds": [], "refs": []}}

for sample in tqdm(test_set):
    pairs = [
        ("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", sample["src_txt"], sample["tgt_txt"]),
        ("HIN_to_ENG", "Translate to ENGLISH:", sample["tgt_txt"], sample["src_txt"]),
    ]

    for mode, instr, src, ref in pairs:
        # MATCH THE TRAINING PROMPT EXACTLY
        prompt = f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"

        # Ensure inputs are moved to GPU in the correct BFLOAT16 format
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

        # Extract only the newly generated tokens
        pred_tokens = output[0][inputs.input_ids.shape[-1]:]
        pred = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

        results.append({
            "mode": mode,
            "source": src,
            "reference": ref,
            "prediction": pred
        })

        metrics[mode]["preds"].append(pred)
        metrics[mode]["refs"].append(ref)
# ============================================================
# >>> FIX START: LID + JSON
# ============================================================
import unicodedata

def is_devanagari(text):
    for ch in text:
        if "DEVANAGARI" in unicodedata.name(ch, ""):
            return True
    return False

lid_correct = []
for r in results:
    if r["mode"] == "ENG_to_HIN":
        lid_correct.append(is_devanagari(r["prediction"]))
    else:
        lid_correct.append(not is_devanagari(r["prediction"]))

true_lid_acc = np.mean(lid_correct)

with open("final_eval_strict.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

df = pd.DataFrame(results)
# >>> FIX END
# ============================================================

def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

e2h_bleu, e2h_chrf = calc_metrics(metrics["ENG_to_HIN"]["preds"], metrics["ENG_to_HIN"]["refs"])
h2e_bleu, h2e_chrf = calc_metrics(metrics["HIN_to_ENG"]["preds"], metrics["HIN_to_ENG"]["refs"])

print("\n" + "="*50)
print("STRICT PAPER-STANDARD METRICS")
print(f"ENG → HIN | BLEU: {e2h_bleu} | chrF: {e2h_chrf}")
print(f"HIN → ENG | BLEU: {h2e_bleu} | chrF: {h2e_chrf}")
print("="*50)
print(f"Strict Script Accuracy: {true_lid_acc:.2%}")
print("="*50)

# ============================================================
# TOP-10 QUALITATIVE (SRC / REF / PRED)
# ============================================================
print("\n TOP-10 DETAILED SAMPLES\n")
for i in range(min(10, len(df))):
    r = df.iloc[i]
    print(f"[{i+1}] {r['mode']}")
    print("SRC :", r["source"])
    print("REF :", r["reference"])
    print("PRED:", r["prediction"])
    print("-" * 80)

# ============================================================
# EXPORT JSONL + ZIP
# ============================================================
out_dir = Path("exports_jsonl")
out_dir.mkdir(exist_ok=True)

eng_path = out_dir / "eng_to_hin_src_ref_pred.jsonl"
hin_path = out_dir / "hin_to_eng_src_ref_pred.jsonl"

with open(eng_path, "w", encoding="utf-8") as fe, open(hin_path, "w", encoding="utf-8") as fh:
    for r in results:
        line = json.dumps(
            {"src": r["source"], "ref": r["reference"], "pred": r["prediction"]},
            ensure_ascii=False
        )
        if r["mode"] == "ENG_to_HIN":
            fe.write(line + "\n")
        else:
            fh.write(line + "\n")

import shutil
shutil.make_archive("translation_outputs", "zip", out_dir)

print("\n JSONL + ZIP CREATED")
print("translation_outputs.zip")
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

print(f"ENG→HIN JSONL records: {eng_hin_count}")
print(f"HIN→ENG JSONL records: {hin_eng_count}")
print(f"Files saved in: {out_dir.resolve()}")

# ============================================================
# UNIVERSAL VISUAL CHECK (Works in Colab & VS Code)
# ============================================================
import pandas as pd
import sys

# 1. Prepare the data
visual_df = df[['mode', 'source', 'reference', 'prediction']].head(10)
visual_df.columns = ['Direction', 'Source Text', 'Ground Truth (Ref)', 'Model Prediction (Pred)']

# 2. Environment-Specific Display
is_colab = 'google.colab' in sys.modules

if is_colab:
    from google.colab import data_table
    data_table.enable_dataframe_formatter()
    print("COLAB MODE: Interactive Table Enabled")
    display(visual_df)
else:
    # VS Code / Terminal Mode
    print("VS CODE MODE: Printing Summary Table")
    # We use to_string() to ensure the terminal doesn't cut off the middle columns
    print(visual_df.to_string(index=False, max_colwidth=50))

# 3. Universal Detailed Text View (Best for comparing scripts/characters)
print("\n" + "═" * 80)
print("DETAILED SAMPLES (Top 10)")
print("═" * 80)

for idx, row in visual_df.iterrows():
    print(f"  SAMPLE #{idx+1} | {row['Direction']}")
    print(f"   [SRC]: {row['Source Text']}")
    print(f"   [REF]: {row['Ground Truth (Ref)']}")
    print(f"   [PRED]: {row['Model Prediction (Pred)']}")
    print("-" * 80)
    
# ============================================================
# SEPARATE CELL: FINAL METRICS EXPORT TO EXCEL
# ============================================================
import pandas as pd
import sacrebleu
import numpy as np
import unicodedata
import os

def export_final_report(results_list, output_folder):
    """
    Calculates BLEU, chrF, and Script Accuracy for both directions 
     and saves the summary to an Excel file.
    """
    # 1. Helper for script detection
    def check_devanagari(text):
        if not text: return False
        for ch in text:
            try:
                if "DEVANAGARI" in unicodedata.name(ch, ""): return True
            except ValueError: continue
        return False

    # 2. Process metrics for each direction
    summary_stats = []
    directions = ["ENG_to_HIN", "HIN_to_ENG"]

    for mode in directions:
        # Filter records for this direction
        subset = [r for r in results_list if r["mode"] == mode]
        if not subset:
            continue
            
        preds = [r["prediction"] for r in subset]
        refs = [r["reference"] for r in subset]
        
        # Calculate sacreBLEU metrics
        bleu_score = sacrebleu.corpus_bleu(preds, [refs]).score
        chrf_score = sacrebleu.corpus_chrf(preds, [refs]).score
        
        # Calculate Script (LID) Accuracy
        lid_hits = []
        for p in preds:
            has_hin = check_devanagari(p)
            # Match: English -> Hindi should have Devanagari; Hindi -> English should NOT
            is_correct_script = has_hin if mode == "ENG_to_HIN" else not has_hin
            lid_hits.append(is_correct_script)
        
        script_acc = np.mean(lid_hits) * 100 if lid_hits else 0
        
        summary_stats.append({
            "Direction": mode.replace("_", " "),
            "BLEU": round(bleu_score, 2),
            "chrF": round(chrf_score, 2),
            "Script Accuracy (%)": round(script_acc, 2),
            "Total Samples": len(subset)
        })

    # 3. Create DataFrame and Save
    report_df = pd.DataFrame(summary_stats)
    
    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    excel_path = os.path.join(output_folder, "final_translation_report.xlsx")
    
    # Save to Excel
    report_df.to_excel(excel_path, index=False)
    
    # 4. Print Summary to Console
    print("\n" + "═"*65)
    print(f"📊 REPORT GENERATED: {excel_path}")
    print("═"*65)
    print(report_df.to_string(index=False))
    print("═"*65 + "\n")

# Execute the export
# This uses 'results' and 'OUTPUT_DIR' from your previous code execution
export_final_report(results, OUTPUT_DIR)
