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
# 1. CONFIGURATION & STRICT FILTERING
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "./gemma3-strict-bidirectional"

MAX_TRAIN_SAMPLES = None 
EVAL_SAMPLES = None

def strict_filter(example):
    sim = SequenceMatcher(None, example["src_txt"].lower(), example["tgt_txt"].lower()).ratio()
    return sim < 0.65 

raw_dataset = load_dataset(DATASET_NAME, "train", split="eng_hin")
filtered_dataset = raw_dataset.filter(strict_filter)

t_limit = MAX_TRAIN_SAMPLES if MAX_TRAIN_SAMPLES is not None else len(filtered_dataset)
e_limit = EVAL_SAMPLES if EVAL_SAMPLES is not None else len(filtered_dataset)

train_set = filtered_dataset.shuffle(seed=42).select(range(t_limit))
test_set = filtered_dataset.shuffle(seed=99).select(range(e_limit))

# ============================================================
# 2. MODEL & LoRA CONFIG (SDPA Optimized)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
# Left padding is required for batched generation
tokenizer.padding_side = "left" 

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32, 
    attn_implementation="sdpa", 
    device_map="auto"
)

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules="all_linear",
    task_type="CAUSAL_LM",
    bias="none"
)

# ============================================================
# 3. BIDIRECTIONAL FORMATTING
# ============================================================
def formatting_prompts_func(example):
    texts = []
    for i in range(len(example["src_txt"])):
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
    args=SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4, 
        num_train_epochs=1,
        logging_steps=10,
        completion_loss=True,
        save_strategy="no",
        report_to="none"
    ),
)

print(f"Starting Full Dataset Training...")
trainer.train()

print("Merging LoRA adapters into base weights...")
model = trainer.model.merge_and_unload() 
model.eval()

model.save_pretrained(f"{OUTPUT_DIR}/final_merged")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_merged")

# ============================================================
# 5. OPTIMIZED BATCHED EVALUATION
# ============================================================
results = []
metrics = {"ENG_to_HIN": {"preds": [], "refs": []}, "HIN_to_ENG": {"preds": [], "refs": []}}

# Batch Size 
BATCH_SIZE = 4 
torch.cuda.empty_cache()

print(f"Starting Fast Batch Evaluation on {len(test_set)} samples...")

for i in tqdm(range(0, len(test_set), BATCH_SIZE)):
    batch = test_set.select(range(i, min(i + BATCH_SIZE, len(test_set))))
    
    for mode in ["ENG_to_HIN", "HIN_to_ENG"]:
        if mode == "ENG_to_HIN":
            instrs = ["Translate to HINDI DEVANAGARI:"] * len(batch)
            srcs, refs = batch["src_txt"], batch["tgt_txt"]
        else:
            instrs = ["Translate to ENGLISH:"] * len(batch)
            srcs, refs = batch["tgt_txt"], batch["src_txt"]

        prompts = [f"<start_of_turn>user\n{ins}\n{s}<end_of_turn>\n<start_of_turn>model\n" for ins, s in zip(instrs, srcs)]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs, 
                max_new_tokens=512,
                use_cache=True, # used for doc-level speed
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        preds = tokenizer.batch_decode(output_tokens[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        for s, r, p in zip(srcs, refs, preds):
            p_clean = p.strip()
            results.append({"mode": mode, "source": s, "reference": r, "prediction": p_clean})
            metrics[mode]["preds"].append(p_clean)
            metrics[mode]["refs"].append(r)

# ============================================================
# 6. METRICS & REPORTS
# ============================================================
import unicodedata

def is_hindi_script(text):
    for char in text:
        if 'DEVANAGARI' in unicodedata.name(char, ''): return True
    return False

def calc_metrics(preds, refs):
    if not preds: return 0, 0
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

e2h_bleu, e2h_chrf = calc_metrics(metrics["ENG_to_HIN"]["preds"], metrics["ENG_to_HIN"]["refs"])
h2e_bleu, h2e_chrf = calc_metrics(metrics["HIN_to_ENG"]["preds"], metrics["HIN_to_ENG"]["refs"])

df = pd.DataFrame(results)
script_accs = [(is_hindi_script(r['prediction']) if "ENG_to_HIN" in r['mode'] else not is_hindi_script(r['prediction'])) for _, r in df.iterrows()]
true_lid_acc = np.mean(script_accs) if script_accs else 0

df.to_csv("final_eval_strict.csv", index=False, encoding='utf-8-sig')
with open("final_eval_strict.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print("\n" + "="*50)
print(f"STRICT METRICS (SDPA Optimized)")
print(f"ENG -> HIN | BLEU: {e2h_bleu} | chrF: {e2h_chrf}")
print(f"HIN -> ENG | BLEU: {h2e_bleu} | chrF: {h2e_chrf}")
print(f"LID Script Accuracy: {true_lid_acc:.2%}")
print("="*50)

# ============================================================
# 7. TERMINAL SAMPLE PRINTOUT (Top 10)
# ============================================================
print("\n QUALITATIVE SAMPLE CHECK (First 10)")
print("-" * 80)
for idx, item in enumerate(results[:10]):
    print(f"SAMPLE #{idx + 1} | {item['mode']}")
    print(f"SRC : {item['source'][:120]}...")
    print(f"REF : {item['reference'][:120]}...")
    print(f"PRED: {item['prediction'][:120]}...")
    print("-" * 80)

# ============================================================
# CREATE CLEAN JSONL & ZIP (Colab)
# ============================================================
out_dir = Path("exports_jsonl")
out_dir.mkdir(exist_ok=True)

with open(out_dir / "eng_to_hin.jsonl", "w") as f_e, open(out_dir / "hin_to_eng.jsonl", "w") as f_h:
    for r in results:
        line = json.dumps({"src": r["source"], "ref": r["reference"], "pred": r["prediction"]}, ensure_ascii=False) + "\n"
        if r["mode"] == "ENG_to_HIN": f_e.write(line)
        else: f_h.write(line)

import shutil
try:
    from google.colab import files
    shutil.make_archive("translation_results", "zip", "exports_jsonl")
    files.download("translation_results.zip")
except:
    print(f"Results saved to {out_dir}")
