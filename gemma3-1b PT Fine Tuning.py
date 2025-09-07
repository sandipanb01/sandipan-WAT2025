#!/usr/bin/env python
"""
Fine-tune google/gemma-3-1b-pt with LoRA (4-bit) using TRL >= 0.9
- Uses Pralekha dataset (train/dev)
- Covers all 12 forward Eng→Indic and 4 reverse Indic→Eng pairs
- Packs sentences into doc-level chunks (≤4096 tokens)
- Auto dtype (T4 → float16, A100+ → bfloat16)
- Saves full model + LoRA adapter (HF-compatible)
- Evaluates BLEU + chrF2 with sacrebleu
"""

import os
import sys
import json
import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import sacrebleu

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-1b-pt"
OUTPUT_DIR = Path("./gemma3-1b-pt-indicdoc")
LANGUAGE_PAIRS = [
    "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal",
    "eng_mar", "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
    "ben_eng", "hin_eng", "tam_eng", "urd_eng"
]
TRAIN_SPLIT = "train"
EVAL_SPLIT = "dev"
EVAL_SAMPLES = 200  # smaller for speed, increase for full eval
MAX_SEQ_LEN = 4096

# ------------------------------
# Data functions
# ------------------------------
def create_doc_level_data(src_lang, tgt_lang, tokenizer, subset=TRAIN_SPLIT, max_samples=None):
    """Pack sentences into doc-level chunks up to 4096 tokens."""
    try:
        ds = load_dataset("ai4bharat/Pralekha", subset, split=f"{src_lang}_{tgt_lang}")
    except Exception as e:
        print(f"[WARN] Could not load {src_lang}_{tgt_lang}: {e}")
        return []

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    docs, src_buf, tgt_buf = [], [], []
    token_count = 0
    for row in ds:
        src = row.get("src_txt") or row.get("src_text", "")
        tgt = row.get("tgt_txt") or row.get("tgt_text", "")
        if not src or not tgt:
            continue

        src_buf.append(src)
        tgt_buf.append(tgt)
        # Count using tokenizer
        token_count = len(tokenizer(" ".join(src_buf)).input_ids)

        if token_count > (MAX_SEQ_LEN - 200):  # leave headroom
            prompt = f"""<start_of_turn>user
Translate this {src_lang} document to {tgt_lang}:
{' '.join(src_buf)}<end_of_turn>
<start_of_turn>model
{' '.join(tgt_buf)}<end_of_turn>"""
            docs.append({"text": prompt})
            src_buf, tgt_buf, token_count = [], [], 0

    if src_buf and tgt_buf:
        prompt = f"""<start_of_turn>user
Translate this {src_lang} document to {tgt_lang}:
{' '.join(src_buf)}<end_of_turn>
<start_of_turn>model
{' '.join(tgt_buf)}<end_of_turn>"""
        docs.append({"text": prompt})

    return docs

def load_all_finetuning_data(tokenizer, subset=TRAIN_SPLIT):
    all_data = []
    for pair in LANGUAGE_PAIRS:
        src, tgt = pair.split("_")
        print(f"[INFO] Loading {src} → {tgt} ({subset}) ...")
        pair_data = create_doc_level_data(src, tgt, tokenizer, subset)
        all_data.extend(pair_data)
        print(f"  Added {len(pair_data)} doc-level samples")
    return all_data

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_model(model_path, tokenizer, lang_pairs, subset=EVAL_SPLIT, max_samples=EVAL_SAMPLES):
    print("[INFO] Starting evaluation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True
    )
    results = {}
    for pair in lang_pairs:
        src, tgt = pair.split("_")
        print(f"[EVAL] {src} → {tgt}")
        try:
            ds = load_dataset("ai4bharat/Pralekha", subset, split=pair)
        except Exception as e:
            print(f"  [WARN] Skipping {pair}: {e}")
            continue
        ds = ds.select(range(min(max_samples, len(ds))))

        preds, refs = [], []
        for row in ds:
            src_text = row.get("src_txt") or row.get("src_text", "")
            ref_text = row.get("tgt_txt") or row.get("tgt_text", "")
            if not src_text or not ref_text:
                continue

            prompt = f"""<start_of_turn>user
Translate this {src} text to {tgt}:
{src_text}<end_of_turn>
<start_of_turn>model"""
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "<start_of_turn>model" in decoded:
                decoded = decoded.split("<start_of_turn>model")[-1].strip()
            if "<end_of_turn>" in decoded:
                decoded = decoded.split("<end_of_turn>")[0].strip()

            preds.append(decoded)
            refs.append(ref_text)

        if preds and refs:
            bleu = sacrebleu.corpus_bleu(preds, [refs])
            chrf = sacrebleu.corpus_chrf(preds, [refs])
            results[pair] = {"BLEU": bleu.score, "chrF2": chrf.score}
            print(f"  BLEU={bleu.score:.2f}, chrF2={chrf.score:.2f}")

    out_path = Path(model_path) / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved evaluation → {out_path}")

# ------------------------------
# Main training flow
# ------------------------------
def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Auto dtype
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
        print(f"[INFO] Using dtype: {compute_dtype}")
    else:
        compute_dtype = torch.float32
        print("[WARN] CUDA not available. Training on CPU will be very slow.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[INFO] Preparing training data...")
    training_data = load_all_finetuning_data(tokenizer, TRAIN_SPLIT)
    if len(training_data) == 0:
        print("[ERROR] No training data found. Exiting.")
        sys.exit(1)

    train_dataset = Dataset.from_list(training_data)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="no",
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        max_grad_norm=0.3,
        report_to="none",
        run_name="gemma3-1b-pt-indicdoc",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving model + tokenizer + adapter...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)

    print("[INFO] Running evaluation...")
    evaluate_model(OUTPUT_DIR, tokenizer, LANGUAGE_PAIRS, subset=EVAL_SPLIT, max_samples=EVAL_SAMPLES)

if __name__ == "__main__":
    main()
