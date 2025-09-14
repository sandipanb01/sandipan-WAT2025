#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
IndicTrans3 doc-level fine-tuning with LoRA (Trainer API)
- Works for all English ↔ Indic Pralekha pairs
"""

import os, json
from pathlib import Path
from typing import List, Dict
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "ai4bharat/IndicTrans3-beta"
OUTPUT_DIR = Path("./indictrans3-lora-finetuned")
LANGUAGE_PAIRS = [
    "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal",
    "eng_mar", "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
    "ben_eng", "hin_eng", "tam_eng", "urd_eng"
]
MAX_SEQ_LEN = 4096

# ------------------------------
# Utility functions
# ------------------------------
def build_translation_prompt(src_lang: str, tgt_lang: str, text: str) -> str:
    """Wraps input into chat-style template for IndicTrans3"""
    messages = [
        {"role": "system", "content": f"You are a helpful translation assistant."},
        {"role": "user", "content": f"Translate this {src_lang} document to {tgt_lang}:\n{text}"}
    ]
    return messages

def prepare_dataset(pair: str, split="train", max_samples=None, tokenizer=None):
    """Load Pralekha and convert to chat-style prompts"""
    src, tgt = pair.split("_")
    try:
        ds = load_dataset("ai4bharat/Pralekha", f"{src}_{tgt}", split=split)
    except Exception as e:
        print(f"[WARN] Could not load {pair}: {e}")
        return []

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    samples = []
    for row in ds:
        src_text = row.get("src_txt") or row.get("src_text", "")
        tgt_text = row.get("tgt_txt") or row.get("tgt_text", "")
        if not src_text or not tgt_text:
            continue

        # Build HF chat template prompt
        messages = build_translation_prompt(src, tgt, src_text)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)

        samples.append({
            "input_text": prompt,
            "target_text": tgt_text
        })
    return samples

# ------------------------------
# Training function
# ------------------------------
def train_indictrans3(
    model_name=MODEL_NAME,
    output_dir=OUTPUT_DIR,
    language_pairs=LANGUAGE_PAIRS,
    max_seq_len=MAX_SEQ_LEN,
    max_train_samples=None
):
    print("[INFO] Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data across pairs
    all_train_data = []
    for pair in language_pairs:
        print(f"[INFO] Loading {pair}...")
        pair_data = prepare_dataset(pair, "train", max_train_samples, tokenizer)
        all_train_data.extend(pair_data)
        print(f"  Added {len(pair_data)} samples")

    if not all_train_data:
        raise ValueError("No training data loaded! Check dataset paths.")

    train_dataset = Dataset.from_list(all_train_data)

    # Tokenization fn
    def tokenize_fn(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_seq_len,
            truncation=True,
            padding="max_length"
        )
        labels = tokenizer(
            batch["target_text"],
            max_length=max_seq_len,
            truncation=True,
            padding="max_length"
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["input_text", "target_text"])

    data_collator = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt")

    # Trainer args
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        save_strategy="epoch",
        logging_steps=50,
        fp16=False,  # IndicTrans3 (Gemma3 backend) unstable in fp16
        bf16=torch.cuda.is_bf16_supported(),
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        dataloader_pin_memory=False,
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=training_args
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    print("[INFO] ✅ Training finished!")

# ------------------------------
# Example usage in Colab
# ------------------------------
# train_indictrans3(max_train_samples=500)
