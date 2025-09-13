#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Colab-friendly IndicTrans3 doc-level fine-tuning with LoRA (Seq2SeqLM)
- Works with all English ↔ Indic language pairs in Pralekha
- Doc-level packing (≤4096 tokens)
- Trainer API style using Hugging Face + PEFT
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
MODEL_NAME = "ai4bharat/IndicTrans3-beta"  # Change if needed
OUTPUT_DIR = Path("./indictrans3-lora-finetuned")
LANGUAGE_PAIRS = [
    "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal",
    "eng_mar", "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
    "ben_eng", "hin_eng", "tam_eng", "urd_eng"
]
MAX_SEQ_LEN = 4096
DOC_TOKEN_BUFFER = 200  # leave headroom when packing docs

# ------------------------------
# Utility functions
# ------------------------------
def load_jsonl(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def pack_doc_level_examples(src_texts: List[str], tgt_texts: List[str], tokenizer, max_len=MAX_SEQ_LEN):
    """Pack sentences into doc-level chunks <= max_len tokens"""
    docs = []
    src_buf, tgt_buf = [], []
    for s, t in zip(src_texts, tgt_texts):
        src_buf.append(s)
        tgt_buf.append(t)
        token_count = len(tokenizer(" ".join(src_buf)).input_ids)
        if token_count > (max_len - DOC_TOKEN_BUFFER):
            docs.append({"input_text": " ".join(src_buf), "target_text": " ".join(tgt_buf)})
            src_buf, tgt_buf = [], []
    if src_buf and tgt_buf:
        docs.append({"input_text": " ".join(src_buf), "target_text": " ".join(tgt_buf)})
    return docs

def prepare_dataset(pair: str, split="train", max_samples=None):
    """Load Pralekha dataset and pack doc-level examples"""
    src, tgt = pair.split("_")
    try:
        ds = load_dataset("ai4bharat/Pralekha", split, split=f"{src}_{tgt}")
    except Exception as e:
        print(f"[WARN] Could not load {pair}: {e}")
        return []

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    src_texts, tgt_texts = [], []
    for row in ds:
        src_text = row.get("src_txt") or row.get("src_text", "")
        tgt_text = row.get("tgt_txt") or row.get("tgt_text", "")
        if src_text and tgt_text:
            src_texts.append(src_text)
            tgt_texts.append(tgt_text)

    return pack_doc_level_examples(src_texts, tgt_texts, tokenizer)

# ------------------------------
# Training function (Colab-friendly)
# ------------------------------
def train_indictrans3(
    model_name=MODEL_NAME,
    output_dir=OUTPUT_DIR,
    language_pairs=LANGUAGE_PAIRS,
    max_seq_len=MAX_SEQ_LEN,
    few_shot=0,  # optional few-shot examples
    max_train_samples=None,
    max_eval_samples=200
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

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    all_train_data = []
    for pair in language_pairs:
        print(f"[INFO] Loading {pair}...")
        pair_data = prepare_dataset(pair, "train", max_train_samples)
        all_train_data.extend(pair_data)
        print(f"  Added {len(pair_data)} doc-level samples")

    train_dataset = Dataset.from_list(all_train_data)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, max_length=max_seq_len, padding="max_length", return_tensors="pt"
    )

    # Trainer arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        save_strategy="epoch",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        bf16=False,
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
# train_indictrans3(max_train_samples=500)  # limit samples for quick Colab test
