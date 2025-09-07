#!/usr/bin/env python
"""
Fine-tune google/gemma-3-1b-pt with LoRA (4-bit) using TRL >= 0.9
Dataset: Iterative Back-Translation (IBT) augmented Pralekha corpus
- Bidirectional for all 16 language pairs
- Document-level translations (max 4096 tokens)
- Saves model + tokenizer + adapter (HF-compatible)
"""

import os
import sys
import torch
from pathlib import Path
from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-1b-pt"
OUTPUT_DIR = Path("./gemma3-1b-pt-indicdoc-ibt")
IBT_DATASET_PATH = Path("./ibt_augmented/ibt_augmented_dataset_iter2")  # from IBT pipeline
MAX_SEQ_LENGTH = 4096

# ------------------------------
# Main fine-tuning
# ------------------------------
def main():
    print(f"[INFO] Loading IBT-augmented dataset from {IBT_DATASET_PATH}...")
    dataset = load_from_disk(str(IBT_DATASET_PATH))
    train_dataset = dataset["train"]

    print(f"[INFO] Training samples: {len(train_dataset)}")

    # ------------------------------
    # Tokenizer
    # ------------------------------
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------
    # Auto dtype selection
    # ------------------------------
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
        print(f"[INFO] Using dtype: {compute_dtype}")
    else:
        compute_dtype = torch.float32
        print("[WARN] CUDA not available. Training on CPU will be very slow.")

    # ------------------------------
    # BitsAndBytes config
    # ------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # ------------------------------
    # Base model
    # ------------------------------
    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ------------------------------
    # LoRA config
    # ------------------------------
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

    # ------------------------------
    # Training args
    # ------------------------------
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
        run_name="gemma3-1b-pt-indicdoc-ibt",
        dataloader_pin_memory=False,
    )

    # ------------------------------
    # Trainer
    # ------------------------------
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("[INFO] Starting training on IBT-augmented dataset...")
    trainer.train()

    # ------------------------------
    # Save
    # ------------------------------
    print("[INFO] Saving model + tokenizer + adapter...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)

    print(f"[INFO] Training complete! Saved → {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
