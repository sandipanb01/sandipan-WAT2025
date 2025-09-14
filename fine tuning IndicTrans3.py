#!/usr/bin/env python
"""
IndicTrans3-beta Doc-level Fine-tuning with LoRA (4-bit)
- Uses Pralekha dataset (train/dev)
- Forward and reverse translation
- Hugging Face apply_chat_template prompts
- Causal LM
- Supports 4-bit quantization
"""

import os, sys
import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "ai4bharat/IndicTrans3-beta"
OUTPUT_DIR = Path("./indictrans3-beta-finetuned")
FORWARD_PAIRS = [
    "eng_ben","eng_guj","eng_hin","eng_kan","eng_mal",
    "eng_mar","eng_ori","eng_pan","eng_tam","eng_tel","eng_urd"
]
MAX_SEQ_LEN = 4096
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 8
EPOCHS = 2
USE_4BIT = True
PRALEKHA_CONFIG = "train"  # valid config: alignable/dev/test/train/unalignable

# ------------------------------
# Helper functions
# ------------------------------
def build_chat_prompt(src_text, tgt_text, src_lang, tgt_lang):
    """HuggingFace apply_chat_template style prompt"""
    return f"""<start_of_turn>user
Translate this {src_lang} text to {tgt_lang}:
{src_text}<end_of_turn>
<start_of_turn>model
{tgt_text}<end_of_turn>"""

def load_pralekha_filtered(pair, max_samples=None):
    """Load Pralekha and filter by language pair"""
    src_lang, tgt_lang = pair.split("_")
    ds = load_dataset("ai4bharat/Pralekha", PRALEKHA_CONFIG, streaming=False)
    # Filter rows matching the source and target languages
    filtered = [r for r in ds if r["src_lang"]==src_lang and r["tgt_lang"]==tgt_lang]
    if max_samples:
        filtered = filtered[:max_samples]
    return filtered

def prepare_training_data(tokenizer, lang_pairs, max_samples=None):
    all_data = []
    for pair in lang_pairs:
        ds = load_pralekha_filtered(pair, max_samples)
        src_lang, tgt_lang = pair.split("_")
        for row in ds:
            src_text = row.get("src_txt") or row.get("src_text") or ""
            tgt_text = row.get("tgt_txt") or row.get("tgt_text") or ""
            if not src_text or not tgt_text:
                continue
            # Forward
            prompt_text = build_chat_prompt(src_text, tgt_text, src_lang, tgt_lang)
            ids = tokenizer(prompt_text, truncation=True, max_length=MAX_SEQ_LEN).input_ids
            all_data.append({"input_ids": ids, "labels": ids})
            # Reverse
            rev_prompt = build_chat_prompt(tgt_text, src_text, tgt_lang, src_lang)
            rev_ids = tokenizer(rev_prompt, truncation=True, max_length=MAX_SEQ_LEN).input_ids
            all_data.append({"input_ids": rev_ids, "labels": rev_ids})
    return Dataset.from_list(all_data)

# ------------------------------
# Main
# ------------------------------
def main():
    print("[INFO] Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    if USE_4BIT and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        quant_config = None

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_config
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[INFO] Preparing training data...")
    train_dataset = prepare_training_data(tokenizer, FORWARD_PAIRS, max_samples=500)
    print(f"[INFO] Training samples: {len(train_dataset)}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        num_train_epochs=EPOCHS,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=(compute_dtype==torch.float16),
        bf16=False,
        optim="paged_adamw_8bit" if USE_4BIT else "adamw_torch",
        max_grad_norm=0.3,
        report_to="none",
        run_name="indictrans3-beta-lora",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving model + tokenizer + LoRA adapter...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    print("[INFO] Fine-tuning complete!")

if __name__ == "__main__":
    main()
