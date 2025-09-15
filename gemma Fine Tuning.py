#!/usr/bin/env python
"""
Gemma-3 IT full Pralekha fine-tuning
- Eng <-> Indic, all 11 languages
- Doc-level translation (max_new_tokens=4096)
- LoRA + 4-bit quantization
- Gradient checkpointing + sequence packing
- Hugging Face chat template
- Assistant-only loss
"""

# ------------------------------
# Install packages
# ------------------------------
!pip install -q --upgrade transformers trl sacrebleu datasets bitsandbytes accelerate peft

import os, sys
from pathlib import Path
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-270m-it"  # can scale to 1B or 12B IT
OUTPUT_DIR = Path("./gemma3-it-pralekha-full")
MAX_SEQ_LEN = 4096
GRAD_ACCUM_STEPS = 8
TRAIN_SPLIT = "train"

# All language pairs for forward + reverse
LANG_PAIRS = [
    "eng_ben","eng_guj","eng_hin","eng_kan","eng_mal",
    "eng_mar","eng_ori","eng_pan","eng_tam","eng_tel","eng_urd"
]

# ------------------------------
# Utility Functions
# ------------------------------
def build_chat_prompt(tokenizer, src_text, tgt_text, src_lang, tgt_lang):
    messages = [
        {"role": "user", "content": f"Translate this {src_lang} text to {tgt_lang}:\n{src_text}"},
        {"role": "assistant", "content": tgt_text}
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            return_tensors="pt",
            return_dict=True
        )
    except AttributeError:
        prompt = f"<start_of_turn>user\nTranslate this {src_lang} text to {tgt_lang}:\n{src_text}<end_of_turn>\n<start_of_turn>model\n{tgt_text}<end_of_turn>"
        return tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN, return_dict=True)

def load_dataset_for_pair(pair, subset=TRAIN_SPLIT, reverse=False):
    src, tgt = pair.split("_")
    examples = []
    try:
        ds = load_dataset("ai4bharat/Pralekha", subset, split=f"{src}_{tgt}")
        for row in ds:
            src_text = row.get("src_txt") or row.get("src_text", "")
            tgt_text = row.get("tgt_txt") or row.get("tgt_text", "")
            if src_text and tgt_text:
                if reverse:
                    src_text, tgt_text = tgt_text, src_text
                    src, tgt = tgt, src
                examples.append({"src": src_text, "tgt": tgt_text, "src_lang": src, "tgt_lang": tgt})
    except Exception as e:
        print(f"[WARN] Could not load {pair} ({subset}): {e}")
    return examples

def load_all_training_data(tokenizer):
    all_data = []
    for pair in LANG_PAIRS:
        # forward
        all_data.extend(load_dataset_for_pair(pair, TRAIN_SPLIT, reverse=False))
        # reverse
        all_data.extend(load_dataset_for_pair(pair, TRAIN_SPLIT, reverse=True))

    # Convert to chat format
    formatted_data = []
    for example in all_data:
        messages = [
            {"role": "user", "content": f"Translate this {example['src_lang']} text to {example['tgt_lang']}:\n{example['src']}"},
            {"role": "assistant", "content": example['tgt']}
        ]
        formatted_data.append({"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)})

    print(f"[INFO] Total training examples: {len(formatted_data)}")
    return formatted_data

# ------------------------------
# Main Training Flow
# ------------------------------
def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"[INFO] Using dtype: {compute_dtype}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print("[INFO] Loading base model (4-bit)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[INFO] Loading training data...")
    training_data = load_all_training_data(tokenizer)
    if len(training_data) == 0:
        print("[ERROR] No training data found. Exiting.")
        sys.exit(1)

    train_dataset = Dataset.from_list(training_data)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=2e-4,
        num_train_epochs=3,
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
        run_name="gemma3-it-pralekha",
        dataloader_pin_memory=False
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LEN,
        packing=True,  # memory efficient sequence packing
        dataset_text_field="text",
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        assistant_only_loss=True  # train only on assistant responses
    )

    print("[INFO] Starting full training...")
    trainer.train()

    print("[INFO] Saving model + tokenizer...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()
