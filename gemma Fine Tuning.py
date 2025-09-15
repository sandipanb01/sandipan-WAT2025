#!/usr/bin/env python
"""
Gemma-3 IT fine-tuning on full Pralekha
- Eng <-> Indic, all 11 languages
- LoRA + 4-bit quantization
- Hugging Face chat template
- Assistant-only loss
- Memory-efficient mapping
"""

# ------------------------------
# Install packages
# ------------------------------
!pip install -q --upgrade transformers trl sacrebleu datasets bitsandbytes accelerate peft

import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-270m-it"  # Can scale to 1B or 12B IT
OUTPUT_DIR = Path("./gemma3-it-pralekha-full")
MAX_SEQ_LEN = 4096
GRAD_ACCUM_STEPS = 8

# Language pairs (forward + reverse)
LANG_PAIRS = [
    "eng_ben","eng_guj","eng_hin","eng_kan","eng_mal",
    "eng_mar","eng_ori","eng_pan","eng_tam","eng_tel","eng_urd"
]

# ------------------------------
# Load tokenizer & model
# ------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

compute_dtype = (
    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    else torch.float16 if torch.cuda.is_available()
    else torch.float32
)
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

# ------------------------------
# Load full Pralekha and filter by language pairs
# ------------------------------
print("[INFO] Loading full Pralekha train dataset...")
ds = load_dataset("ai4bharat/Pralekha", "train")

# Filter for our desired language pairs
def filter_pair(example):
    pair = example["src_lang"] + "_" + example["tgt_lang"]
    return pair in LANG_PAIRS or pair[::-1] in LANG_PAIRS  # forward + reverse

filtered_ds = ds.filter(filter_pair)

# ------------------------------
# Apply chat template for SFTTrainer
# ------------------------------
def map_to_chat_format(example):
    messages = [
        {"role": "user", "content": f"Translate this {example['src_lang']} text to {example['tgt_lang']}:\n{example['src_txt']}"},
        {"role": "You are a tarnslation assistant", "content": example['tgt_txt']}
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

formatted_ds = filtered_ds.map(map_to_chat_format, batched=False)

print(f"[INFO] Total training examples after filtering: {len(formatted_ds)}")

# ------------------------------
# Training arguments
# ------------------------------
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

# ------------------------------
# Initialize SFTTrainer
# ------------------------------
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_ds,
    tokenizer=tokenizer,
    max_seq_length=MAX_SEQ_LEN,
    packing=True,
    dataset_text_field="text",
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    assistant_only_loss=True
)

# ------------------------------
# Start training
# ------------------------------
print("[INFO] Starting full training...")
trainer.train()

print("[INFO] Saving model + tokenizer...")
trainer.save_model()
tokenizer.save_pretrained(OUTPUT_DIR)
