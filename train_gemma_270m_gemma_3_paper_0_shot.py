import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "./gemma3-270m-hindi-ft"
MAX_TRAIN_SAMPLES = 50  # Set to an integer (e.g., 5000) or None for full

# ============================================================
# MODEL & TOKENIZER (Strict Float32 Alignment)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32, # Forced Float32 for 270M stability
    device_map="auto",
    attn_implementation="eager" # Eager for maximum compatibility in float32
)

# ============================================================
# PEFT & DATA PREP
# ============================================================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['src_txt'])):
        # Exact Gemma-3 turn template
        text = (
            f"<start_of_turn>user\n"
            f"Translate from ENGLISH to HINDI:\n{example['src_txt'][i]}<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"{example['tgt_txt'][i]}<end_of_turn>"
        )
        output_texts.append(text)
    return {'text': output_texts} # Return a dictionary with 'text' key

# ============================================================
# SFT CONFIGURATION
# ============================================================
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",    # SFTTrainer looks for this if not using formatting_func
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,           # High LR for small models per arXiv:2402.17193
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    fp16=False,                   # Ensure Float32
    bf16=False,                   # Ensure Float32
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    report_to="none",
    completion_only_loss=True,    # Ensure completion only loss is enabled
    packing=False                 # Ensure packing is false for completion_only_loss to work as expected
)

# ============================================================
# TRAINER EXECUTION
# ============================================================
dataset = load_dataset(DATASET_NAME, "train", split="eng_hin")

if MAX_TRAIN_SAMPLES is not None:
    dataset = dataset.shuffle(seed=42).select(range(min(len(dataset), MAX_TRAIN_SAMPLES)))

# Apply formatting function *before* passing to SFTTrainer
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=['src_txt', 'tgt_txt'] # Remove original columns
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=sft_config
)

print(f"Starting Fine-Tuning on {len(dataset)} samples in Float32...")
trainer.train()

trainer.save_model(OUTPUT_DIR)
print(f"Training complete. Adapter saved to {OUTPUT_DIR}")
