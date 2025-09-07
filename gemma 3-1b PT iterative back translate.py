#!/usr/bin/env python
"""
Iterative Back-Translation with fine-tuned Gemma-3-1B-PT
- Uses monolingual data from Pralekha (src-only or tgt-only sides)
- Generates synthetic parallel corpus using the model
- Merges synthetic + gold pairs
- Fine-tunes further (LoRA) for N iterations
"""

import os
import torch
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer

# ------------------------------
# Config
# ------------------------------
BASE_MODEL = Path("./gemma3-1b-pt-indicdoc")  # already fine-tuned checkpoint
OUTPUT_ROOT = Path("./ibt_gemma3-1b-pt")
LANGUAGE_PAIRS = [
    ("eng", "hin"),  # example pair, extend to all
]
N_ITER = 3         # number of IBT cycles
SYN_SAMPLES = 500  # synthetic per iteration per pair
REAL_SAMPLES = 200 # gold samples for balance

# ------------------------------
# Utils: load model for inference
# ------------------------------
def load_model_for_inference(model_path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True
    )
    model.eval()
    return tokenizer, model

def build_prompt(src, tgt, text):
    return f"""<start_of_turn>user
Translate this {src} text to {tgt}:
{text}<end_of_turn>
<start_of_turn>model"""

def translate_batch(tokenizer, model, src, tgt, texts, max_new_tokens=128):
    outputs = []
    for text in texts:
        prompt = build_prompt(src, tgt, text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
        if "<start_of_turn>model" in decoded:
            decoded = decoded.split("<start_of_turn>model")[-1].strip()
        if "<end_of_turn>" in decoded:
            decoded = decoded.split("<end_of_turn>")[0].strip()
        outputs.append(decoded)
    return outputs

# ------------------------------
# Step 1: Create synthetic parallel data
# ------------------------------
def create_synthetic_parallel(pair, tokenizer, model, max_samples=SYN_SAMPLES, subset="train"):
    src, tgt = pair
    print(f"[IBT] Generating synthetic {tgt}→{src} for {pair} ...")

    # Load monolingual data (e.g., target side sentences only)
    try:
        ds = load_dataset("ai4bharat/Pralekha", subset, split=f"{src}_{tgt}")
    except Exception as e:
        print(f"  [WARN] Skipping {pair}: {e}")
        return Dataset.from_list([])

    ds = ds.select(range(min(max_samples, len(ds))))

    # Back-translate: tgt → src
    tgt_texts = ds["tgt_txt"]
    synthetic_src = translate_batch(tokenizer, model, tgt, src, tgt_texts)

    # Build synthetic parallel
    synthetic = []
    for s, t in zip(synthetic_src, tgt_texts):
        prompt = f"""<start_of_turn>user
Translate this {src} text to {tgt}:
{s}<end_of_turn>
<start_of_turn>model
{t}<end_of_turn>"""
        synthetic.append({"text": prompt})

    return Dataset.from_list(synthetic)

# ------------------------------
# Step 2: Merge gold + synthetic
# ------------------------------
def prepare_training_dataset(pair, synthetic_ds, real_samples=REAL_SAMPLES):
    src, tgt = pair
    print(f"[IBT] Preparing gold + synthetic mix for {pair}")

    try:
        gold = load_dataset("ai4bharat/Pralekha", "train", split=f"{src}_{tgt}")
        gold = gold.select(range(min(real_samples, len(gold))))
    except Exception as e:
        print(f"  [WARN] Could not load gold: {e}")
        return synthetic_ds

    formatted_gold = []
    for row in gold:
        src_text, tgt_text = row["src_txt"], row["tgt_txt"]
        prompt = f"""<start_of_turn>user
Translate this {src} text to {tgt}:
{src_text}<end_of_turn>
<start_of_turn>model
{tgt_text}<end_of_turn>"""
        formatted_gold.append({"text": prompt})

    gold_ds = Dataset.from_list(formatted_gold)
    return concatenate_datasets([gold_ds, synthetic_ds])

# ------------------------------
# Step 3: Fine-tune with LoRA on mix
# ------------------------------
def finetune_iteration(iter_id, train_ds, base_model=BASE_MODEL):
    print(f"[IBT] Fine-tuning iteration {iter_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model),
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    out_dir = OUTPUT_ROOT / f"ibt_iter{iter_id}"
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=50,
        save_strategy="epoch",
        bf16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        max_seq_length=2048,
        packing=False,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(out_dir)
    return out_dir

# ------------------------------
# Main IBT Loop
# ------------------------------
def main():
    current_model = BASE_MODEL
    tokenizer, model = load_model_for_inference(current_model)

    for i in range(1, N_ITER + 1):
        print(f"\n========== Iteration {i}/{N_ITER} ==========")
        for pair in LANGUAGE_PAIRS:
            synthetic_ds = create_synthetic_parallel(pair, tokenizer, model)
            mixed_ds = prepare_training_dataset(pair, synthetic_ds)

            # Fine-tune on combined data
            current_model = finetune_iteration(i, mixed_ds, base_model=current_model)

        # reload updated model for next loop
        tokenizer, model = load_model_for_inference(current_model)

    print("[IBT] Finished all iterations.")

if __name__ == "__main__":
    main()
