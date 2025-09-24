# ======================================================
# Gemma-IT Fine-tuning for Pralekha (Zero-shot / One-shot)
# ======================================================

import os
from pathlib import Path
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu

# ------------------------------
# TorchDynamo + deterministic fixes
# ------------------------------
torch._dynamo.reset()
torch._dynamo.disable()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./gemma3-pralekha")
MAX_SEQ_LEN = 1024
TRAIN_SPLIT = "train"
EVAL_SPLIT = "dev"
LANGUAGE_PAIRS = ["eng_hin", "hin_eng"]
TRAIN_SAMPLES = 100
EVAL_SAMPLES = 10

# Toggle evaluation mode
# "zero_shot" → model sees only raw source
# "one_shot" → model sees HF chat prompt once before generating
EVAL_MODE = "one_shot"  # or "zero_shot"

# ------------------------------
# Build prompt utility (HF Chat Template)
# ------------------------------
def build_prompt(src_text, src_lang, tgt_lang, tokenizer):
    messages = [
        {"role": "user", "content": f"Translate this {src_lang} document to {tgt_lang}:\n{src_text}\n"},
        {"role": "assistant", "content": ""}
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ------------------------------
# Dataset builder
# ------------------------------
def load_streaming_dataset(tokenizer, split="train", max_samples=100):
    examples = []
    for pair in LANGUAGE_PAIRS:
        src, tgt = pair.split("_")
        actual_split = "train" if split == "dev" else split

        try:
            ds = load_dataset("ai4bharat/Pralekha", split=actual_split, data_dir=actual_split, streaming=False)
        except Exception as e:
            print(f"[WARN] cannot load split={split} for pair {pair}: {e}")
            continue

        added = 0
        for i, row in enumerate(ds):
            if split == "dev" and i < 1000:
                continue

            src_txt = row.get("src_txt") or row.get("src_text") or ""
            tgt_txt = row.get("tgt_txt") or row.get("tgt_text") or ""
            if not src_txt or not tgt_txt:
                continue

            prompt_ids = tokenizer(build_prompt(src_txt, src, tgt, tokenizer), add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(tgt_txt, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

            # Safe concatenation
            input_ids = (prompt_ids + target_ids)[:MAX_SEQ_LEN]
            attention_mask = [1] * len(input_ids)
            prompt_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            labels = labels[:MAX_SEQ_LEN]

            examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "src_txt": src_txt,
                "tgt_txt": tgt_txt,
                "src_lang": src,
                "tgt_lang": tgt
            })

            added += 1
            if max_samples and added >= max_samples:
                break
    return Dataset.from_list(examples)

# ------------------------------
# Training
# ------------------------------
def train_model():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, from_slow=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.float32
    print(f"[INFO] Using compute dtype: {compute_dtype}")

    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=compute_dtype, device_map="auto", trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=16, target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[INFO] Building datasets...")
    training_data = load_streaming_dataset(tokenizer, split=TRAIN_SPLIT, max_samples=TRAIN_SAMPLES)
    eval_data = load_streaming_dataset(tokenizer, split=EVAL_SPLIT, max_samples=EVAL_SAMPLES)

    training_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=False, fp16=False,
        optim="paged_adamw_32bit",
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        max_grad_norm=0.3,
        report_to="none",
        run_name="gemma3-pralekha",
        dataloader_pin_memory=False,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
    )

    trainer = SFTTrainer(model=model, args=training_config, train_dataset=training_data, tokenizer=tokenizer)

    print("[INFO] Starting training...")
    trainer.train()

    # Save model + tokenizer
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    return model, tokenizer, eval_data

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_model(model, tokenizer, eval_data, mode="one_shot"):
    print(f"[INFO] Starting {mode} evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds, refs = [], []

    for ex in eval_data:
        if mode == "zero_shot":
            input_ids = torch.tensor([tokenizer(ex["src_txt"], add_special_tokens=True)["input_ids"]]).to(device)
            attention_mask = torch.ones_like(input_ids)
        else:  # one_shot
            input_ids = torch.tensor([tokenizer(build_prompt(ex["src_txt"], ex["src_lang"], ex["tgt_lang"], tokenizer), add_special_tokens=False)["input_ids"]]).to(device)
            attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False
            )
        pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        preds.append(pred_text)
        refs.append(ex["tgt_txt"])

    chrf2 = sacrebleu.corpus_chrf(preds, [refs])
    print(f"[RESULT] chrF2 Score ({mode}): {chrf2.score:.2f}")

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    model, tokenizer, eval_data = train_model()
    evaluate_model(model, tokenizer, eval_data, mode=EVAL_MODE)
