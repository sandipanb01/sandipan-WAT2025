#!/usr/bin/env python
"""
Gemma-3 doc-level fine-tuning with LoRA (4-bit) using TRL >=0.9
- Uses Pralekha dataset (train/dev)
- Supports Eng→Indic and reverse Indic→Eng by flipping data
- Doc-level translation (max_new_tokens=4096)
- Hugging Face apply_chat_template style prompts
- Auto-dtype for 4-bit quantization (float16 / bfloat16)
"""

import os, sys, json
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
import sacrebleu

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-1b-pt"
OUTPUT_DIR = Path("./gemma3-1b-pt-indicdoc")
FORWARD_PAIRS = [
    "eng_ben","eng_guj","eng_hin","eng_kan","eng_mal",
    "eng_mar","eng_ori","eng_pan","eng_tam","eng_tel","eng_urd"
]
MAX_SEQ_LEN = 4096
TRAIN_SPLIT = "train"
EVAL_SPLIT = "dev"
EVAL_SAMPLES = 200

# ------------------------------
# Utility & Builder Functions
# ------------------------------
def build_chat_prompt(src_text, tgt_text, src_lang, tgt_lang):
    return f"""<start_of_turn>user
Translate this {src_lang} text to {tgt_lang}:
{src_text}<end_of_turn>
<start_of_turn>model
{tgt_text}<end_of_turn>"""

def load_doc_level_dataset(pair, tokenizer, subset=TRAIN_SPLIT, max_samples=None, reverse=False):
    src, tgt = pair.split("_")
    try:
        ds = load_dataset("ai4bharat/Pralekha", subset, split=f"{src}_{tgt}")
    except Exception as e:
        print(f"[WARN] Could not load {pair}: {e}")
        return []

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    docs = []
    src_buf, tgt_buf = [], []
    for row in ds:
        src_text = row.get("src_txt") or row.get("src_text", "")
        tgt_text = row.get("tgt_txt") or row.get("tgt_text", "")
        if not src_text or not tgt_text:
            continue

        if reverse:
            src_text, tgt_text = tgt_text, src_text
            src, tgt = tgt, src

        src_buf.append(src_text)
        tgt_buf.append(tgt_text)
        token_count = len(tokenizer(" ".join(src_buf)).input_ids)

        if token_count > (MAX_SEQ_LEN - 200):
            docs.append({"text": build_chat_prompt(" ".join(src_buf), " ".join(tgt_buf), src, tgt)})
            src_buf, tgt_buf = [], []

    if src_buf and tgt_buf:
        docs.append({"text": build_chat_prompt(" ".join(src_buf), " ".join(tgt_buf), src, tgt)})

    return docs

def load_all_training_data(tokenizer, max_samples=None):
    all_data = []
    for pair in FORWARD_PAIRS:
        print(f"[INFO] Loading {pair} forward...")
        forward_data = load_doc_level_dataset(pair, tokenizer, TRAIN_SPLIT, max_samples)
        all_data.extend(forward_data)

        print(f"[INFO] Creating reverse {pair}...")
        reverse_data = load_doc_level_dataset(pair, tokenizer, TRAIN_SPLIT, max_samples, reverse=True)
        all_data.extend(reverse_data)

        print(f"  Added {len(forward_data)} forward + {len(reverse_data)} reverse doc-level samples")
    return all_data

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_model(model_path, tokenizer, subset=EVAL_SPLIT, max_samples=EVAL_SAMPLES):
    print("[INFO] Evaluating model...")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    results = {}
    for pair in FORWARD_PAIRS:
        for reverse in [False, True]:
            src, tgt = pair.split("_")
            if reverse:
                src, tgt = tgt, src
                pair_name = f"{tgt}_{src}"
            else:
                pair_name = pair
            try:
                ds = load_dataset("ai4bharat/Pralekha", subset, split=f"{pair}")
            except Exception as e:
                print(f"[WARN] Skipping {pair_name}: {e}")
                continue
            ds = ds.select(range(min(max_samples, len(ds))))
            preds, refs = [], []
            for row in ds:
                src_text = row.get("src_txt") or row.get("src_text", "")
                tgt_text = row.get("tgt_txt") or row.get("tgt_text", "")
                if reverse:
                    src_text, tgt_text = tgt_text, src_text
                prompt = f"<start_of_turn>user\nTranslate this {src} text to {tgt}:\n{src_text}<end_of_turn>\n<start_of_turn>model"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=MAX_SEQ_LEN)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "<start_of_turn>model" in decoded:
                    decoded = decoded.split("<start_of_turn>model")[-1].strip()
                if "<end_of_turn>" in decoded:
                    decoded = decoded.split("<end_of_turn>")[0].strip()
                preds.append(decoded)
                refs.append(tgt_text)
            if preds and refs:
                bleu = sacrebleu.corpus_bleu(preds, [refs])
                chrf = sacrebleu.corpus_chrf(preds, [refs])
                results[pair_name] = {"BLEU": bleu.score, "chrF2": chrf.score}
                print(f"  {pair_name} BLEU={bleu.score:.2f}, chrF2={chrf.score:.2f}")

    out_path = Path(model_path)/"eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved evaluation results → {out_path}")

# ------------------------------
# Main training flow
# ------------------------------
def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Auto dtype for 4-bit quantization
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
        print(f"[INFO] Using dtype: {compute_dtype}")
    else:
        compute_dtype = torch.float32
        print("[WARN] CUDA not available. Training on CPU will be very slow.")

    # BitsAndBytes 4-bit config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print("[INFO] Loading model with 4-bit quantization...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

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

    print("[INFO] Loading training data...")
    training_data = load_all_training_data(tokenizer)
    if len(training_data) == 0:
        print("[ERROR] No training data found!")
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
        fp16=(compute_dtype==torch.float16),
        bf16=(compute_dtype==torch.bfloat16),
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        report_to="none",
        run_name="gemma3-1b-pt-indicdoc",
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

    print("[INFO] Saving model + tokenizer + LoRA adapter...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)

    evaluate_model(OUTPUT_DIR, tokenizer, subset=EVAL_SPLIT, max_samples=EVAL_SAMPLES)

if __name__ == "__main__":
    main()
