# Colab guaranteed setup for bitsandbytes + 4-bit LoRA fine-tuning
!pip uninstall -y bitsandbytes
!pip install -U bitsandbytes
!pip install -U transformers trl accelerate datasets sacrebleu

# Restart the runtime after this to ensure Colab uses the latest bitsandbytes
#import os
#os.kill(os.getpid(), 9)  # forces runtime restart
#!/usr/bin/env python
"""
Gemma-3 doc-level fine-tuning (LoRA + 4-bit quantization) on Pralekha dataset
- Handles Eng->Indic and Indic->Eng
- Doc-level translation (max_new_tokens=4096)
- Hugging Face chat template prompts
- Gradient checkpointing + sequence packing for long docs
"""

# ------------------------------
# Install required packages
# ------------------------------
!pip install -q --upgrade transformers trl sacrebleu datasets bitsandbytes accelerate

import os, sys, json
from pathlib import Path
import torch
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
# Utility Functions
# ------------------------------
def build_chat_prompt(tokenizer, src_text, tgt_text, src_lang, tgt_lang):
    """
    Returns tokenized input using Hugging Face chat-style template
    """
    messages = [
        {"role": "user", "content": f"Translate this {src_lang} text to {tgt_lang}:\n{src_text}"},
        {"role": "assistant", "content": tgt_text}
    ]
    # Using tokenizer’s chat template method
    try:
        return tokenizer.apply_chat_template(messages, padding=True, return_tensors="pt")
    except AttributeError:
        # fallback if tokenizer does not support apply_chat_template
        prompt = f"<start_of_turn>user\nTranslate this {src_lang} text to {tgt_lang}:\n{src_text}<end_of_turn>\n<start_of_turn>model\n{tgt_text}<end_of_turn>"
        return tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

def load_dataset_for_pair(pair, subset=TRAIN_SPLIT, max_samples=None, reverse=False):
    src, tgt = pair.split("_")
    ds = load_dataset("ai4bharat/Pralekha", subset, split=f"{src}_{tgt}")
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    examples = []
    for row in ds:
        src_text = row.get("src_txt") or row.get("src_text", "")
        tgt_text = row.get("tgt_txt") or row.get("tgt_text", "")
        if src_text and tgt_text:
            if reverse:
                src_text, tgt_text = tgt_text, src_text
                src, tgt = tgt, src
            examples.append({"src": src_text, "tgt": tgt_text, "src_lang": src, "tgt_lang": tgt})
    return examples

def load_all_training_data(max_samples=None):
    all_data = []
    for pair in FORWARD_PAIRS:
        # Forward: Eng -> Indic
        all_data.extend(load_dataset_for_pair(pair, TRAIN_SPLIT, max_samples, reverse=False))
        # Reverse: Indic -> Eng
        all_data.extend(load_dataset_for_pair(pair, TRAIN_SPLIT, max_samples, reverse=True))
    print(f"[INFO] Total training examples: {len(all_data)}")
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
            pair_name = f"{src}_{tgt}" if not reverse else f"{tgt}_{src}"
            try:
                ds = load_dataset("ai4bharat/Pralekha", subset, split=f"{src}_{tgt}")
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
                    src, tgt = tgt, src
                inputs = build_chat_prompt(tokenizer, src_text, "", src, tgt).to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=MAX_SEQ_LEN)
                decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
                if "<start_of_turn>model" in decoded:
                    decoded = decoded.split("<start_of_turn>model")[-1].strip()
                if "<end_of_turn>" in decoded:
                    decoded = decoded.split("<end_of_turn>")[0].strip()
                preds.append(decoded.strip())
                refs.append(tgt_text.strip())
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
# Main Training
# ------------------------------
def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

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
            device_map="auto" if torch.cuda.is_available() else None,
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
    training_data = load_all_training_data()
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

    print("[INFO] Starting fine-tuning...")
    trainer.train()

    print("[INFO] Saving model + tokenizer + LoRA adapter...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)

    evaluate_model(OUTPUT_DIR, tokenizer, subset=EVAL_SPLIT, max_samples=EVAL_SAMPLES)

if __name__ == "__main__":
    main()
