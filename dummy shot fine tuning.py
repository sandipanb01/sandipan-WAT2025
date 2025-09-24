# ======================================================
# Gemma-IT Fine-tuning for Pralekha (One-shot + Plots + Download)
# ======================================================

import os, json
from pathlib import Path
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu
import matplotlib.pyplot as plt

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

LANG_CODE_MAP = {
    "eng": "English",
    "hin": "Hindi",
    "beng": "Bengali"
}

# ------------------------------
# Build HF Chat Prompt
# ------------------------------
def build_prompt(src_text, src_lang, tgt_lang, tokenizer):
    src_lang_name = LANG_CODE_MAP.get(src_lang, src_lang)
    tgt_lang_name = LANG_CODE_MAP.get(tgt_lang, tgt_lang)
    messages = [
        {"role": "user", "content": f"Translate this {src_lang_name} document to {tgt_lang_name}:\n{src_text}\n"},
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

        ds = load_dataset("ai4bharat/Pralekha", split=actual_split, data_dir=actual_split, streaming=False)
        added = 0
        for i, row in enumerate(ds):
            if split == "dev" and i < 1000:
                continue

            row_src_lang = row.get("src_lang") or src
            row_tgt_lang = row.get("tgt_lang") or tgt
            if row_src_lang != src or row_tgt_lang != tgt:
                continue

            src_txt = row.get("src_txt") or row.get("src_text") or ""
            tgt_txt = row.get("tgt_txt") or row.get("tgt_text") or ""
            if not src_txt or not tgt_txt:
                continue

            prompt_ids = tokenizer(build_prompt(src_txt, src, tgt, tokenizer), add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(tgt_txt, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

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
            })
            added += 1
            if max_samples and added >= max_samples:
                break
    return Dataset.from_list(examples)

# ------------------------------
# Training function
# ------------------------------
def train_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, from_slow=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=compute_dtype, device_map="auto", trust_remote_code=True, attn_implementation="eager"
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=16, target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

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
        bf16=False,
        fp16=False,
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

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=training_data,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    return model, tokenizer, eval_data, trainer

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_model(model, tokenizer, eval_data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    preds, refs = [], []
    total_loss = 0

    # Disable torch.compile / Dynamo for evaluation
    torch._dynamo.reset()
    with torch.inference_mode():
        for ex in eval_data:
            input_ids = torch.tensor([ex["input_ids"]], device=device)
            attention_mask = torch.tensor([ex["attention_mask"]], device=device)
            labels = torch.tensor([ex["labels"]], device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            generated = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=64)
            preds.append(tokenizer.decode(generated[0], skip_special_tokens=True))
            refs.append(ex["tgt_txt"])

    avg_loss = total_loss / len(eval_data) if eval_data else 0
    chrf = sacrebleu.corpus_chrf(preds, [refs])
    print(f"[RESULT] Avg Eval Loss: {avg_loss:.4f}, chrF: {chrf.score:.2f}")

# ------------------------------
# Plot & download
# ------------------------------
def plot_and_download_metrics(trainer):
    logs = trainer.state.log_history
    steps = [l.get("step") for l in logs if "loss" in l]
    losses = [l.get("loss") for l in logs if "loss" in l]
    grad_norms = [l.get("grad_norm") for l in logs if "grad_norm" in l]
    lr = [l.get("learning_rate") for l in logs if "learning_rate" in l]

    plt.figure(figsize=(15,4))
    plt.subplot(1,3,1)
    plt.plot(steps, losses, label="Loss"); plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss")
    plt.subplot(1,3,2)
    plt.plot(steps[:len(grad_norms)], grad_norms, label="Grad Norm", color="orange"); plt.xlabel("Step"); plt.title("Grad Norm")
    plt.subplot(1,3,3)
    plt.plot(steps[:len(lr)], lr, label="LR", color="green"); plt.xlabel("Step"); plt.title("Learning Rate")
    plt.tight_layout()
    plt.show()

    plt.savefig("training_metrics.png")
    print("✅ Training metrics saved as 'training_metrics.png'. Download via Colab Files tab.")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    model, tokenizer, eval_data, trainer = train_model()
    evaluate_model(model, tokenizer, eval_data)
    plot_and_download_metrics(trainer)
