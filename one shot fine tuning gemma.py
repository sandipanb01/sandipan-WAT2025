# ======================================================
# Gemma-IT Fine-tuning for Pralekha (Random One-shot + Streaming + English↔Hindi + Clean Outputs + Plots + Download)
# ======================================================

import os, json, random
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

# Only English↔Hindi
LANGUAGE_PAIRS = ["eng_hin", "hin_eng"]
TRAIN_SAMPLES = 100
EVAL_SAMPLES = 10

LANG_CODE_MAP = {
    "eng": "English",
    "hin": "Hindi"
}

# ------------------------------
# Build HF Chat Prompt (Random One-shot)
# ------------------------------
def build_prompt(src_text, src_lang, tgt_lang, example_pair, tokenizer):
    example_src, example_tgt = example_pair
    src_lang_name = LANG_CODE_MAP.get(src_lang, src_lang)
    tgt_lang_name = LANG_CODE_MAP.get(tgt_lang, tgt_lang)
    
    messages = [
        {"role": "user", "content": f"Translate this {src_lang_name} text to {tgt_lang_name}:\n{example_src}"},
        {"role": "assistant", "content": f"{example_tgt}"},
        {"role": "user", "content": f"Now translate this {src_lang_name} text to {tgt_lang_name}:\n{src_text}"},
        {"role": "assistant", "content": ""}
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ------------------------------
# Dataset builder (streaming, memory-safe random one-shot, filtered for English↔Hindi)
# ------------------------------
def load_streaming_dataset(tokenizer, split="train", max_samples=100, one_shot=True):
    examples = []

    for pair in LANGUAGE_PAIRS:
        src, tgt = pair.split("_")
        actual_split = "train" if split == "dev" else split

        # Preselect one-shot example using reservoir sampling
        one_shot_example = ("", "")
        if one_shot:
            ds = load_dataset("ai4bharat/Pralekha", split=actual_split, streaming=True, data_dir=actual_split)
            count = 0
            for row in ds:
                # Filter only English↔Hindi examples
                if row.get("src_lang") not in ["eng", "hin"] or row.get("tgt_lang") not in ["eng", "hin"]:
                    continue
                src_txt = row.get("src_txt") or row.get("src_text") or ""
                tgt_txt = row.get("tgt_txt") or row.get("tgt_text") or ""
                if not src_txt or not tgt_txt:
                    continue
                if random.randint(0, count) == 0:
                    one_shot_example = (src_txt, tgt_txt)
                count += 1
                if count >= 500:  # limit to first 500 rows for speed
                    break

        # Load streaming dataset again for actual examples
        ds = load_dataset("ai4bharat/Pralekha", split=actual_split, streaming=True, data_dir=actual_split)
        added = 0
        for row in ds:
            # Filter only English↔Hindi
            if row.get("src_lang") not in ["eng", "hin"] or row.get("tgt_lang") not in ["eng", "hin"]:
                continue
            src_txt = row.get("src_txt") or row.get("src_text") or ""
            tgt_txt = row.get("tgt_txt") or row.get("tgt_text") or ""
            if not src_txt or not tgt_txt:
                continue

            if one_shot:
                prompt_ids = tokenizer(
                    build_prompt(src_txt, src, tgt, one_shot_example, tokenizer),
                    add_special_tokens=False
                )["input_ids"]
            else:
                prompt_ids = tokenizer(
                    build_prompt(src_txt, src, tgt, ("", ""), tokenizer),
                    add_special_tokens=False
                )["input_ids"]

            target_ids = tokenizer(tgt_txt, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

            if len(prompt_ids) >= MAX_SEQ_LEN:
                prompt_ids = prompt_ids[-(MAX_SEQ_LEN - 10):]
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
                "direction": pair
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

    training_data = load_streaming_dataset(tokenizer, split=TRAIN_SPLIT, max_samples=TRAIN_SAMPLES, one_shot=True)
    eval_data = load_streaming_dataset(tokenizer, split=EVAL_SPLIT, max_samples=EVAL_SAMPLES, one_shot=True)

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
# Evaluation (Clean predictions + CHRF per direction)
# ------------------------------
def evaluate_model(model, tokenizer, eval_data, save_jsonl=True, jsonl_path="eval_predictions.jsonl", max_new_tokens=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    preds, refs, dirs, srcs = [], [], [], []
    total_loss = 0

    torch._dynamo.reset()
    with torch.inference_mode():
        for ex in eval_data:
            input_ids = torch.tensor([ex["input_ids"]], device=device)
            attention_mask = torch.tensor([ex["attention_mask"]], device=device)
            labels = torch.tensor([ex["labels"]], device=device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=2.0
            )
            gen_text = tokenizer.decode(generated[0], skip_special_tokens=True)

            # Remove prompt prefix
            prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            pred_only = gen_text[len(prompt_text):].strip()
            if pred_only.lower().startswith("assistant:"):
                pred_only = pred_only[len("assistant:"):].strip()

            preds.append(pred_only)
            refs.append(ex["tgt_txt"])
            srcs.append(ex["src_txt"])
            dirs.append(ex["direction"])

    avg_loss = total_loss / len(eval_data) if eval_data else 0
    print(f"[RESULT] Avg Eval Loss: {avg_loss:.4f}")

    # CHRF per direction
    total_chrf = []
    for pair in LANGUAGE_PAIRS:
        pair_preds = [p for p, d in zip(preds, dirs) if d==pair]
        pair_refs = [[r] for r, d in zip(refs, dirs) if d==pair]
        if pair_preds:
            chrf = sacrebleu.corpus_chrf(pair_preds, pair_refs)
            total_chrf.append(chrf.score)
            print(f"[RESULT] {pair} chrF: {chrf.score:.2f}")
    if total_chrf:
        print(f"[RESULT] Avg chrF (both directions): {sum(total_chrf)/len(total_chrf):.2f}")

    # Save JSONL
    if save_jsonl:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for s, p, r, d in zip(srcs, preds, refs, dirs):
                f.write(json.dumps({"src": s, "pred": p, "ref": r, "direction": d}, ensure_ascii=False) + "\n")
        print(f"✅ Saved predictions to {jsonl_path}")

        # Optional: print JSONL for inspection
        print("\n=== JSONL Translation Output ===")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                print(line.strip())
        print("=== End of JSONL ===")

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
    evaluate_model(model, tokenizer, eval_data, max_new_tokens=512)
    plot_and_download_metrics(trainer)
# ------------------------------
# Colab: View JSONL predictions in a table
# ------------------------------
import pandas as pd

jsonl_path = "eval_predictions.jsonl"

# Load JSONL into a DataFrame
data = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

df = pd.DataFrame(data)

# Display first 20 rows nicely
df.head(20)
    
    
