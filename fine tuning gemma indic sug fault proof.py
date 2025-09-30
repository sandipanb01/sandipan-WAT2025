# ======================================================
# Gemma-IT Fine-tuning for Pralekha
# (Random One-shot + Streaming + English↔Indian + Clean Outputs + Plots + Download)
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
import pandas as pd

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./gemma3-pralekha")
MAX_SEQ_LEN = 1024
TRAIN_SPLIT = "train"
EVAL_SPLIT = "dev"

# Indian languages in Pralekha
INDIAN_LANGS = ["hin", "ben", "tam", "tel", "mal", "kan", "mar", "guj", "urd", "pan", "ori", "asm", "kok", "mai", "san", "nep"]
LANG_CODE_MAP = {
    "eng": "English",
    "hin": "Hindi",
    "ben": "Bengali",
    "tam": "Tamil",
    "tel": "Telugu",
    "mal": "Malayalam",
    "kan": "Kannada",
    "mar": "Marathi",
    "guj": "Gujarati",
    "urd": "Urdu",
    "pan": "Punjabi",
    "ori": "Odia",
    "asm": "Assamese",
    "kok": "Konkani",
    "mai": "Maithili",
    "san": "Sanskrit",
    "nep": "Nepali"
}

TRAIN_SAMPLES_PER_PAIR = 50
EVAL_SAMPLES_PER_PAIR = 10

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
# Dataset builder (all English↔Indic pairs)
# ------------------------------
def load_streaming_dataset(tokenizer, split="train", samples_per_pair=50, one_shot=True):
    examples = []
    actual_split = "train" if split == "dev" else split

    for lang in INDIAN_LANGS:
        # Preselect one-shot example
        one_shot_example = ("", "")
        if one_shot:
            ds = load_dataset("ai4bharat/Pralekha", split=actual_split, streaming=True, data_dir=actual_split)
            count = 0
            for row in ds:
                sl, tl = row.get("src_lang"), row.get("tgt_lang")
                if not sl or not tl or sl == tl:
                    continue
                if not ((sl == "eng" and tl == lang) or (sl == lang and tl == "eng")):
                    continue
                src_txt = row.get("src_txt") or row.get("src_text") or ""
                tgt_txt = row.get("tgt_txt") or row.get("tgt_text") or ""
                if not src_txt or not tgt_txt:
                    continue
                if random.randint(0, count) == 0:
                    one_shot_example = (src_txt, tgt_txt)
                count += 1
                if count >= 500:
                    break

        # Stream actual examples
        ds = load_dataset("ai4bharat/Pralekha", split=actual_split, streaming=True, data_dir=actual_split)
        added = 0
        for row in ds:
            sl, tl = row.get("src_lang"), row.get("tgt_lang")
            if not sl or not tl or sl == tl:
                continue
            if not ((sl == "eng" and tl == lang) or (sl == lang and tl == "eng")):
                continue
            src_txt = row.get("src_txt") or row.get("src_text") or ""
            tgt_txt = row.get("tgt_txt") or row.get("tgt_text") or ""
            if not src_txt or not tgt_txt:
                continue

            # Determine direction
            eng, indic = (src_txt, tgt_txt) if sl == "eng" else (tgt_txt, src_txt)

            for src, tgt, direction in [(eng, indic, f"eng_{lang}"), (indic, eng, f"{lang}_eng")]:
                prompt = build_prompt(src, direction.split("_")[0], direction.split("_")[1],
                                      one_shot_example if one_shot else ("", ""), tokenizer)
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                target_ids = tokenizer(tgt, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

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
                    "src_txt": src,
                    "tgt_txt": tgt,
                    "direction": direction
                })

            added += 1
            if samples_per_pair and added >= samples_per_pair:
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

    training_data = load_streaming_dataset(tokenizer, split=TRAIN_SPLIT, samples_per_pair=TRAIN_SAMPLES_PER_PAIR, one_shot=True)
    eval_data = load_streaming_dataset(tokenizer, split=EVAL_SPLIT, samples_per_pair=EVAL_SAMPLES_PER_PAIR, one_shot=True)

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
# Evaluation (with segfault fix)
# ------------------------------
def evaluate_model(model, tokenizer, eval_data, save_jsonl=True, jsonl_path="eval_predictions.jsonl", max_new_tokens=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    preds, refs, dirs, srcs = [], [], [], []
    total_loss = 0

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
                max_new_tokens=min(max_new_tokens, 256),  # safer cap
                do_sample=False,
                repetition_penalty=2.0
            )
            gen_text = tokenizer.decode(generated[0], skip_special_tokens=True)

            prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            pred_only = gen_text[len(prompt_text):].strip()
            if pred_only.lower().startswith("assistant:"):
                pred_only = pred_only[len("assistant:"):].strip()

            preds.append(pred_only)
            refs.append(ex["tgt_txt"])
            srcs.append(ex["src_txt"])
            dirs.append(ex["direction"])

            if device == "cuda":
                torch.cuda.empty_cache()  # flush VRAM

    avg_loss = total_loss / len(eval_data) if eval_data else 0
    print(f"[RESULT] Avg Eval Loss: {avg_loss:.4f}")

    total_chrf = []
    for d in set(dirs):
        pair_preds = [p for p, dd in zip(preds, dirs) if dd==d]
        pair_refs = [[r] for r, dd in zip(refs, dirs) if dd==d]
        if pair_preds:
            chrf = sacrebleu.corpus_chrf(pair_preds, pair_refs)
            total_chrf.append(chrf.score)
            print(f"[RESULT] {d} chrF: {chrf.score:.2f}")
    if total_chrf:
        print(f"[RESULT] Avg chrF (all English↔Indic): {sum(total_chrf)/len(total_chrf):.2f}")

    if save_jsonl:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for s, p, r, d in zip(srcs, preds, refs, dirs):
                f.write(json.dumps({"src": s, "pred": p, "ref": r, "direction": d}, ensure_ascii=False) + "\n")
        print(f"✅ Saved predictions to {jsonl_path}")

# ------------------------------
# Plot training metrics
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
    if grad_norms: plt.plot(steps[:len(grad_norms)], grad_norms, label="Grad Norm", color="orange"); plt.xlabel("Step"); plt.title("Grad Norm")
    plt.subplot(1,3,3)
    if lr: plt.plot(steps[:len(lr)], lr, label="LR", color="green"); plt.xlabel("Step"); plt.title("Learning Rate")
    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()
    print("✅ Training metrics saved as 'training_metrics.png'.")

# ------------------------------
# Lang-wise chrF aggregation
# ------------------------------
def plot_langwise_chrf(jsonl_path="eval_predictions.jsonl"):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    df = pd.DataFrame(data)

    lang_scores = {}
    for lang in INDIAN_LANGS:
        subset = df[df["direction"].isin([f"eng_{lang}", f"{lang}_eng"])]
        if len(subset) == 0:
            continue
        preds = subset["pred"].tolist()
        refs = [[r] for r in subset["ref"].tolist()]
        if preds:
            chrf = sacrebleu.corpus_chrf(preds, refs).score
            lang_scores[LANG_CODE_MAP[lang]] = chrf

    if not lang_scores:
        print("⚠️ No lang-wise scores computed (empty eval set?)")
        return

    langs = list(lang_scores.keys())
    scores = [lang_scores[l] for l in langs]
    plt.figure(figsize=(10,5))
    plt.bar(langs, scores, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("chrF Score")
    plt.title("chrF Scores per Indic Language (avg eng↔X)")
    plt.tight_layout()
    plt.savefig("langwise_chrf.png")
    plt.show()
    print("✅ Lang-wise chrF plot saved as 'langwise_chrf.png'.")
    print("Ranking (best→worst):")
    for l, s in sorted(lang_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{l}: {s:.2f}")

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    model, tokenizer, eval_data, trainer = train_model()
    evaluate_model(model, tokenizer, eval_data, max_new_tokens=512)
    plot_and_download_metrics(trainer)
    plot_langwise_chrf("eval_predictions.jsonl")
