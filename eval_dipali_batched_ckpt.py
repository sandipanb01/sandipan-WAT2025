# ============================================================
# 0. IMPORTS
# ============================================================
import os
import json
import torch
import sacrebleu
import numpy as np
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# 1. GLOBAL CONFIG
# ============================================================
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

BASE_MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
EVAL_SPLIT = "test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_ROOT = Path("./gemma3_outputs/checkpoints")
OUTPUT_ROOT = Path("./checkpoint_eval_outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 8
MAX_NEW_TOKENS = 2400

# ============================================================
# 2. METRICS
# ============================================================
def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

# ============================================================
# 3. LOAD DATA (ONCE)
# ============================================================
print("📥 Loading Pralekha test split...")
raw_dataset = load_dataset(DATASET_NAME, EVAL_SPLIT, split="eng_hin")

# ============================================================
# 4. TOKENIZER (ONCE)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 5. PROMPT BUILDERS (ADVISOR STYLE)
# ============================================================
def build_prompt_eng_to_hin(example):
    prompt = (
        "Translate the following text from English to Hindi:\n"
        f"English: {example['src_txt']}\n"
        "Hindi:"
    )

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokens = tokenizer(prompt, truncation=True)

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "reference": example["tgt_txt"],
        "source": example["src_txt"],
    }

def build_prompt_hin_to_eng(example):
    prompt = (
        "Translate the following text from Hindi to English:\n"
        f"Hindi: {example['tgt_txt']}\n"
        "English:"
    )

    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokens = tokenizer(prompt, truncation=True)

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "reference": example["src_txt"],
        "source": example["tgt_txt"],
    }

# Build datasets ONCE
dataset_e2h = raw_dataset.map(build_prompt_eng_to_hin, remove_columns=raw_dataset.column_names)
dataset_h2e = raw_dataset.map(build_prompt_hin_to_eng, remove_columns=raw_dataset.column_names)

# ============================================================
# 6. DISCOVER CHECKPOINTS
# ============================================================
checkpoints = sorted(
    [p for p in CHECKPOINT_ROOT.iterdir() if p.name.startswith("checkpoint-")],
    key=lambda x: int(x.name.split("-")[-1])
)

assert checkpoints, "❌ No checkpoints found"
print(f"✅ Found {len(checkpoints)} checkpoints")

summary_rows = []

# ============================================================
# 7. MAIN CHECKPOINT LOOP
# ============================================================
for ckpt in checkpoints:
    step = int(ckpt.name.split("-")[-1])
    print(f"\n🚀 Evaluating checkpoint-{step}")

    ckpt_out = OUTPUT_ROOT / ckpt.name
    ckpt_out.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # 7.1 LOAD BASE + LoRA → MERGE
    # --------------------------------------------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(base_model, ckpt)
    model = model.merge_and_unload()
    model.eval()

    all_preds = {"ENG_to_HIN": [], "HIN_to_ENG": []}
    all_refs  = {"ENG_to_HIN": [], "HIN_to_ENG": []}
    jsonl_rows = []

    # --------------------------------------------------------
    # 7.2 BATCHED INFERENCE FUNCTION
    # --------------------------------------------------------
    def run_eval(dataset, mode):
        for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc=mode):
            batch = dataset[i:i+BATCH_SIZE]

            padded = tokenizer.pad(
                {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                },
                return_tensors="pt",
            )

            input_ids = padded["input_ids"].to(model.device)
            attention_mask = padded["attention_mask"].to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                    temperature=0.1,
                    repetition_penalty=1.1
                )

            new_tokens = outputs[:, input_ids.shape[1]:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

            for pred, ref, src in zip(decoded, batch["reference"], batch["source"]):
                all_preds[mode].append(pred)
                all_refs[mode].append(ref)
                jsonl_rows.append({
                    "mode": mode,
                    "src": src,
                    "ref": ref,
                    "pred": pred,
                })

    # --------------------------------------------------------
    # 7.3 RUN BOTH DIRECTIONS
    # --------------------------------------------------------
    run_eval(dataset_e2h, "ENG_to_HIN")
    run_eval(dataset_h2e, "HIN_to_ENG")

    # --------------------------------------------------------
    # 7.4 METRICS
    # --------------------------------------------------------
    e2h_bleu, e2h_chrf = calc_metrics(
        all_preds["ENG_to_HIN"], all_refs["ENG_to_HIN"]
    )
    h2e_bleu, h2e_chrf = calc_metrics(
        all_preds["HIN_to_ENG"], all_refs["HIN_to_ENG"]
    )

    summary_rows.append({
        "step": step,
        "ENG_to_HIN_BLEU": e2h_bleu,
        "ENG_to_HIN_chrF": e2h_chrf,
        "HIN_to_ENG_BLEU": h2e_bleu,
        "HIN_to_ENG_chrF": h2e_chrf,
    })

    # --------------------------------------------------------
    # 7.5 SAVE JSONL
    # --------------------------------------------------------
    with open(ckpt_out / "eng_to_hin.jsonl", "w", encoding="utf-8") as fe, \
         open(ckpt_out / "hin_to_eng.jsonl", "w", encoding="utf-8") as fh:

        for r in jsonl_rows:
            line = json.dumps(
                {"src": r["src"], "ref": r["ref"], "pred": r["pred"]},
                ensure_ascii=False
            )
            if r["mode"] == "ENG_to_HIN":
                fe.write(line + "\n")
            else:
                fh.write(line + "\n")

    # --------------------------------------------------------
    # 7.6 CLEANUP
    # --------------------------------------------------------
    del model, base_model
    torch.cuda.empty_cache()

# ============================================================
# 8. SAVE METRICS + PLOTS
# ============================================================
df = pd.DataFrame(summary_rows).sort_values("step")
df.to_csv(OUTPUT_ROOT / "checkpoint_metrics.csv", index=False)

plt.figure()
plt.plot(df["step"], df["ENG_to_HIN_BLEU"], label="ENG→HIN BLEU")
plt.plot(df["step"], df["HIN_to_ENG_BLEU"], label="HIN→ENG BLEU")
plt.xlabel("Training Step")
plt.ylabel("BLEU")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "bleu_vs_steps.png")
plt.close()

plt.figure()
plt.plot(df["step"], df["ENG_to_HIN_chrF"], label="ENG→HIN chrF")
plt.plot(df["step"], df["HIN_to_ENG_chrF"], label="HIN→ENG chrF")
plt.xlabel("Training Step")
plt.ylabel("chrF")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "chrf_vs_steps.png")
plt.close()

print("\n✅ ALL CHECKPOINTS EVALUATED")
print("📊 Metrics → checkpoint_metrics.csv")
print("📈 Plots saved")
