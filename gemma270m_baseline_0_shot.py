# ======================================================
# ✅ BASELINE ZERO-SHOT EVALUATION (NO FINETUNING)
# Model: google/gemma-3-270m-it (or 4B if you change it)
# Dataset: ai4bharat/Pralekha
# Metrics: sacreBLEU + chrF
# ======================================================

import os, json, zipfile, torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, IterableDataset
from functools import partial
from tqdm import tqdm
import sacrebleu

# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./baseline_eval_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_NEW_TOKENS = 512
EVAL_BATCH_SIZE = 8
MAX_SAMPLES = 100   # set None for full streaming eval

DIRECTIONS = ["eng_hin", "hin_eng"]

# ------------------------------ MODEL LOAD
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer


# ------------------------- EVAL PROMPT --------------------------
def build_eval_prompt_messages(example, src_lang, tgt_lang):
    return [
        {
            "role": "user",
            "content": f"Translate this {src_lang} text to {tgt_lang}:\n{example['src_txt']}"
        },
        {"role": "assistant", "content": ""}
    ]


# -------------------- Streaming Dataset Wrapper -----------------
class EvalDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, src_lang, tgt_lang):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __iter__(self):
        for ex in self.dataset:
            if self.src_lang == "eng":
                src_text = ex["src_txt"]
                ref_text = ex["tgt_txt"]
            else:
                src_text = ex["tgt_txt"]
                ref_text = ex["src_txt"]

            messages = build_eval_prompt_messages(
                {"src_txt": src_text}, self.src_lang, self.tgt_lang
            )

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "reference": ref_text.strip()
            }


# ---------------------- Collate Function ------------------------
def eval_collate_fn(batch, tokenizer):
    input_ids = [x["input_ids"] for x in batch]
    refs = [x["reference"] for x in batch]

    enc = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    )

    return enc["input_ids"], enc["attention_mask"], refs


# -------------------- Generation -----------------
def generate_batch(model, tokenizer, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    preds = []
    for i in range(len(outputs)):
        prompt_len = attention_mask[i].sum().item()
        gen_ids = outputs[i][prompt_len:]
        preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

    return preds


# ------------------ Dataset Loader ----------------------
def load_pralekha():
    return load_dataset(
        "ai4bharat/Pralekha",
        name="train",
        split="eng_hin",
        streaming=True
    )


# ----------------- Evaluation Function --------------------------
def evaluate_direction(model, tokenizer, src_lang, tgt_lang):
    raw_ds = load_pralekha()
    eval_ds = EvalDataset(raw_ds, tokenizer, src_lang, tgt_lang)

    loader = DataLoader(
        eval_ds,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=partial(eval_collate_fn, tokenizer=tokenizer),
        num_workers=0
    )

    preds, refs = [], []
    processed = 0

    pbar = tqdm(desc=f"Evaluating {src_lang}→{tgt_lang}")

    for input_ids, attention_mask, batch_refs in loader:
        batch_preds = generate_batch(model, tokenizer, input_ids, attention_mask)

        preds.extend(batch_preds)
        refs.extend(batch_refs)

        processed += len(batch_refs)
        pbar.update(len(batch_refs))

        if MAX_SAMPLES and processed >= MAX_SAMPLES:
            break

    pbar.close()

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.metrics.CHRF(word_order=0).corpus_score(preds, [refs]).score

    print(f"{src_lang}→{tgt_lang} | BLEU={bleu:.2f} | chrF={chrf:.3f}")
    return bleu, chrf


# ------------------------- MAIN ----------------------------
if __name__ == "__main__":
    model, tokenizer = load_model()

    results = {}
    for split in DIRECTIONS:
        src, tgt = split.split("_")
        results[split] = evaluate_direction(model, tokenizer, src, tgt)

    print("\n✅ BASELINE RESULTS")
    for k, (b, c) in results.items():
        print(f"{k}: BLEU={b:.2f}, chrF={c:.3f}")
      
# ------------------ JSONL EXPORT --------------------------------
OUTPUT_DIR = Path("./baseline_eval_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

directions = ["eng_hin", "hin_eng"]
max_samples_export = 100
batch_size = 8

jsonl_files = []

for split in directions:
    src, tgt = split.split("_")

    raw_ds = load_pralekha_split(src, tgt)
    eval_ds = EvalDataset(raw_ds, tokenizer, src, tgt)

    collate = partial(eval_collate_fn, tokenizer=tokenizer)
    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0
    )

    save_path = OUTPUT_DIR / f"{split}_pred_refs.jsonl"
    processed = 0

    with open(save_path, "w", encoding="utf-8") as f:
        for input_ids, attention_mask, refs in loader:
            preds = generate_batch(model, tokenizer, input_ids, attention_mask)

            for p, r in zip(preds, refs):
                f.write(json.dumps(
                    {"prediction": p, "reference": r},
                    ensure_ascii=False
                ) + "\n")

            processed += len(refs)
            if processed >= max_samples_export:
                break

    jsonl_files.append(save_path)
    print(f"Saved {processed} examples to {save_path}")


# ------------------ ZIP -----------------------------------------
zip_path = OUTPUT_DIR / "pred_refs_eng_hin.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for f in jsonl_files:
        zipf.write(f, arcname=f.name)

print(f"ZIP saved at: {zip_path}")

# Optional (Colab only)
# from google.colab import files
# files.download(str(zip_path))
