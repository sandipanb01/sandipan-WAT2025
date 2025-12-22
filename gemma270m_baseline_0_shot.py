# ======================================================
# ✅ BASELINE ZERO-SHOT EVALUATION (NO FINETUNING)
# Model: google/gemma-3-270m-it (or 4B if you change it)
# Dataset: ai4bharat/Pralekha
# Metrics: sacreBLEU + chrF
# JSONL export + top 10 examples + ZIP
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
                "reference": ref_text.strip(),
                "src_text": src_text.strip()
            }

# ---------------------- Collate Function ------------------------
def eval_collate_fn(batch, tokenizer):
    input_ids = [x["input_ids"] for x in batch]
    refs = [x["reference"] for x in batch]
    src_texts = [x["src_text"] for x in batch]

    enc = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    )

    return enc["input_ids"], enc["attention_mask"], refs, src_texts

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
def evaluate_direction(model, tokenizer, src_lang, tgt_lang, jsonl_files):
    raw_ds = load_pralekha()
    eval_ds = EvalDataset(raw_ds, tokenizer, src_lang, tgt_lang)

    loader = DataLoader(
        eval_ds,
        batch_size=EVAL_BATCH_SIZE,
        collate_fn=partial(eval_collate_fn, tokenizer=tokenizer),
        num_workers=0
    )

    preds, refs, src_texts = [], [], []
    processed = 0
    jsonl_file = OUTPUT_DIR / f"{src_lang}_{tgt_lang}_pred_refs.jsonl"

    with open(jsonl_file, "w", encoding="utf-8") as f:
        pbar = tqdm(desc=f"Evaluating {src_lang}→{tgt_lang}")
        for input_ids, attention_mask, batch_refs, batch_srcs in loader:
            batch_preds = generate_batch(model, tokenizer, input_ids, attention_mask)

            for p, r, s in zip(batch_preds, batch_refs, batch_srcs):
                f.write(json.dumps({"src": s, "prediction": p, "reference": r}, ensure_ascii=False) + "\n")

            preds.extend(batch_preds)
            refs.extend(batch_refs)
            src_texts.extend(batch_srcs)
            processed += len(batch_refs)
            pbar.update(len(batch_refs))

            if MAX_SAMPLES and processed >= MAX_SAMPLES:
                break
        pbar.close()

    jsonl_files.append(jsonl_file)

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.metrics.CHRF(word_order=0).corpus_score(preds, [refs]).score

    print(f"{src_lang}→{tgt_lang} | BLEU={bleu:.2f} | chrF={chrf:.3f}")

    # Show top 10 predictions + references
    print("\nTop 10 examples:")
    for i in range(min(10, len(preds))):
        print(f"{i+1}. SRC: {src_texts[i]}")
        print(f"   PRED: {preds[i]}")
        print(f"   REF : {refs[i]}\n")

    return bleu, chrf

# ------------------------- MAIN ----------------------------
if __name__ == "__main__":
    model, tokenizer = load_model()

    results = {}
    jsonl_files = []

    for split in DIRECTIONS:
        src, tgt = split.split("_")
        results[split] = evaluate_direction(model, tokenizer, src, tgt, jsonl_files)

    print("\n✅ BASELINE RESULTS")
    for k, (b, c) in results.items():
        print(f"{k}: BLEU={b:.2f}, chrF={c:.3f}")

    # ------------------ ZIP EXPORT -------------------------
    zip_path = OUTPUT_DIR / "pred_refs_baseline.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for f in jsonl_files:
            zipf.write(f, arcname=f.name)
    print(f"\n📦 JSONL files zipped at: {zip_path}")
