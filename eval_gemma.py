#----------------------------------------------------------------
# EVALUATION (Bidirectional ENG↔HIN from eng_hin split)
#----------------------------------------------------------------

import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from functools import partial
from datasets import load_dataset
from tqdm import tqdm
import sacrebleu
import json
import zipfile
from pathlib import Path

# ================================================================
# NOTE:
# We intentionally load ONLY the eng_hin split and evaluate BOTH
# eng→hin and hin→eng by swapping source/target fields.
# This treats eng_hin as a bidirectional parallel corpus and ensures
# strict data comparability across directions.
# ================================================================


# ------------------------- EVAL PROMPT --------------------------
def build_eval_prompt_messages(example, src_lang, tgt_lang):
    user_prompt = f"Translate this {src_lang} text to {tgt_lang}:\n{example['src_txt']}"
    return [
        {"role": "user", "content": user_prompt},
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
            # Direction handling (INTENTIONAL)
            if self.src_lang == "eng" and self.tgt_lang == "hin":
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


# -------------------- Generation (SAFE slicing) -----------------
def generate_batch(model, tokenizer, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    preds = []
    for i in range(len(outputs)):
        # True prompt length per example
        prompt_len = attention_mask[i].sum().item()
        gen_ids = outputs[i][prompt_len:]
        preds.append(
            tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        )

    return preds


# ------------------ Dataset Loader (FIXED) ----------------------
def load_pralekha_split(lang1, lang2):
    split = "eng_hin"
    print(f"Dataset load info: split='{split}'")
    return load_dataset(
        "ai4bharat/Pralekha",
        name="train",
        split=split,
        streaming=True
    )


# ----------------- Evaluation Function --------------------------
def evaluate_direction(model, tokenizer, src_lang, tgt_lang,
                       max_samples=200, batch_size=8):

    raw_ds = load_pralekha_split(src_lang, tgt_lang)
    eval_ds = EvalDataset(raw_ds, tokenizer, src_lang, tgt_lang)

    collate = partial(eval_collate_fn, tokenizer=tokenizer)

    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0   # REQUIRED for IterableDataset
    )

    preds, refs = [], []
    processed = 0

    if max_samples is None:
        max_samples = float("inf")

    pbar = tqdm(desc=f"Evaluating {src_lang}→{tgt_lang}")

    for input_ids, attention_mask, batch_refs in loader:
        batch_preds = generate_batch(
            model, tokenizer, input_ids, attention_mask
        )

        preds.extend(batch_preds)
        refs.extend(batch_refs)

        processed += len(batch_refs)
        pbar.update(len(batch_refs))

        if processed >= max_samples:
            break

    pbar.close()

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.metrics.CHRF(word_order=0).corpus_score(preds, [refs]).score

    print(f"{src_lang}→{tgt_lang} | BLEU={bleu:.2f} | chrF={chrf:.3f}\n")
    return bleu, chrf


# ------------------------- Main Loop ----------------------------
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES
    model, tokenizer, trainer = train_model(max_samples=max_samples)

    results = {}
    for split in DIRECTIONS:
        src, tgt = split.split("_")
        bleu, chrf = evaluate_direction(
            model,
            tokenizer,
            src,
            tgt,
            batch_size=EVAL_BATCH_SIZE,
            max_samples=max_samples
        )
        results[split] = {"BLEU": bleu, "chrF": chrf}

    print("\n✅ Final Results (ENG↔HIN):")
    for split, scores in results.items():
        print(f"{split}: BLEU={scores['BLEU']:.2f}, chrF={scores['chrF']:.3f}")


