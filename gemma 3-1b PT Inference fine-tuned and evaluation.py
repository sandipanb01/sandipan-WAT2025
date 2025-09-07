#!/usr/bin/env python
"""
Unified Inference + Evaluation for fine-tuned Gemma-3-1B-PT
- Supports single, batch, and document-level translation
- Evaluates BLEU + chrF2 on Pralekha dev/test
"""

import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu

# Path to fine-tuned model
MODEL_PATH = Path("./gemma3-1b-pt-indicdoc")

# All language pairs (same as training)
LANGUAGE_PAIRS = [
    "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal",
    "eng_mar", "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
    "ben_eng", "hin_eng", "tam_eng", "urd_eng"
]

# ------------------------------
# Load model + tokenizer
# ------------------------------
def load_model(model_path=MODEL_PATH):
    print(f"[INFO] Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True
    )
    model.eval()
    return tokenizer, model

# ------------------------------
# Prompt formatting
# ------------------------------
def build_prompt(src_lang, tgt_lang, text, doc_level=False):
    if doc_level:
        return f"""<start_of_turn>user
Translate this {src_lang} document to {tgt_lang}:
{text}<end_of_turn>
<start_of_turn>model"""
    else:
        return f"""<start_of_turn>user
Translate this {src_lang} text to {tgt_lang}:
{text}<end_of_turn>
<start_of_turn>model"""

# ------------------------------
# Translation
# ------------------------------
def translate_text(tokenizer, model, src_lang, tgt_lang, text, doc_level=False, max_new_tokens=512):
    """Translate a single text or document"""
    prompt = build_prompt(src_lang, tgt_lang, text, doc_level)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only the translation
    if "<start_of_turn>model" in decoded:
        decoded = decoded.split("<start_of_turn>model")[-1].strip()
    if "<end_of_turn>" in decoded:
        decoded = decoded.split("<end_of_turn>")[0].strip()

    return decoded

def batch_translate(tokenizer, model, src_lang, tgt_lang, texts, doc_level=False):
    """Translate a list of texts"""
    return [translate_text(tokenizer, model, src_lang, tgt_lang, t, doc_level) for t in texts]

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_model(tokenizer, model, lang_pairs, subset="dev", max_samples=200):
    print(f"[INFO] Starting evaluation on {subset} split...")
    results = {}

    for pair in lang_pairs:
        src, tgt = pair.split("_")
        print(f"[EVAL] {src} → {tgt}")

        try:
            ds = load_dataset("ai4bharat/Pralekha", subset, split=pair)
        except Exception as e:
            print(f"  [WARN] Skipping {pair}: {e}")
            continue

        ds = ds.select(range(min(max_samples, len(ds))))
        preds, refs = [], []

        for row in ds:
            src_text = row.get("src_txt") or row.get("src_text", "")
            ref_text = row.get("tgt_txt") or row.get("tgt_text", "")
            if not src_text or not ref_text:
                continue

            pred = translate_text(tokenizer, model, src, tgt, src_text)
            preds.append(pred)
            refs.append(ref_text)

        if preds and refs:
            bleu = sacrebleu.corpus_bleu(preds, [refs])
            chrf = sacrebleu.corpus_chrf(preds, [refs])
            results[pair] = {"BLEU": bleu.score, "chrF2": chrf.score}
            print(f"  BLEU={bleu.score:.2f}, chrF2={chrf.score:.2f}")

    out_path = MODEL_PATH / f"eval_results_{subset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Saved evaluation → {out_path}")

# ------------------------------
# Main demo
# ------------------------------
def main():
    tokenizer, model = load_model()

    # Example 1: Single sentence
    src, tgt = "eng", "ben"
    text = "India is a diverse country."
    out = translate_text(tokenizer, model, src, tgt, text)
    print(f"\n[{src} → {tgt}] {text}")
    print(f"Translation: {out}")

    # Example 2: Batch
    sentences = ["The weather is nice today.", "Artificial Intelligence is the future."]
    outs = batch_translate(tokenizer, model, "eng", "hin", sentences)
    for s, o in zip(sentences, outs):
        print(f"\n[eng → hin] {s}")
        print(f"Translation: {o}")

    # Example 3: Document-level
    doc = (
        "India is a diverse country. It has many languages and cultures. "
        "The capital city is New Delhi. Cricket is a very popular sport."
    )
    doc_out = translate_text(tokenizer, model, "eng", "tam", doc, doc_level=True)
    print(f"\n[eng → tam] Document Translation:\n{doc_out}")

    # 🔹 Run evaluation
    evaluate_model(tokenizer, model, ["eng_ben", "eng_hin", "hin_eng"], subset="dev", max_samples=50)

if __name__ == "__main__":
    main()
