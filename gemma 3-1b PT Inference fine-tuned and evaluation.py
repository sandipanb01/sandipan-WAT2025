#!/usr/bin/env python
"""
Unified Inference + Evaluation for fine-tuned Gemma-3-1B-PT
- Supports single, batch, and document-level translation
- Evaluates BLEU + chrF2 on Pralekha dev/test
- Exports results to Excel and optionally Google Sheets
"""

import torch
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu
import pandas as pd

# ------------------------------
# Optional Google Sheets support
# ------------------------------
try:
    import gspread
    from gspread_dataframe import set_with_dataframe
    HAS_GSPREAD = True
except ImportError:
    HAS_GSPREAD = False

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
    if "<start_of_turn>model" in decoded:
        decoded = decoded.split("<start_of_turn>model")[-1].strip()
    if "<end_of_turn>" in decoded:
        decoded = decoded.split("<end_of_turn>")[0].strip()
    return decoded

def batch_translate(tokenizer, model, src_lang, tgt_lang, texts, doc_level=False):
    return [translate_text(tokenizer, model, src_lang, tgt_lang, t, doc_level) for t in texts]

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_model(tokenizer, model, lang_pairs, subset="dev", max_samples=200, export_excel=True, export_gsheet=False):
    print(f"[INFO] Starting evaluation on {subset} split...")
    results = []

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
            results.append({"pair": pair, "BLEU": bleu.score, "chrF2": chrf.score})
            print(f"  BLEU={bleu.score:.2f}, chrF2={chrf.score:.2f}")

    # Save JSON
    out_path = MODEL_PATH / f"eval_results_{subset}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved JSON → {out_path}")

    # Save Excel
    if export_excel:
        df = pd.DataFrame(results)
        excel_path = MODEL_PATH / f"eval_results_{subset}.xlsx"
        df.to_excel(excel_path, index=False, engine="openpyxl")
        print(f"[INFO] Saved Excel → {excel_path}")

    # Save Google Sheets
    if export_gsheet:
        if not HAS_GSPREAD:
            print("[WARN] gspread not installed. Skipping Google Sheets export.")
        else:
            gc = gspread.service_account(filename="service_account.json")
            sh = gc.open("Gemma3 IndicEval")  # must exist already
            worksheet = sh.worksheet(subset) if subset in [w.title for w in sh.worksheets()] else sh.add_worksheet(title=subset, rows="100", cols="20")
            df = pd.DataFrame(results)
            worksheet.clear()
            set_with_dataframe(worksheet, df)
            print("[INFO] Saved results to Google Sheets.")

# ------------------------------
# Main demo
# ------------------------------
def main():
    tokenizer, model = load_model()

    # Quick demo
    text = "India is a diverse country."
    print("\n[eng → ben] Demo Translation:")
    print(translate_text(tokenizer, model, "eng", "ben", text))

    # Run evaluation + export
    evaluate_model(tokenizer, model, ["eng_ben", "eng_hin", "hin_eng"], subset="dev", max_samples=50, export_excel=True, export_gsheet=False)

if __name__ == "__main__":
    main()
