# ======================================================
# ✅ Zero-Shot Evaluation for Gemma-3-270M-IT
# ======================================================

import argparse, json, torch
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu.metrics import CHRF, BLEU


# ------------------------------ CONFIG (STRICT PARITY)
MODEL_NAME = "google/gemma-3-270m-it"

MAX_SEQ_LEN = 1024          # unused in eval (kept for parity)
MAX_NEW_TOKENS = 512        # USED
BATCH_SIZE = 1              # unused
GRAD_ACCUM = 4              # unused
MAX_TRAIN_STEPS = 100       # unused
EVAL_BATCH_SIZE = 8         # optional future batching
FULL_DATASET = False        # SET TO TRUE FOR FULL EVAL
MAX_COLAB_SAMPLES = 10     # SET TO NONE FOR FULL EVAL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

DIRECTIONS = [("eng", "hin"), ("hin", "eng")]

# ------------------------------ PROMPT (IDENTICAL SEMANTICS)
def build_prompt(src_text, src_lang, tgt_lang):
    prompt = f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src_text}"
    return {
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

# ------------------------------ MODEL
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()
    return model, tok

# ------------------------------ GENERATION
@torch.no_grad()
def generate(model, tokenizer, prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt_text):].strip()

# ------------------------------ MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["dev","test"])
    ap.add_argument("--out_dir", default="./zeroshot_eval")

    # Default for Colab
    args = ap.parse_args([])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("🔹 Loading model...")
    model, tokenizer = load_model()

    for src_lang, tgt_lang in DIRECTIONS:
        print(f"\n🔹 Evaluating {src_lang} → {tgt_lang}")

        hyp_path = out_dir / f"hyp.{src_lang}_{tgt_lang}.jsonl"
        ref_path = out_dir / f"ref.{src_lang}_{tgt_lang}.jsonl"

        # Always load the eng_hin split and swap src/tgt later if needed
        ds = load_dataset(
            "ai4bharat/Pralekha",
            args.split,
            split="eng_hin"
        )

        if not FULL_DATASET:
            ds = ds.select(range(min(len(ds), MAX_COLAB_SAMPLES)))

        hyps, refs = [], []

        for ex in tqdm(ds):
            current_src_text = ""
            current_ref_text = ""

            if src_lang == "eng" and tgt_lang == "hin":
                current_src_text = ex["src_txt"]
                current_ref_text = ex["tgt_txt"]
            elif src_lang == "hin" and tgt_lang == "eng":
                current_src_text = ex["tgt_txt"] # Hindi from original tgt_txt
                current_ref_text = ex["src_txt"] # English from original src_txt
            else:
                # This case should not be reached with current DIRECTIONS
                pass

            prompt = build_prompt(current_src_text, src_lang, tgt_lang)
            prompt_txt = tokenizer.apply_chat_template(
                prompt["messages"],
                tokenize=False,
                add_generation_prompt=True
            )

            hyp = generate(model, tokenizer, prompt_txt)

            hyps.append(hyp)
            refs.append(current_ref_text)

        # ------------------------------ SAVE JSONL
        with hyp_path.open("w", encoding="utf-8") as f:
            for h in hyps:
                json.dump([h], f, ensure_ascii=False)
                f.write("\n")

        with ref_path.open("w", encoding="utf-8") as f:
            for r in refs:
                json.dump([r], f, ensure_ascii=False)
                f.write("\n")

        # ------------------------------ METRICS
        chrf = CHRF().corpus_score(hyps, [refs]).score
        bleu = BLEU().corpus_score(hyps, [refs]).score

        print("\n==============================")
        print(f"{src_lang} → {tgt_lang}")
        print(f"ChrF  : {chrf:.4f}")
        print(f"BLEU  : {bleu:.4f}")
        print(f"Samples : {len(hyps)}")
        print("==============================")

    print("\n✓ All outputs written to:", out_dir.resolve())

# ------------------------------
if __name__ == "__main__":
    main()
