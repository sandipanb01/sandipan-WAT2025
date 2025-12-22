# ======================================================
# ✅ 5-Shot Evaluation for Gemma-3-270M-IT
# ======================================================

import json, torch, tempfile
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
FEW_SHOT_K = 5              # USED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANG_LABELS = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

DIRECTIONS = [("eng","hin"), ("hin","eng")]

# ------------------------------ MODEL
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float32
    )
    model.eval()
    return model, tok

# ------------------------------ HELPERS (FROM SCRIPT A)
def word_len(txt):
    return len(txt.split())

def build_block(src, tgt, src_lbl, tgt_lbl):
    return f"{src_lbl}: {src}\n\n{tgt_lbl}: {tgt}"

def build_final_block(src, src_lbl, tgt_lbl):
    return f"{src_lbl}: {src}\n\n{tgt_lbl}:"

# ------------------------------ GENERATION
@torch.no_grad()
def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False
    )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded[len(prompt):].strip()

# ------------------------------ MAIN
def main():
    out_dir = Path("./fewshot_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model()

    for src_lang, tgt_lang in DIRECTIONS:
        print(f"\n🔹 5-SHOT EVALUATION {src_lang} → {tgt_lang}")

        # Always load the eng_hin split
        # We will swap src/tgt text within the examples based on direction
        ds_base = load_dataset(
            "ai4bharat/Pralekha",
            "test",
            split="eng_hin"
        )

        if not FULL_DATASET:
            ds_base = ds_base.select(range(min(len(ds_base), MAX_COLAB_SAMPLES)))

        src_lbl = LANG_LABELS[src_lang]
        tgt_lbl = LANG_LABELS[tgt_lang]

        # ------------------ select few-shot examples
        examples = []
        few_shot_candidates_ds = load_dataset(
            "ai4bharat/Pralekha",
            "test",
            split="eng_hin"
        )
        for ex_candidate in few_shot_candidates_ds:
            # For few-shot examples, filter based on the English side's word count
            # Assuming 'src_txt' is English in 'eng_hin' split for filtering purpose
            if 100 < word_len(ex_candidate["src_txt"]) <= 200:
                # Construct the (src, tgt) pair for the few-shot example based on current direction
                if src_lang == "eng" and tgt_lang == "hin":
                    examples.append((ex_candidate["src_txt"], ex_candidate["tgt_txt"]))
                elif src_lang == "hin" and tgt_lang == "eng":
                    examples.append((ex_candidate["tgt_txt"], ex_candidate["src_txt"]))

                if len(examples) == FEW_SHOT_K:
                    break

        if len(examples) < FEW_SHOT_K:
            print(f"[WARN] only {len(examples)} examples found for few-shot prompts")

        example_blocks = [
            build_block(s, t, src_lbl, tgt_lbl) for s, t in examples
        ]

        hyps, refs = [], []

        for ex in tqdm(ds_base): # Iterate over the (possibly subsampled) base dataset
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

            parts = (
                [f"Translate the given input document to {tgt_lbl} language. "
                 f"Generate only the translation. Do not generate any other tokens."]
                + example_blocks
                + [build_final_block(current_src_text, src_lbl, tgt_lbl)]
            )

            prompt = "\n\n".join(parts)
            hyp = generate(model, tokenizer, prompt)

            hyps.append(hyp)
            refs.append(current_ref_text)

        # ------------------ SAVE
        hyp_path = out_dir / f"hyp.{src_lang}_{tgt_lang}.5shot.jsonl"
        ref_path = out_dir / f"ref.{src_lang}_{tgt_lang}.jsonl"

        with hyp_path.open("w", encoding="utf-8") as f:
            for h in hyps:
                json.dump([h], f, ensure_ascii=False)
                f.write("\n")

        with ref_path.open("w", encoding="utf-8") as f:
            for r in refs:
                json.dump([r], f, ensure_ascii=False)
                f.write("\n")

        # ------------------ METRICS
        chrf = CHRF().corpus_score(hyps, [refs]).score
        bleu = BLEU().corpus_score(hyps, [refs]).score

        print("\n==============================")
        print(f"{src_lang} → {tgt_lang} | 5-SHOT")
        print(f"ChrF : {chrf:.4f}")
        print(f"BLEU : {bleu:.4f}")
        print(f"Samples : {len(hyps)}")
        print("==============================")

    print("\n✓ 5-shot evaluation complete.")

# ------------------------------
if __name__ == "__main__":
    main()
