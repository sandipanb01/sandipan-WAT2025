# ======================================================
# ✅ Universal Evaluation for any HF causal/instruct LM
# (Streaming, One-Shot, HuggingFace Chat Template)
# ======================================================

import os, math, warnings, json, zipfile
from pathlib import Path
from itertools import islice
import torch
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu
from tqdm import tqdm
from IPython.display import display, Markdown

# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-270m-it"  # replace with any HF causal/instruct LM
OUTPUT_DIR = Path("/content/universal_eval_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
EVAL_BATCH_SIZE = 8
MAX_NEW_TOKENS = 256

INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

# ------------------------------ PROMPT BUILDER (One-Shot + HF Chat Template)
def build_prompt(src, src_lang, tgt_lang, example, tokenizer):
    ex_src, ex_tgt = example
    msgs = [
        {"role":"user","content":f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{ex_src}"},
        {"role":"assistant","content":ex_tgt},
        {"role":"user","content":f"Now translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"},
        {"role":"assistant","content":""}
    ]
    return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

# ------------------------------ STREAMING DATASET
def stream_examples(tokenizer, max_samples=None):
    dataset_name = "ai4bharat/Pralekha"
    splits = get_dataset_split_names(dataset_name, "dev")

    for split in splits:
        parts = split.split("_")
        if len(parts) != 2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        ds = load_dataset(dataset_name, split=split, streaming=True, name="dev")
        one_shot = ("","")
        for row in islice(ds, 50):
            s,t = row.get("src_txt",""), row.get("tgt_txt","")
            if len(s.split())>5 and len(t.split())>5:
                one_shot = (s,t)
                break

        ds = load_dataset(dataset_name, split=split, streaming=True, name="dev")
        count = 0
        for row in ds:
            if max_samples and count >= max_samples: break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            for s_txt,t_txt,dirn in [(eng,indic,f"eng_{lang}"),(indic,eng,f"{lang}_eng")]:
                yield {
                    "input_text": build_prompt(s_txt, dirn.split("_")[0], dirn.split("_")[1], one_shot, tokenizer),
                    "target_text": t_txt,
                    "direction": dirn
                }
            count += 1

# ------------------------------ EVALUATION
def evaluate_model(model, tokenizer, max_samples_per_split=None, batch_size=EVAL_BATCH_SIZE):
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    preds, refs = {}, {}
    for lang in INDIAN_LANGS:
        for d in [f"eng_{lang}", f"{lang}_eng"]:
            preds[d], refs[d] = [], []

    splits = get_dataset_split_names("ai4bharat/Pralekha","dev")
    print("\n🔍 Starting batched evaluation...\n")

    for split in tqdm(splits, desc="Evaluating language pairs"):
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        ds = load_dataset("ai4bharat/Pralekha", split=split, streaming=True, name="dev")
        batch_prompts, batch_refs, batch_dirs, count = [], [], [], 0

        for row in ds:
            if max_samples_per_split and count >= max_samples_per_split: break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            batch_prompts += [
                build_prompt(eng,"eng",lang,("Example","Example"),tokenizer),
                build_prompt(indic,lang,"eng",("Example","Example"),tokenizer)
            ]
            batch_refs += [indic, eng]
            batch_dirs += [f"eng_{lang}", f"{lang}_eng"]
            count += 1

            if len(batch_prompts) >= batch_size:
                enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.pad_token_id)
                decs = tokenizer.batch_decode(out, skip_special_tokens=True)
                for dirn, pred, ref in zip(batch_dirs, decs, batch_refs):
                    preds[dirn].append(pred.strip())
                    refs[dirn].append(ref.strip())
                batch_prompts, batch_refs, batch_dirs = [], [], []

    # ---------------- METRICS
    bleu_scores, chrf_scores = {}, {}
    for d in preds:
        if not preds[d]: continue
        bleu_scores[d] = sacrebleu.corpus_bleu(preds[d],[refs[d]]).score
        chrf_scores[d] = sacrebleu.corpus_chrf(preds[d], [[r] for r in refs[d]]).score

    # ---------------- TOP-10 PREVIEW
    print("\n🔠 Sample Translations (Top 10 per direction):\n")
    for d in preds.keys():
        display(Markdown(f"### {d.upper()}"))
        for i in range(min(10,len(preds[d]))):
            display(Markdown(f"**Ref:** {refs[d][i]}  \n**Pred:** {preds[d][i]}"))

    return bleu_scores, chrf_scores

# ------------------------------ MAIN
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # Load baseline model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float32)

    # Evaluate full dev set (or limit with max_samples_per_split)
    bleu, chrf = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        max_samples_per_split=200,  # None = full dev set (~250k+ examples)
        batch_size=EVAL_BATCH_SIZE
    )

    # Display metrics
    import pandas as pd
    data = []
    for d in sorted(bleu.keys()):
        data.append({"Direction": d, "BLEU": round(bleu[d],2), "chrF": round(chrf[d],2)})
    df_metrics = pd.DataFrame(data).sort_values("Direction").reset_index(drop=True)
    display(df_metrics.style.background_gradient(cmap="YlGnBu", subset=["BLEU","chrF"]))

    avg_bleu = sum(bleu.values()) / len(bleu)
    avg_chrf = sum(chrf.values()) / len(chrf)
    print(f"\n🧮 Mean BLEU: {avg_bleu:.2f}, Mean chrF: {avg_chrf:.2f}")
