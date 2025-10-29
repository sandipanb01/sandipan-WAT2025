# ======================================================
# ✅ Universal Evaluation for any Hugging Face instruct/causal LM
# (Streaming, Fast Evaluation, Metrics, Top-10 Preview, FP16/BF16, multi-GPU aware)
# Uses Pralekha via IterableDataset for memory-efficient batched inference.
# ======================================================

import os, json, zipfile, math, warnings, time
from pathlib import Path
from itertools import islice
import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForCausalLM
import sacrebleu
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from IPython.display import display, Markdown

# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("/content/universal_output") #Path("/kaggle/working/universal_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

USE_SMALL_SUBSET = False   # 🔀 Toggle: True = fast subset (~200 samples), False = full Pralekha
MAX_SEQ_LEN = 1024
EVAL_BATCH_SIZE = 8
FORCE_FP16 = True
FORCE_BF16 = False

INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

# ------------------------------ PROMPT BUILDER
def build_prompt(src, src_lang, tgt_lang, example, tokenizer=None):
    ex_src, ex_tgt = example
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        msgs = [
            {"role":"user","content":f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{ex_src}"},
            {"role":"assistant","content":ex_tgt},
            {"role":"user","content":f"Now translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"},
            {"role":"assistant","content":""}
        ]
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    else:
        return (f"Example translation ({LANG_MAP[src_lang]} → {LANG_MAP[tgt_lang]}):\n"
                f"{ex_src} → {ex_tgt}\n\n"
                f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}")

# ------------------------------ STREAMING DATASET
def stream_examples(tokenizer, max_samples=None):
    dataset_name = "ai4bharat/Pralekha"
    config_name = "dev"
    splits = get_dataset_split_names(dataset_name, config_name)
    if not splits:
        raise RuntimeError("Could not get splits for ai4bharat/Pralekha (check datasets or network).")

    for split in splits:
        parts = split.split("_")
        if len(parts) != 2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS + ["eng"] or tl not in INDIAN_LANGS + ["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        # small one-shot example for natural prompting
        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        one_shot = ("","")
        try:
            for row in islice(ds,50):
                s,t = row.get("src_txt",""), row.get("tgt_txt","")
                if s and t and len(s.split())>5 and len(t.split())>5:
                    one_shot = (s,t)
                    break
        except Exception as e:
            print(f"⚠️ Warning finding one-shot for {split}: {e}")
            one_shot = ("","")

        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        count = 0
        for row in ds:
            if max_samples and count >= max_samples:
                break
            s,t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            for s_txt, t_txt, dirn in [(eng,indic,f"eng_{lang}"), (indic,eng,f"{lang}_eng")]:
                yield {
                    "input_text": build_prompt(s_txt, dirn.split("_")[0], dirn.split("_")[1], one_shot, tokenizer),
                    "target_text": t_txt,
                    "direction": dirn
                }
            count += 1

# ------------------------------ ITERABLE DATASET WRAPPER
class PralekhaDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=None):
        self.tok = tokenizer
        self.max_samples = max_samples
    def __iter__(self):
        for ex in stream_examples(self.tok, self.max_samples):
            s_enc = self.tok(ex["input_text"], truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=False)
            yield {"input_text": ex["input_text"], "target_text": ex["target_text"], "direction": ex["direction"]}

# ------------------------------ MODEL PREPARATION
def prepare_model_for_eval():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        print(f"⚡ Detected CUDA GPU: {gpu_name}")
        if (any(x in gpu_name for x in ["a100","h100","a6000"]) and not FORCE_FP16) or FORCE_BF16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        print("⚠️ No CUDA detected — running on CPU")
        dtype = torch.float32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_NAME} with dtype={dtype} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    print("✅ Model ready.")
    return model, tok

# ------------------------------ EVALUATION
def evaluate_model(model, tok, batch_size=EVAL_BATCH_SIZE, max_new_tokens=256):
    subset_size = 200 if USE_SMALL_SUBSET else None
    ds = PralekhaDataset(tok, max_samples=subset_size)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=1)

    preds, refs = {}, {}
    for lang in INDIAN_LANGS:
        for d in [f"eng_{lang}", f"{lang}_eng"]:
            preds[d], refs[d] = [], []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    print(f"\n🔍 Evaluating on {'SMALL SUBSET' if USE_SMALL_SUBSET else 'FULL'} Pralekha...\n")

    for batch in tqdm(dl, desc="Evaluating language pairs"):
        dirs = batch["direction"]
        texts = batch["input_text"]
        refs_list = batch["target_text"]
        enc = tok(list(texts), return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        decs = tok.batch_decode(out, skip_special_tokens=True)
        for d, p, r in zip(dirs, decs, refs_list):
            preds[d].append(p.strip())
            refs[d].append(r.strip())

    # Save predictions
    sub_zip = OUTPUT_DIR / "baseline_submission.zip"
    with zipfile.ZipFile(sub_zip, "w") as zf:
        for d in preds:
            if not preds[d]: continue
            n_chunks = math.ceil(len(preds[d]) / 1000)
            for i in range(n_chunks):
                chunk = preds[d][i*1000:(i+1)*1000]
                if not chunk: continue
                fp = OUTPUT_DIR / f"{d.replace('_','_2_')}_{i+1}.jsonl"
                with open(fp, "w", encoding="utf-8") as f:
                    for p in chunk:
                        f.write(json.dumps([p], ensure_ascii=False)+"\n")
                zf.write(fp, fp.name)
    print(f"\n✅ Baseline submission saved → {sub_zip}")

    bleu_scores, chrf_scores = {}, {}
    for d in preds:
        if not preds[d]: continue
        try:
            bleu_scores[d] = sacrebleu.corpus_bleu(preds[d], [refs[d]]).score
            chrf_scores[d] = sacrebleu.corpus_chrf(preds[d], [[r for r in refs[d]]]).score
        except Exception as e:
            print(f"⚠️ Metric error for {d}: {e}")
            bleu_scores[d] = chrf_scores[d] = 0.0

    return bleu_scores, chrf_scores

# ------------------------------ MAIN
if __name__ == "__main__":
    start = time.time()
    model, tok = prepare_model_for_eval()

    bleu, chrf = evaluate_model(model, tok)
    print(f"\n⏱️ Total time: {(time.time()-start)/60:.2f} minutes")

    data=[]
    for d in sorted(set(list(bleu.keys())+list(chrf.keys()))):
        data.append({"Direction": d, "BLEU": round(bleu.get(d,0.0),2), "chrF": round(chrf.get(d,0.0),2)})
    df = pd.DataFrame(data).sort_values("Direction").reset_index(drop=True)
    display(Markdown("## 📋 Baseline Translation Quality Metrics"))
    display(df.style.background_gradient(cmap="YlGnBu", subset=["BLEU","chrF"]))

    avg_bleu = sum(bleu.values())/len(bleu) if bleu else 0
    avg_chrf = sum(chrf.values())/len(chrf) if chrf else 0
    print(f"\n🧮 Average BLEU: {avg_bleu:.2f}, Average chrF: {avg_chrf:.2f}")

    plot_dir = OUTPUT_DIR / "baseline_metric_plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    def plot_metric(name, scores):
        if not scores: return
        langs, vals = list(scores.keys()), list(scores.values())
        plt.figure(figsize=(12,6)); plt.bar(langs, vals)
        plt.title(f"{name} per Direction"); plt.xticks(rotation=45,ha="right")
        plt.grid(axis='y',linestyle='--',alpha=0.6)
        plt.tight_layout()
        path = plot_dir / f"{name.lower()}_plot.png"
        plt.savefig(path); plt.close()
        print(f"✅ Saved {name} plot → {path}")

    plot_metric("BLEU", bleu)
    plot_metric("chrF", chrf)

    print("\n✅ Evaluation finished — outputs in:", OUTPUT_DIR)
