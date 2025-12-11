# ======================================================
# ✅ Universal Fine-tuning + Evaluation for Hugging Face instruct/causal LM
# (Streaming, LoRA, Fast Evaluation, Metrics, Top-10 Preview)
# Patched: Top-10 ASCII Table + Scores + Plots + JSONL + ZIP
# Refactored: Manual tokenization removed for training
# Safe tokenization filters applied to avoid empty prompts
# ======================================================

import os, json, zipfile, math, warnings, gc
from pathlib import Path
from itertools import islice
import torch
from datasets import load_dataset, get_dataset_split_names
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import numpy as np
from tabulate import tabulate

# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = Path("./universal_output_best")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 382
MAX_NEW_TOKENS = 256
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_TRAIN_STEPS = 50
EVAL_BATCH_SIZE = 8
FULL_DATASET = False
MAX_COLAB_SAMPLES = 50

# ------------------------------ BEAM SWITCH
BEAM_MODE = "A"  # "A" or "B"
BEAM_KWARGS = dict(num_beams=3, early_stopping=True) if BEAM_MODE=="A" else dict(num_beams=3, length_penalty=1.0)

INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

# ------------------------------ PROMPT BUILDERS
def build_prompt(src, src_lang, tgt_lang, example=None, tokenizer=None):
    ex_src, ex_tgt = ("", "")
    if example:
        ex_src, ex_tgt = example

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        msgs = []
        if ex_src and ex_tgt:
            msgs.append({"role":"user","content":f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{ex_src}"})
            msgs.append({"role":"assistant","content":ex_tgt})
        msgs.append({"role":"user","content":f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"})
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    if ex_src and ex_tgt:
        return f"Example translation ({LANG_MAP[src_lang]} → {LANG_MAP[tgt_lang]}):\n{ex_src} → {ex_tgt}\n\nTranslate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"
    else:
        return f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"

def eval_prompt(src, src_lang, tgt_lang):
    return f"Translate the following {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}\nTranslation: "

def extract_answer(full_output, prompt):
    if not full_output:
        return ""
    try:
        if prompt in full_output:
            return full_output.split(prompt, 1)[1].strip()
    except Exception:
        pass
    markers = ["Translation:", "Output:", "Answer:", "Translation -", "Translation —"]
    for m in markers:
        if m in full_output:
            return full_output.split(m, 1)[1].strip()
    return full_output.strip()

# ------------------------------ STREAMING DATASET
def stream_examples(tokenizer=None, max_samples=None):
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"
    splits = get_dataset_split_names(dataset_name, config_name)
    for split in splits:
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS + ["eng"] or tl not in INDIAN_LANGS + ["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        one_shot = ("","")
        for row in islice(ds,50):
            s = row.get("src_txt",""); t = row.get("tgt_txt","")
            if len(s.split())>5 and len(t.split())>5: one_shot=(s,t); break

        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        count=0
        for row in ds:
            if max_samples and count>=max_samples: break
            s=row.get("src_txt",""); t=row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            use_example = one_shot if (one_shot[0] and one_shot[1] and "dev" not in split) else None

            for s_txt, t_txt, dirn in [(eng,indic,f"eng_{lang}"),(indic,eng,f"{lang}_eng")]:
                prompt = build_prompt(s_txt, dirn.split("_")[0], dirn.split("_")[1], use_example, tokenizer)
                if not prompt or not prompt.strip(): continue
                yield {"input_text": prompt, "target_text": t_txt, "direction": dirn}
            count+=1

# ------------------------------ DATASET WRAPPER
class PralekhaDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=None):
        self.max_samples = max_samples
        self.tok = tokenizer
    def __iter__(self):
        for ex in stream_examples(self.tok, self.max_samples):
            enc = self.tok(ex["input_text"], truncation=True, max_length=MAX_SEQ_LEN)
            if not enc.get("input_ids"): continue
            enc["labels"] = self.tok(ex["target_text"], truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
            yield enc

# ------------------------------ MODEL PREP
from accelerate import init_empty_weights

def detect_lora_modules(model):
    modules=[]
    for n,m in model.named_modules():
        n_lower=n.lower()
        if any(x in n_lower for x in ["q_proj","k_proj","gate_proj","v_proj","o_proj","up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"]):
            modules.append(n.split(".")[-1])
    return list(set(modules))

# ------------------------------ MODEL PREP (Colab/T4 safe, no offload)
def prepare_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ⚡ Load model with specified dtype and low_cpu_mem_usage
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,  # Specify dtype here
        device_map="auto"
    )

 
    try: model.gradient_checkpointing_enable()
    except: pass

    target_modules = detect_lora_modules(model)
    if not target_modules:
        target_modules = [
            "q_proj","k_proj","gate_proj","v_proj","o_proj",
            "up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"
        ]
    print(f"⚡ LoRA target modules: {target_modules}")
    lora_cfg = LoraConfig(
        r=2, lora_alpha=4, target_modules=target_modules,
        lora_dropout=0.1, task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)
    return model, tok

# ------------------------------ TRAINING
def train_model(max_samples=None):
    model, tok = prepare_model()
    ds = PralekhaDataset(tok, max_samples=max_samples)
    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=MAX_TRAIN_STEPS,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.1,
        gradient_checkpointing=True
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    return model, tok, trainer

# ------------------------------ EVALUATION (fully safe, with top-10 previews)
def evaluate_model(model, tok, max_new_tokens=MAX_NEW_TOKENS, max_samples_per_split=None, batch_size=EVAL_BATCH_SIZE):
    import gc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    preds, refs, inputs = {}, {}, {}
    target_lang = "hin"
    for d in [f"eng_{target_lang}", f"{target_lang}_eng"]:
        preds[d], refs[d], inputs[d] = [], [], []

    splits = get_dataset_split_names("ai4bharat/Pralekha","dev")
    print("🔍 Starting batched evaluation (ENG<->HIN only)...") # Replaced logger.info

    for split in tqdm(splits):
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts
        if not ((sl=="eng" and tl==target_lang) or (sl==target_lang and tl=="eng")): continue
        lang = tl if sl=="eng" else sl
        ds = load_dataset("ai4bharat/Pralekha", split=split, streaming=True, name="dev")

        batch_prompts, batch_prompts_raw, batch_refs, batch_dirs, batch_inputs, count = [], [], [], [], [], 0
        for row in ds:
            if max_samples_per_split and count >= max_samples_per_split: break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            p_eng2hin = eval_prompt(eng, "eng", lang)
            p_hin2eng = eval_prompt(indic, lang, "eng")
            if not p_eng2hin.strip() or not p_hin2eng.strip(): continue

            batch_prompts += [p_eng2hin, p_hin2eng]
            batch_prompts_raw += [p_eng2hin, p_hin2eng]
            batch_refs += [indic.strip(), eng.strip()]
            batch_dirs += [f"eng_{lang}", f"{lang}_eng"]
            batch_inputs += [eng.strip(), indic.strip()]
            count += 1

            # --- Process full batch
            if len(batch_prompts) >= batch_size:
                process_batch(model, tok, batch_prompts, batch_refs, batch_dirs, batch_inputs, preds, refs, inputs, device)
                batch_prompts, batch_prompts_raw, batch_refs, batch_dirs, batch_inputs = [], [], [], [], []

        # --- Process any remaining last batch
        if batch_prompts:
            process_batch(model, tok, batch_prompts, batch_refs, batch_dirs, batch_inputs, preds, refs, inputs, device)

    # --- JSONL + ZIP outputs
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    for d in preds:
        out_file = OUTPUT_DIR / f"{d}_pred_ref.jsonl"
        with open(out_file,"w",encoding="utf-8") as f:
            for inp, p, r in zip(inputs[d], preds[d], refs[d]):
                f.write(json.dumps({"input_text": inp, "pred": p, "ref": r}, ensure_ascii=False)+"\n")
        print(f"✅ JSONL saved: {out_file}") # Replaced logger.info

    sub_zip = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(sub_zip,"w") as zf:
        for d in preds:
            n_chunks = math.ceil(len(preds[d])/1000)
            for i in range(n_chunks):
                chunk = preds[d][i*1000:(i+1)*1000]
                if not chunk: continue
                fp = OUTPUT_DIR / f"{d.replace('_','_2_')}_{i+1}.jsonl"
                with open(fp,"w",encoding="utf-8") as f:
                    for p in chunk: f.write(json.dumps([p],ensure_ascii=False)+"\n")
                zf.write(fp, fp.name)
    print(f"✅ Submission ZIP saved: {sub_zip}") # Replaced logger.info

    # --- Top-10 preview tables
    for d in preds:
        top_n = min(10, len(preds[d]))
        preview = [{"Input": inputs[d][i], "Prediction": preds[d][i], "Reference": refs[d][i]} for i in range(top_n)]
        print(f"\n🔹 Top-10 preview for {d}:\n")
        print(tabulate(preview, headers="keys", tablefmt="grid"))
        # Save TXT
        with open(OUTPUT_DIR / f"{d}_top10_preview.txt","w",encoding="utf-8") as f:
            f.write(tabulate(preview, headers="keys", tablefmt="grid"))
        # Save JSON
        with open(OUTPUT_DIR / f"{d}_top10_preview.json","w",encoding="utf-8") as f:
            json.dump(preview, f, ensure_ascii=False, indent=2)

    # --- compute metrics robustly
    bleu_scores, chrf_scores, comet_scores = {}, {}, {}
    for d in preds:
        if not preds[d]: continue
        refs_list = refs[d]
        try: bleu_scores[d] = sacrebleu.corpus_bleu(preds[d], [refs_list]).score
        except: bleu_scores[d] = 0.0
        try: chrf_scores[d] = sacrebleu.corpus_chrf(preds[d], [refs_list]).score
        except: chrf_scores[d] = 0.0
        comet_scores[d] = 0.0

    # --- save metrics JSON
    with open(OUTPUT_DIR / "metrics.json","w",encoding="utf-8") as f:
        json.dump({"bleu":bleu_scores,"chrf":chrf_scores,"comet":comet_scores}, f, ensure_ascii=False, indent=2)

    # --- plots
    plot_dir = OUTPUT_DIR / "metric_plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    def plot_metric(metric_name, scores_dict):
        if not scores_dict: return
        langs, vals = list(scores_dict.keys()), [scores_dict[k] for k in scores_dict]
        plt.figure(figsize=(12,6))
        plt.bar(langs, vals, color='skyblue')
        plt.title(f"{metric_name} Scores per Direction", fontsize=16)
        plt.xlabel("Language Direction", fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{metric_name.lower()}_per_direction.png")
        plt.close()
    plot_metric("BLEU", bleu_scores)
    plot_metric("chrF", chrf_scores)
    plot_metric("COMET", comet_scores)
    print("📈 Metrics plots saved.") # Replaced logger.info
    return bleu_scores, chrf_scores, comet_scores


# --- helper to process a batch safely
def process_batch(model, tok, batch_prompts, batch_refs, batch_dirs, batch_inputs, preds, refs, inputs, device):
    import gc
    batch_prompts_clean = [p for p in batch_prompts if p and p.strip()]
    batch_dirs_clean   = [d for p,d in zip(batch_prompts, batch_dirs) if p and p.strip()]
    batch_refs_clean   = [r for p,r in zip(batch_prompts, batch_refs) if p and p.strip()]
    batch_inputs_clean = [i for p,i in zip(batch_prompts, batch_inputs) if p and p.strip()]

    if not batch_prompts_clean:
        print("⚠️ Entire batch empty after filtering, skipping...")
        return

    # --- Tokenize safely
    enc = tok(batch_prompts_clean, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
    if not enc.get("input_ids"):
        return
    enc = {k:v.to(device) for k,v in enc.items()}

    # --- generate
    with torch.no_grad():
        out_ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, **BEAM_KWARGS)
    if isinstance(out_ids, torch.Tensor): out_ids = out_ids.cpu().tolist()

    raw_lengths = [len(tok(p, add_special_tokens=False)["input_ids"]) for p in batch_prompts_clean]

    # --- decode
    for i, (dirn, ref, inp, prompt_len) in enumerate(zip(batch_dirs_clean, batch_refs_clean, batch_inputs_clean, raw_lengths)):
        seq = out_ids[i] if isinstance(out_ids[i], list) else list(out_ids[i])
        gen_tokens = seq[prompt_len:] if len(seq) > prompt_len else seq
        pred_text = tok.decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        preds[dirn].append(pred_text)
        refs[dirn].append(ref)
        inputs[dirn].append(inp)

    torch.cuda.empty_cache()
    gc.collect()


# ------------------------------ TRAIN CURVE & LOSS PLOTS
def plot_training(trainer):
    logs = trainer.state.log_history
    df=pd.DataFrame(logs)
    if "loss" in df.columns:
        df["loss_smooth"]=df["loss"].rolling(window=10,min_periods=1).mean()
        plt.figure(figsize=(8,4)); plt.plot(df["step"],df["loss"],alpha=0.5,color="gray",label="Raw Loss")
        plt.plot(df["step"],df["loss_smooth"],color="blue",linewidth=2,label="Smoothed Loss")
        plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss Over Steps"); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"training_loss_smooth.png"); plt.close()
    if "learning_rate" in df.columns:
        plt.figure(figsize=(8,4)); plt.plot(df["step"],df["learning_rate"],color="orange",label="Learning Rate")
        plt.xlabel("Step"); plt.ylabel("LR"); plt.title("Learning Rate Schedule"); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"learning_rate_trend.png"); plt.close()
    if "epoch" in df.columns and "loss" in df.columns:
        plt.figure(figsize=(8,4)); plt.scatter(df["epoch"],df["loss"],color="green",s=20,alpha=0.6,label="Raw Loss per Epoch")
        epoch_means = df.groupby("epoch")["loss"].mean()
        plt.plot(epoch_means.index,epoch_means.values,color="red",linewidth=2,label="Mean Loss per Epoch")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss per Epoch"); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"epoch_loss_trend.png"); plt.close()
    if "loss_smooth" in df.columns and len(df)>5:
        df["loss_derivative"]=np.gradient(df["loss_smooth"])
        plt.figure(figsize=(8,4)); plt.plot(df["step"],df["loss_derivative"],color="purple",label="d(Loss)/d(Step)")
        plt.axhline(0,color="black",linestyle="--",alpha=0.5)
        plt.xlabel("Step"); plt.ylabel("Loss Change"); plt.title("Loss Change Rate"); plt.legend(); plt.tight_layout(); plt.savefig(OUTPUT_DIR/"loss_derivative_curve.png"); plt.close()

# ------------------------------ MAIN
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES

    # 1️⃣ Train
    model, tok, trainer = train_model(max_samples=max_samples)

    # 2️⃣ Evaluate
    bleu, chrf, comet = evaluate_model(model, tok, max_samples_per_split=None if FULL_DATASET else 50, batch_size=EVAL_BATCH_SIZE)

    # 3️⃣ Plot training curves
    plot_training(trainer)

    # 4️⃣ Final summary
    print("\n✅ Training complete!") 
    print(f"📁 All outputs saved to: {OUTPUT_DIR}") 
    print(f"   - Model weights") 
    print(f"   - eng_hin_pred_ref.jsonl") 
    print(f"   - hin_eng_pred_ref.jsonl") 
    print(f"   - submission.zip") 
    print(f"   - training_loss_smooth.png") 
    print(f"   - learning_rate_trend.png") 
    print(f"   - epoch_loss_trend.png") 
    print(f"   - loss_derivative_curve.png") 
    print(f"   - metric_plots/ (BLEU, chrF, COMET)") 
