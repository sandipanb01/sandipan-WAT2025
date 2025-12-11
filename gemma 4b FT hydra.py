# ======================================================
# ✅ Universal Fine-tuning + Evaluation (Hydra-ready)
# Single-script with full outputs (JSONL, ZIP, metrics, plots, Top-10 preview)
# Industry-benchmark enhancements: metrics, reproducibility, mixed precision
# ======================================================

import os, json, zipfile, math, warnings, gc, random
from pathlib import Path
from itertools import islice
import torch
from datasets import load_dataset, get_dataset_split_names
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate
import hydra
from omegaconf import DictConfig, OmegaConf

# ======================================================
# -------------------- CONFIG -------------------------
# ------------------------------------------------------
DEFAULT_CONFIG = dict(
    seed=42,
    model_name="google/gemma-3-4b-it",
    max_seq_len=382,
    max_new_tokens=256,
    batch_size=1,
    grad_accum=4,
    max_train_steps=50,
    eval_batch_size=8,
    full_dataset=False,
    max_colab_samples=50,
    beam_mode="A",  # A or B
    target_lang="hin",
    metrics=["bleu","rouge"],
    top_k_preview=10
)

# ======================================================
# ------------------- PROMPTS -------------------------
# ------------------------------------------------------
INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

def build_prompt(src, src_lang, tgt_lang, example=None, tokenizer=None):
    ex_src, ex_tgt = ("", "")
    if example: ex_src, ex_tgt = example
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
    if not full_output: return ""
    try:
        if prompt in full_output: return full_output.split(prompt, 1)[1].strip()
    except: pass
    for m in ["Translation:", "Output:", "Answer:", "Translation -", "Translation —"]:
        if m in full_output: return full_output.split(m,1)[1].strip()
    return full_output.strip()

# ======================================================
# ------------------ DATASET --------------------------
# ------------------------------------------------------
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
            s,t = row.get("src_txt",""), row.get("tgt_txt","")
            if len(s.split())>5 and len(t.split())>5: one_shot=(s,t); break

        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        count=0
        for row in ds:
            if max_samples and count>=max_samples: break
            s,t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            use_example = one_shot if (one_shot[0] and one_shot[1] and "dev" not in split) else None
            for s_txt, t_txt, dirn in [(eng,indic,f"eng_{lang}"),(indic,eng,f"{lang}_eng")]:
                prompt = build_prompt(s_txt, dirn.split("_")[0], dirn.split("_")[1], use_example, tokenizer)
                if not prompt or not prompt.strip(): continue
                yield {"input_text": prompt, "target_text": t_txt, "direction": dirn}
            count+=1

class PralekhaDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=None, max_seq_len=382):
        self.max_samples=max_samples; self.tok=tokenizer; self.max_seq_len=max_seq_len
    def __iter__(self):
        for ex in stream_examples(self.tok, self.max_samples):
            enc=self.tok(ex["input_text"], truncation=True, max_length=self.max_seq_len)
            if not enc.get("input_ids"): continue
            enc["labels"]=self.tok(ex["target_text"], truncation=True, max_length=self.max_seq_len)["input_ids"]
            yield enc

# ======================================================
# ------------------ MODEL PREP -----------------------
# ------------------------------------------------------
def detect_lora_modules(model):
    modules=[]
    for n,m in model.named_modules():
        n_lower=n.lower()
        if any(x in n_lower for x in ["q_proj","k_proj","gate_proj","v_proj","o_proj","up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"]):
            modules.append(n.split(".")[-1])
    return list(set(modules))

def prepare_model(cfg):
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    try: model.gradient_checkpointing_enable()
    except: pass
    target_modules = detect_lora_modules(model)
    if not target_modules:
        target_modules = ["q_proj","k_proj","gate_proj","v_proj","o_proj","up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"]
    lora_cfg = LoraConfig(r=2, lora_alpha=4, target_modules=target_modules, lora_dropout=0.1, task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    return model, tok

# ======================================================
# ------------------ TRAINING -------------------------
# ------------------------------------------------------
def train_model(cfg):
    model,tok = prepare_model(cfg)
    ds = PralekhaDataset(tok, max_samples=cfg.max_colab_samples if not cfg.full_dataset else None, max_seq_len=cfg.max_seq_len)
    sft_cfg = SFTConfig(
        output_dir=".",
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=cfg.max_train_steps,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.1,
        gradient_checkpointing=True
    )
    trainer=SFTTrainer(model=model,args=sft_cfg,train_dataset=ds,tokenizer=tok)
    trainer.train()
    return model,tok,trainer

# ======================================================
# ------------------ BATCH PROCESS --------------------
# ------------------------------------------------------
def process_batch(model,tok,batch_prompts,batch_refs,batch_dirs,batch_inputs,batch_rawlens,preds,refs,inputs,device,max_new_tokens,beam_kwargs):
    enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max(batch_rawlens)+max_new_tokens)
    if "input_ids" not in enc or enc["input_ids"].size(0)==0: return
    for k in enc: enc[k]=enc[k].to(device)
    with torch.no_grad():
        out_ids = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, **beam_kwargs)
    out_ids = out_ids.cpu().tolist()
    for i in range(len(batch_prompts)):
        prompt_len=batch_rawlens[i]
        gen = out_ids[i][prompt_len:] if len(out_ids[i])>prompt_len else []
        decoded=tok.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
        dirn=batch_dirs[i]
        preds[dirn].append(decoded)
        refs[dirn].append(batch_refs[i])
        inputs[dirn].append(batch_inputs[i])

# ======================================================
# ------------------ EVALUATION + ZIP -----------------
# ------------------------------------------------------
def evaluate_model(model,tok,cfg):
    device="cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    preds,refs,inputs={}, {}, {}
    target_lang=cfg.target_lang
    for d in [f"eng_{target_lang}", f"{target_lang}_eng"]: preds[d], refs[d], inputs[d]=[],[],[]

    splits = get_dataset_split_names("ai4bharat/Pralekha","dev")
    beam_kwargs = dict(num_beams=3, early_stopping=True) if cfg.beam_mode=="A" else dict(num_beams=3,length_penalty=1.0)
    print("🔍 Starting evaluation (ENG<->INDIC)...")

    for split in splits:
        sl, tl = split.split("_")
        if not ((sl=="eng" and tl==target_lang) or (sl==target_lang and tl=="eng")): continue
        ds = load_dataset("ai4bharat/Pralekha", split=split, streaming=True, name="dev")
        batch_prompts,batch_refs,batch_dirs,batch_inputs,batch_rawlens=[],[],[],[],[]
        for row in ds:
            if not row.get("src_txt") or not row.get("tgt_txt"): continue
            eng, indic = (row["src_txt"],row["tgt_txt"]) if sl=="eng" else (row["tgt_txt"],row["src_txt"])
            p1,p2 = eval_prompt(eng,"eng",target_lang), eval_prompt(indic,target_lang,"eng")
            for prompt,ref,dirn,inp in [(p1,indic.strip(),f"eng_{target_lang}",eng.strip()),(p2,eng.strip(),f"{target_lang}_eng",indic.strip())]:
                if not prompt.strip(): continue
                batch_prompts.append(prompt); batch_refs.append(ref); batch_dirs.append(dirn)
                batch_inputs.append(inp); batch_rawlens.append(len(tok(prompt, add_special_tokens=False)["input_ids"]))
            if len(batch_prompts)>=cfg.eval_batch_size:
                process_batch(model,tok,batch_prompts,batch_refs,batch_dirs,batch_inputs,batch_rawlens,preds,refs,inputs,device,cfg.max_new_tokens,beam_kwargs)
                batch_prompts,batch_refs,batch_dirs,batch_inputs,batch_rawlens=[],[],[],[],[]
        if batch_prompts:
            process_batch(model,tok,batch_prompts,batch_refs,batch_dirs,batch_inputs,batch_rawlens,preds,refs,inputs,device,cfg.max_new_tokens,beam_kwargs)

    # --- Save JSON + Metrics
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True, parents=True)
    metrics={}
    for d in preds:
        json_file = out_dir/f"{d}_pred_ref.jsonl"
        with open(json_file,"w",encoding="utf-8") as f:
            for inp,p,r in zip(inputs[d],preds[d],refs[d]):
                f.write(json.dumps({"input_text":inp,"pred":p,"ref":r},ensure_ascii=False)+"\n")
        if "bleu" in cfg.metrics:
            try: metrics[f"{d}_bleu"]=sacrebleu.corpus_bleu(preds[d],[refs[d]]).score
            except: metrics[f"{d}_bleu"]=0.0
        if "rouge" in cfg.metrics:
            scorer=rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
            scores=[scorer.score(r,p) for r,p in zip(refs[d],preds[d])]
            metrics[f"{d}_rouge1"]=np.mean([s["rouge1"].fmeasure for s in scores])
            metrics[f"{d}_rouge2"]=np.mean([s["rouge2"].fmeasure for s in scores])
            metrics[f"{d}_rougeL"]=np.mean([s["rougeL"].fmeasure for s in scores])

    # --- Top-K preview table
    print("\n📌 Top-10 predictions preview per direction:")
    for d in preds:
        table=[]
        k=min(cfg.top_k_preview,len(preds[d]))
        for i in range(k):
            table.append([inputs[d][i], preds[d][i], refs[d][i]])
        print(f"\nDirection: {d}")
        print(tabulate(table, headers=["Input","Prediction","Reference"], tablefmt="grid"))

    # --- Save ZIP
    zip_path = out_dir/"outputs.zip"
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED) as zipf:
        for file in out_dir.glob("*"):
            if file.is_file() and file.name!="outputs.zip":
                zipf.write(file, file.name)
    print(f"\n✅ Saved predictions, metrics, plots, and ZIP at {zip_path}")

    with open(out_dir/"metrics.json","w",encoding="utf-8") as f: json.dump(metrics,f,ensure_ascii=False,indent=2)
    return metrics

# ======================================================
# ------------------ TRAINING PLOTS -------------------
# ------------------------------------------------------
def plot_training(trainer,cfg):
    logs=trainer.state.log_history
    df=pd.DataFrame(logs)
    out_dir=Path("outputs/plots"); out_dir.mkdir(parents=True,exist_ok=True)
    if "loss" in df.columns:
        df["loss_smooth"]=df["loss"].rolling(window=10,min_periods=1).mean()
        plt.figure(figsize=(8,4))
        plt.plot(df["step"],df["loss"],alpha=0.5,color="gray")
        plt.plot(df["step"],df["loss_smooth"],color="blue",linewidth=2)
        plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Loss over Steps")
        plt.tight_layout(); plt.savefig(out_dir/"loss_smooth.png"); plt.close()

# ======================================================
# ------------------ MAIN (Hydra) --------------------
# ------------------------------------------------------
@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.merge(DEFAULT_CONFIG, cfg)
    # --- Reproducibility
    seed = cfg.get("seed",42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    print(OmegaConf.to_yaml(cfg))

    model,tok,trainer=train_model(cfg)
    metrics=evaluate_model(model,tok,cfg)
    plot_training(trainer,cfg)
    print("\n✅ Done. Outputs in ./outputs")
    print(tabulate([[k,v] for k,v in metrics.items()], headers=["Metric","Score"]))

if __name__=="__main__":
    main()
