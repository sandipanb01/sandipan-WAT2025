# -*- coding: utf-8 -*-
# ======================================================
# ✅ CORRECTED IBT PIPELINE - SYNCED WITH REFERENCE CODE
# Round-0 BT (IndicTrans) + Gemma SFT + 2 IBT rounds (R1, R2)
# Model: google/gemma-3-270m-it
# Style: Matches your working parallel data code exactly
# ======================================================

!pip install IndicTransToolkit

import os, random, torch, warnings, gc, json, zipfile
from pathlib import Path
from itertools import islice
from functools import partial
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------- CONFIG (TINY FOR QUICK TEST)
MODEL_NAME = "google/gemma-3-270m-it"
INDIC_TO_EN_CKPT = "ai4bharat/indictrans2-indic-en-1B"
EN_TO_INDIC_CKPT = "ai4bharat/indictrans2-en-indic-1B"

WORK_DIR = Path("./ibt_pipeline")
OUTPUT_DIR = Path("./ibt_outputs")
WORK_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

N_MONO = 6        # VERY SMALL ON PURPOSE
N_TEST = 10
MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 64
MAX_STEPS = 50
BATCH_SIZE = 1
GRAD_ACCUM = 2
INDIC_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ---------------- LOAD MONOLINGUAL DATA (FOR TRAINING)
print("[LOAD] IITB-IndicMonoDoc")

try:
    print("  Attempting: Loading full dataset and filtering by language...")
    mono = load_dataset("cfilt/IITB-IndicMonoDoc", split="train", trust_remote_code=True)

    hi_docs = [x["text"] for x in mono if x["lang"] == "hi"][:N_MONO]
    en_docs = [x["text"] for x in mono if x["lang"] == "en"][:N_MONO]

    assert len(hi_docs) == len(en_docs), f"Mismatch: {len(hi_docs)} Hindi vs {len(en_docs)} English"
    print(f"  ✓ Loaded {len(hi_docs)} Hindi + {len(en_docs)} English docs for TRAINING")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    print("  Fallback: Using parallel corpus as monolingual...")
    parallel = load_dataset("cfilt/iitb-english-hindi", split="train", streaming=True)

    parallel_samples = list(islice(parallel, N_MONO))
    hi_docs = [x["translation"]["hi"] for x in parallel_samples]
    en_docs = [x["translation"]["en"] for x in parallel_samples]
    print(f"  ✓ Loaded {len(hi_docs)} Hindi + {len(en_docs)} English texts from parallel corpus")

# ---------------- LOAD TEST DATA (SEPARATE FROM TRAINING)
print("\n[LOAD] Loading separate TEST data from Pralekha")

test_parallel = load_dataset("ai4bharat/Pralekha", name="train", split="eng_hin", streaming=True)
test_samples = list(islice(test_parallel, N_MONO, N_MONO + N_TEST))

test_en = [x["src_txt"] for x in test_samples]
test_hi = [x["tgt_txt"] for x in test_samples]

print(f"  ✓ Loaded {len(test_en)} English + {len(test_hi)} Hindi test samples (SEPARATE from training)")

# ---------------- INDIC TRANS (ROUND 0) - SEQUENTIAL LOADING
print("\n[ROUND-0] IndicTrans backtranslation (sequential to save memory)")

def initialize_indic_model(ckpt_dir):
    """Initialize IndicTrans model"""
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    if device == "cuda":
        model.half()
    model.eval()
    return tokenizer, model

def batch_translate_no_cache(texts, src_lang, tgt_lang, model, tokenizer, ip, batch_size=2):
    """Translate texts in batches"""
    all_outputs = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)

        model_inputs = tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(model.device)

        translated_tokens = model.generate(**model_inputs, use_cache=False, max_length=256)

        outputs = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

        batch_outputs = ip.postprocess_batch(outputs, lang=tgt_lang)
        all_outputs.extend(batch_outputs)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_outputs

# Initialize processor once
from IndicTransToolkit.processor import IndicProcessor
ip = IndicProcessor(inference=True)

# === STEP 1: Hindi → English ===
print("  [1/2] Loading Indic→English model and translating...")
tokenizer_ie, model_ie = initialize_indic_model(INDIC_TO_EN_CKPT)

bt_en_r0 = batch_translate_no_cache(
    hi_docs,
    src_lang="hin_Deva",
    tgt_lang="eng_Latn",
    model=model_ie,
    tokenizer=tokenizer_ie,
    ip=ip,
    batch_size=INDIC_BATCH_SIZE
)
print(f"      Created {len(bt_en_r0)} synthetic English from Hindi")

del model_ie, tokenizer_ie
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
print(f"      Cleaned up Indic→English model")

# === STEP 2: English → Hindi ===
print("  [2/2] Loading English→Indic model and translating...")
tokenizer_ei, model_ei = initialize_indic_model(EN_TO_INDIC_CKPT)

bt_hi_r0 = batch_translate_no_cache(
    en_docs,
    src_lang="eng_Latn",
    tgt_lang="hin_Deva",
    model=model_ei,
    tokenizer=tokenizer_ei,
    ip=ip,
    batch_size=INDIC_BATCH_SIZE
)
print(f"      Created {len(bt_hi_r0)} synthetic Hindi from English")

del model_ei, tokenizer_ei, ip
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
print(f"      Cleaned up English→Indic model")

# ---------------- PREPARE MODEL (MATCHING REFERENCE CODE)
def prepare_model():
    """Load model and tokenizer - EXACT match to reference"""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    try:
        model.gradient_checkpointing_enable()
    except:
        pass

    return model, tok

# ---------------- BUILD DATASET (MATCHING REFERENCE CODE)
def build_bidirectional_dataset(src1, tgt1, src2, tgt2, lang1, lang2):
    """Build dataset with proper language labels for both directions"""
    data = []
    # Direction 1: src1 → tgt1
    for src, tgt in zip(src1, tgt1):
        prompt = f"Translate this {lang1} text to {lang2}:\n{src}"
        data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": tgt},
            ]
        })
    # Direction 2: src2 → tgt2
    for src, tgt in zip(src2, tgt2):
        prompt = f"Translate this {lang2} text to {lang1}:\n{src}"
        data.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": tgt},
            ]
        })
    return Dataset.from_list(data)

# Round-0 dataset: Synthetic as SOURCE, Original as TARGET
print("\n[DATASET] Building Round-0 training data")
round0_ds = build_bidirectional_dataset(
    src1=bt_en_r0,      # Synthetic English
    tgt1=hi_docs,       # Original Hindi
    src2=bt_hi_r0,      # Synthetic Hindi
    tgt2=en_docs,       # Original English
    lang1="English",
    lang2="Hindi"
)
print(f"Round-0 dataset: {len(round0_ds)} examples (both directions)")

# ---------------- EVALUATION FUNCTIONS (MATCHING REFERENCE CODE)

def build_eval_prompt_messages(src_text, src_lang, tgt_lang):
    """Build evaluation prompt - EXACT match to reference"""
    user_prompt = f"Translate this {src_lang} text to {tgt_lang}:\n{src_text}"
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": ""}
    ]

class EvalDataset(IterableDataset):
    """Streaming Dataset Wrapper - EXACT match to reference"""
    def __init__(self, src_texts, ref_texts, tokenizer, src_lang, tgt_lang):
        self.src_texts = src_texts
        self.ref_texts = ref_texts
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __iter__(self):
        for src_text, ref_text in zip(self.src_texts, self.ref_texts):
            messages = build_eval_prompt_messages(src_text, self.src_lang, self.tgt_lang)

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "reference": ref_text.strip(),
                "source": src_text.strip()
            }

def eval_collate_fn(batch, tokenizer):
    """Collate Function - EXACT match to reference"""
    input_ids = [x["input_ids"] for x in batch]
    refs = [x["reference"] for x in batch]
    srcs = [x["source"] for x in batch]

    enc = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    )

    return enc["input_ids"], enc["attention_mask"], refs, srcs

def generate_batch(model, tokenizer, input_ids, attention_mask):
    """Generation - EXACT match to reference"""
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
        prompt_len = attention_mask[i].sum().item()
        gen_ids = outputs[i][prompt_len:]
        preds.append(
            tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        )

    return preds

def evaluate_direction(model, tokenizer, src_texts, ref_texts, src_lang, tgt_lang,
                       round_name, batch_size=8):
    """Evaluation Function - Saves input/pred/ref JSONL"""

    print(f"\n[EVAL {round_name}] {src_lang}→{tgt_lang}")

    eval_ds = EvalDataset(src_texts, ref_texts, tokenizer, src_lang, tgt_lang)
    collate = partial(eval_collate_fn, tokenizer=tokenizer)

    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0
    )

    preds, refs, sources = [], [], []

    pbar = tqdm(desc=f"Evaluating {src_lang}→{tgt_lang}", total=len(src_texts))

    for input_ids, attention_mask, batch_refs, batch_srcs in loader:
        batch_preds = generate_batch(model, tokenizer, input_ids, attention_mask)

        preds.extend(batch_preds)
        refs.extend(batch_refs)
        sources.extend(batch_srcs)

        pbar.update(len(batch_refs))

    pbar.close()

    # Calculate metrics
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.metrics.CHRF(word_order=0).corpus_score(preds, [refs]).score

    print(f"  {src_lang}→{tgt_lang} | BLEU={bleu:.2f} | chrF={chrf:.2f}")

    # Save JSONL to OUTPUT_DIR
    jsonl_path = OUTPUT_DIR / f"{round_name}_{src_lang}_{tgt_lang}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for src, pred, ref in zip(sources, preds, refs):
            f.write(json.dumps({
                "input": src,
                "prediction": pred,
                "reference": ref
            }, ensure_ascii=False) + "\n")

    print(f"  Saved: {jsonl_path}")

    return bleu, chrf

def gemma_translate(model, tokenizer, texts, src_lang, tgt_lang, batch_size=2):
    """Generate translations using current Gemma model"""
    eval_ds = EvalDataset(texts, [""] * len(texts), tokenizer, src_lang, tgt_lang)
    collate = partial(eval_collate_fn, tokenizer=tokenizer)

    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0
    )

    preds = []
    for input_ids, attention_mask, _, _ in tqdm(loader, desc=f"Translating {src_lang}→{tgt_lang}"):
        batch_preds = generate_batch(model, tokenizer, input_ids, attention_mask)
        preds.extend(batch_preds)

    return preds

# ---------------- LOAD GEMMA + LORA (MATCHING REFERENCE CODE)
print("\n[SETUP] Loading Gemma with LoRA")
model, tok = prepare_model()

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",

)

model = get_peft_model(model, peft_config)
print(f"Trainable params: {model.print_trainable_parameters()}")

# ---------------- FINETUNE ROUND 0 (MATCHING REFERENCE CODE)
print("\n[TRAIN] Gemma Round-0")
cfg = SFTConfig(
    output_dir=str(WORK_DIR / "gemma_r0"),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    num_train_epochs=1,
    max_steps=MAX_STEPS,
    save_steps=MAX_STEPS,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    warmup_ratio=0.05,
    gradient_checkpointing=True,
    completion_only_loss=True,
    packing=False
)

trainer = SFTTrainer(
    model=model,
    args=cfg,
    train_dataset=round0_ds,
    peft_config=peft_config,
)

trainer.train()

model.save_pretrained(WORK_DIR / "gemma_r0" / "final")
tok.save_pretrained(WORK_DIR / "gemma_r0" / "final")

# Initialize metrics dictionary
metrics = {}

# Evaluate Round-0 on SEPARATE TEST DATA
print("\n" + "="*60)
print("EVALUATING ROUND-0 (on held-out test set)")
print("="*60)
bleu_en_hi_r0, chrf_en_hi_r0 = evaluate_direction(
    model, tok, test_en, test_hi, "English", "Hindi", "r0", batch_size=EVAL_BATCH_SIZE
)
bleu_hi_en_r0, chrf_hi_en_r0 = evaluate_direction(
    model, tok, test_hi, test_en, "Hindi", "English", "r0", batch_size=EVAL_BATCH_SIZE
)

metrics["round-0"] = {
    "eng_to_hin": {"bleu": bleu_en_hi_r0, "chrf": chrf_en_hi_r0},
    "hin_to_eng": {"bleu": bleu_hi_en_r0, "chrf": chrf_hi_en_r0}
}

# ---------------- ITERATIVE BT R1 + R2
print("\n" + "="*60)
print("STARTING ITERATIVE BACKTRANSLATION")
print("="*60)

for r in [1, 2]:
    print(f"\n[IBT ROUND {r}]")

    # Generate NEW synthetic translations using current Gemma
    print(f"  Step 1: Generating synthetic English from {len(hi_docs)} Hindi texts")
    gen_en = gemma_translate(model, tok, hi_docs, "Hindi", "English", batch_size=EVAL_BATCH_SIZE)

    print(f"  Step 2: Generating synthetic Hindi from {len(en_docs)} English texts")
    gen_hi = gemma_translate(model, tok, en_docs, "English", "Hindi", batch_size=EVAL_BATCH_SIZE)

    # Build dataset: Synthetic as SOURCE, Original as TARGET
    print(f"  Step 3: Building Round-{r} dataset")
    round_ds = build_bidirectional_dataset(
        src1=gen_en,        # Synthetic English
        tgt1=hi_docs,       # Original Hindi
        src2=gen_hi,        # Synthetic Hindi
        tgt2=en_docs,       # Original English
        lang1="English",
        lang2="Hindi"
    )
    print(f"    Dataset size: {len(round_ds)} examples (both directions)")

    # Update config for this round
    cfg.output_dir = str(WORK_DIR / f"gemma_r{r}")
    cfg.save_steps = MAX_STEPS

    # Train (model already has LoRA from R0)
    print(f"  Step 4: Training Gemma Round-{r}")
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=round_ds,
    )
    trainer.train()

    # Save checkpoint
    model.save_pretrained(WORK_DIR / f"gemma_r{r}" / "final")
    tok.save_pretrained(WORK_DIR / f"gemma_r{r}" / "final")

    # Evaluate this round
    print(f"\n  Step 5: Evaluating Round-{r}")
    bleu_en_hi, chrf_en_hi = evaluate_direction(
        model, tok, test_en, test_hi, "English", "Hindi", f"r{r}", batch_size=EVAL_BATCH_SIZE
    )
    bleu_hi_en, chrf_hi_en = evaluate_direction(
        model, tok, test_hi, test_en, "Hindi", "English", f"r{r}", batch_size=EVAL_BATCH_SIZE
    )

    metrics[f"round-{r}"] = {
        "eng_to_hin": {"bleu": bleu_en_hi, "chrf": chrf_en_hi},
        "hin_to_eng": {"bleu": bleu_hi_en, "chrf": chrf_hi_en}
    }

    print(f"  ✓ Round {r} complete")

print("\n" + "="*60)
print("✅ PIPELINE COMPLETED SUCCESSFULLY")
print("="*60)

# Print metrics summary
print("\n📊 METRICS SUMMARY:")
print("="*60)
for round_name, round_metrics in metrics.items():
    print(f"\n{round_name.upper()}:")
    for direction, scores in round_metrics.items():
        dir_label = "English→Hindi" if direction == "eng_to_hin" else "Hindi→English"
        print(f"  {dir_label}: BLEU={scores['bleu']:.2f}, chrF={scores['chrf']:.2f}")

print(f"\nAll checkpoints saved in: {WORK_DIR}")
print(f"All JSONL outputs saved in: {OUTPUT_DIR}")
print("\nFiles generated:")
for r in ["r0", "r1", "r2"]:
    print(f"  - {r}_English_Hindi.jsonl")
    print(f"  - {r}_Hindi_English.jsonl")

# Create ZIP file with all JSONL results from OUTPUT_DIR
print("\n[ZIP] Creating archive of all results...")
zip_path = OUTPUT_DIR / "ibt_results.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for jsonl_file in OUTPUT_DIR.glob("*.jsonl"):
        zipf.write(jsonl_file, arcname=jsonl_file.name)

print(f"✓ ZIP saved at: {zip_path}")
