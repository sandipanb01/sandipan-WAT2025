# -- coding: utf-8 --
# ======================================================
# ✅ UPDATED IBT PIPELINE
# Round-0 BT (IndicTrans) + Gemma SFT + 2 IBT rounds (R1, R2)
# Model: google/gemma-3-4b-it
# Languages: English ↔ Hindi ONLY
# Fine-tune & Eval aligned with reference scripts
# ======================================================

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
import sacrebleu
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────
MODEL_NAME        = "google/gemma-3-4b-it"
INDIC_TO_EN_CKPT  = "ai4bharat/indictrans2-indic-en-1B"
EN_TO_INDIC_CKPT  = "ai4bharat/indictrans2-en-indic-1B"

WORK_DIR   = Path("./ibt_pipeline")
OUTPUT_DIR = Path("./ibt_outputs")
WORK_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

N_MONO          = 6        # monolingual docs for training
N_TEST          = 10       # held-out test samples
MAX_SEQ_LEN     = 8192
MAX_NEW_TOKENS  = 256
MAX_STEPS       = 50
BATCH_SIZE      = 4
GRAD_ACCUM      = 8
INDIC_BATCH_SIZE = 2
EVAL_BATCH_SIZE  = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42

random.seed(SEED)
torch.manual_seed(SEED)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# ──────────────────────────────────────────────────────
# LOAD MONOLINGUAL DATA (TRAINING)
# ──────────────────────────────────────────────────────
print("[LOAD] IITB-IndicMonoDoc")

try:
    print("  Attempting: Loading full dataset and filtering by language...")
    mono = load_dataset("cfilt/IITB-IndicMonoDoc", split="train", trust_remote_code=True)
    hi_docs = [x["text"] for x in mono if x["lang"] == "hi"][:N_MONO]
    en_docs = [x["text"] for x in mono if x["lang"] == "en"][:N_MONO]
    assert len(hi_docs) == len(en_docs), f"Mismatch: {len(hi_docs)} Hindi vs {len(en_docs)} English"
    print(f"  ✓ Loaded {len(hi_docs)} Hindi + {len(en_docs)} English docs")

except Exception as e:
    print(f"  ✗ Failed: {e}")
    print("  Fallback: Using parallel corpus as monolingual...")
    parallel = load_dataset("cfilt/iitb-english-hindi", split="train", streaming=True)
    parallel_samples = list(islice(parallel, N_MONO))
    hi_docs = [x["translation"]["hi"] for x in parallel_samples]
    en_docs = [x["translation"]["en"] for x in parallel_samples]
    print(f"  ✓ Loaded {len(hi_docs)} Hindi + {len(en_docs)} English texts from parallel corpus")


# ──────────────────────────────────────────────────────
# LOAD TEST DATA  (held-out, never used in training)
# ──────────────────────────────────────────────────────
print("\n[LOAD] Separate TEST data from Pralekha (eng_hin split)")

test_parallel  = load_dataset(
    "ai4bharat/Pralekha", name="train", split="eng_hin", streaming=True
)
test_samples = list(islice(test_parallel, N_MONO, N_MONO + N_TEST))
test_en = [x["src_txt"] for x in test_samples]
test_hi = [x["tgt_txt"] for x in test_samples]
print(f"  ✓ Loaded {len(test_en)} EN + {len(test_hi)} HI test samples")


# ──────────────────────────────────────────────────────
# INDICTRANS HELPERS
# ──────────────────────────────────────────────────────
def initialize_indic_model(ckpt_dir):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


def batch_translate_no_cache(texts, src_lang, tgt_lang, model, tokenizer, ip, batch_size=2):
    all_outputs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = ip.preprocess_batch(batch, src_lang=src_lang, tgt_lang=tgt_lang)
        model_inputs = tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True, max_length=256
        ).to(model.device)
        translated_tokens = model.generate(**model_inputs, use_cache=False, max_length=256)
        outputs = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        all_outputs.extend(ip.postprocess_batch(outputs, lang=tgt_lang))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return all_outputs


# ──────────────────────────────────────────────────────
# ROUND-0  IndicTrans back-translation
# ──────────────────────────────────────────────────────
print("\n[ROUND-0] IndicTrans back-translation (BF16)")

from IndicTransToolkit.processor import IndicProcessor
ip = IndicProcessor(inference=True)

# Hindi → English
print("  [1/2] Indic→EN model ...")
tokenizer_ie, model_ie = initialize_indic_model(INDIC_TO_EN_CKPT)
bt_en_r0 = batch_translate_no_cache(
    hi_docs, "hin_Deva", "eng_Latn", model_ie, tokenizer_ie, ip, INDIC_BATCH_SIZE
)
print(f"      Created {len(bt_en_r0)} synthetic English sentences")
del model_ie, tokenizer_ie
torch.cuda.empty_cache(); gc.collect()

# English → Hindi
print("  [2/2] EN→Indic model ...")
tokenizer_ei, model_ei = initialize_indic_model(EN_TO_INDIC_CKPT)
bt_hi_r0 = batch_translate_no_cache(
    en_docs, "eng_Latn", "hin_Deva", model_ei, tokenizer_ei, ip, INDIC_BATCH_SIZE
)
print(f"      Created {len(bt_hi_r0)} synthetic Hindi sentences")
del model_ei, tokenizer_ei, ip
torch.cuda.empty_cache(); gc.collect()


# ──────────────────────────────────────────────────────
# DATASET BUILDER
# Prompt format matches reference fine-tuning script:
#   prompt/completion dict, English↔Hindi only
# ──────────────────────────────────────────────────────
def make_en2hi_example(src_en: str, tgt_hi: str) -> dict:
    """English → Hindi training example."""
    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Translate the following sentence from English to Hindi.\n\n"
                    f"English: {src_en}"
                ),
            }
        ],
        "completion": [{"role": "assistant", "content": tgt_hi}],
    }


def make_hi2en_example(src_hi: str, tgt_en: str) -> dict:
    """Hindi → English training example."""
    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    "Translate the following sentence from Hindi to English.\n\n"
                    f"Hindi: {src_hi}"
                ),
            }
        ],
        "completion": [{"role": "assistant", "content": tgt_en}],
    }


def build_bidirectional_dataset(
    synthetic_en, original_hi,   # EN→HI direction: synthetic EN as source
    synthetic_hi, original_en,   # HI→EN direction: synthetic HI as source
) -> Dataset:
    """
    Synthetic text is always the SOURCE (input to translate FROM),
    original text is always the TARGET (gold translation).

    EN→HI : model sees synthetic English, must produce original Hindi
    HI→EN : model sees synthetic Hindi,   must produce original English
    """
    data = []
    for syn_en, orig_hi in zip(synthetic_en, original_hi):
        data.append(make_en2hi_example(syn_en, orig_hi))
    for syn_hi, orig_en in zip(synthetic_hi, original_en):
        data.append(make_hi2en_example(syn_hi, orig_en))
    return Dataset.from_list(data)


# Round-0 dataset
print("\n[DATASET] Building Round-0 training data")
round0_ds = build_bidirectional_dataset(
    synthetic_en=bt_en_r0,   # synth EN  → target: original HI
    original_hi=hi_docs,
    synthetic_hi=bt_hi_r0,   # synth HI  → target: original EN
    original_en=en_docs,
)
print(f"  Round-0 dataset: {len(round0_ds)} examples")
print(f"  Sample prompt:\n  {round0_ds[0]['prompt'][0]['content'][:120]}")


# ──────────────────────────────────────────────────────
# MODEL + LoRA  (matches reference script exactly)
# ──────────────────────────────────────────────────────
def prepare_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    try:
        mdl.gradient_checkpointing_enable()
    except Exception:
        pass
    return mdl, tok


lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[          # explicit modules, matching reference script
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)


def make_sft_config(output_subdir: str) -> SFTConfig:
    return SFTConfig(
        output_dir=str(WORK_DIR / output_subdir),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=1,
        max_steps=MAX_STEPS,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        bf16=True,
        max_length=MAX_SEQ_LEN,
        packing=False,
        completion_only_loss=True,
    )


# ──────────────────────────────────────────────────────
# EVALUATION HELPERS
# Prompt format mirrors reference eval script (build_prompt_wat)
# ──────────────────────────────────────────────────────

def build_eval_prompt(src_text: str, direction: str, tokenizer) -> list[int]:
    """
    direction: "en2hi" or "hi2en"
    Returns tokenized input_ids (list[int]).
    """
    if direction == "en2hi":
        content = (
            "Translate the following text from English to Hindi:\n"
            f"English: {src_text}\n"
            "Hindi: "
        )
    else:
        content = (
            "Translate the following text from Hindi to English:\n"
            f"Hindi: {src_text}\n"
            "English: "
        )

    messages = [{"role": "user", "content": content}]
    prompt_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = tokenizer(prompt_str, truncation=True, padding=False)
    return tokens["input_ids"]


class EvalDataset(IterableDataset):
    def _init_(self, src_texts, ref_texts, tokenizer, direction):
        self.src_texts  = src_texts
        self.ref_texts  = ref_texts
        self.tokenizer  = tokenizer
        self.direction  = direction

    def _iter_(self):
        for src, ref in zip(self.src_texts, self.ref_texts):
            input_ids = build_eval_prompt(src, self.direction, self.tokenizer)
            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "reference": ref.strip(),
                "source":    src.strip(),
            }


def eval_collate_fn(batch, tokenizer):
    input_ids = [x["input_ids"] for x in batch]
    refs = [x["reference"] for x in batch]
    srcs = [x["source"]    for x in batch]
    enc  = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], refs, srcs


def generate_batch(model, tokenizer, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    preds = []
    for i in range(len(outputs)):
        prompt_len = attention_mask[i].sum().item()
        gen_ids = outputs[i][prompt_len:]
        preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return preds


def evaluate_direction(model, tokenizer, src_texts, ref_texts,
                       direction: str, round_name: str, batch_size=4):
    """
    direction: "en2hi" or "hi2en"
    Saves JSONL to OUTPUT_DIR and returns (bleu, chrf).
    """
    dir_label = "English→Hindi" if direction == "en2hi" else "Hindi→English"
    print(f"\n[EVAL {round_name}] {dir_label}")

    ds      = EvalDataset(src_texts, ref_texts, tokenizer, direction)
    collate = partial(eval_collate_fn, tokenizer=tokenizer)
    loader  = DataLoader(ds, batch_size=batch_size, collate_fn=collate, num_workers=0)

    preds, refs, sources = [], [], []

    for input_ids, attn_mask, batch_refs, batch_srcs in tqdm(
        loader, desc=dir_label, total=len(src_texts) // batch_size + 1
    ):
        batch_preds = generate_batch(model, tokenizer, input_ids, attn_mask)
        preds.extend(batch_preds)
        refs.extend(batch_refs)
        sources.extend(batch_srcs)

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    bleu = round(bleu, 2)
    chrf = round(chrf, 2)

    print(f"  {dir_label} | BLEU={bleu} | chrF={chrf}")

    jsonl_path = OUTPUT_DIR / f"{round_name}_{direction}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for src, pred, ref in zip(sources, preds, refs):
            f.write(json.dumps({"input": src, "prediction": pred, "reference": ref},
                               ensure_ascii=False) + "\n")
    print(f"  Saved: {jsonl_path}")

    return bleu, chrf


def gemma_translate(model, tokenizer, texts, direction: str, batch_size=4) -> list[str]:
    """Generate translations using the current Gemma model (for IBT loop)."""
    ds      = EvalDataset(texts, [""] * len(texts), tokenizer, direction)
    collate = partial(eval_collate_fn, tokenizer=tokenizer)
    loader  = DataLoader(ds, batch_size=batch_size, collate_fn=collate, num_workers=0)

    preds = []
    dir_label = "EN→HI" if direction == "en2hi" else "HI→EN"
    for input_ids, attn_mask, _, _ in tqdm(loader, desc=f"Translating {dir_label}"):
        preds.extend(generate_batch(model, tokenizer, input_ids, attn_mask))
    return preds


# ──────────────────────────────────────────────────────
# LOAD GEMMA + LoRA
# ──────────────────────────────────────────────────────
print("\n[SETUP] Loading Gemma-3-4B-IT with LoRA (BF16)")
model, tok = prepare_model_and_tokenizer()
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ──────────────────────────────────────────────────────
# TRAIN ROUND-0
# ──────────────────────────────────────────────────────
print("\n[TRAIN] Round-0 SFT")

trainer = SFTTrainer(
    model=model,
    train_dataset=round0_ds,
    peft_config=lora_config,
    args=make_sft_config("gemma_r0"),
    processing_class=tok,        # matches reference script
)
trainer.train()

model.save_pretrained(WORK_DIR / "gemma_r0" / "final")
tok.save_pretrained(WORK_DIR / "gemma_r0" / "final")


# ──────────────────────────────────────────────────────
# EVAL ROUND-0
# ──────────────────────────────────────────────────────
metrics = {}

print("\n" + "="*60)
print("EVALUATING ROUND-0 (held-out test set)")
print("="*60)

bleu_en_hi_r0, chrf_en_hi_r0 = evaluate_direction(
    model, tok, test_en, test_hi, "en2hi", "r0", EVAL_BATCH_SIZE
)
bleu_hi_en_r0, chrf_hi_en_r0 = evaluate_direction(
    model, tok, test_hi, test_en, "hi2en", "r0", EVAL_BATCH_SIZE
)

metrics["round-0"] = {
    "en2hi": {"bleu": bleu_en_hi_r0, "chrf": chrf_en_hi_r0},
    "hi2en": {"bleu": bleu_hi_en_r0, "chrf": chrf_hi_en_r0},
}


# ──────────────────────────────────────────────────────
# ITERATIVE BACK-TRANSLATION  R1 + R2
# ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("STARTING ITERATIVE BACK-TRANSLATION  (R1, R2)")
print("="*60)

for r in [1, 2]:
    print(f"\n{'='*60}")
    print(f"[IBT ROUND {r}]")
    print(f"{'='*60}")

    # Step 1: Current Gemma generates synthetic translations
    print(f"  Step 1: HI→EN — translating {len(hi_docs)} Hindi docs")
    gen_en = gemma_translate(model, tok, hi_docs, "hi2en", EVAL_BATCH_SIZE)

    print(f"  Step 2: EN→HI — translating {len(en_docs)} English docs")
    gen_hi = gemma_translate(model, tok, en_docs, "en2hi", EVAL_BATCH_SIZE)

    # Step 2: Build dataset (synthetic as source, original as target)
    print(f"  Step 3: Building Round-{r} dataset")
    round_ds = build_bidirectional_dataset(
        synthetic_en=gen_en,     # Gemma-generated EN  → target: original HI
        original_hi=hi_docs,
        synthetic_hi=gen_hi,     # Gemma-generated HI  → target: original EN
        original_en=en_docs,
    )
    print(f"    Dataset size: {len(round_ds)} examples")

    # Step 3: Train
    print(f"  Step 4: Training Round-{r}")
    trainer = SFTTrainer(
        model=model,
        train_dataset=round_ds,
        args=make_sft_config(f"gemma_r{r}"),
        processing_class=tok,
    )
    trainer.train()

    model.save_pretrained(WORK_DIR / f"gemma_r{r}" / "final")
    tok.save_pretrained(WORK_DIR / f"gemma_r{r}" / "final")

    # Step 4: Evaluate
    print(f"  Step 5: Evaluating Round-{r}")
    bleu_en_hi, chrf_en_hi = evaluate_direction(
        model, tok, test_en, test_hi, "en2hi", f"r{r}", EVAL_BATCH_SIZE
    )
    bleu_hi_en, chrf_hi_en = evaluate_direction(
        model, tok, test_hi, test_en, "hi2en", f"r{r}", EVAL_BATCH_SIZE
    )

    metrics[f"round-{r}"] = {
        "en2hi": {"bleu": bleu_en_hi, "chrf": chrf_en_hi},
        "hi2en": {"bleu": bleu_hi_en, "chrf": chrf_hi_en},
    }

    print(f"  ✓ Round {r} complete")


# ──────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("✅ PIPELINE COMPLETE")
print("="*60)

print("\n📊 METRICS SUMMARY:")
for round_name, rm in metrics.items():
    print(f"\n  {round_name.upper()}:")
    print(f"    English→Hindi : BLEU={rm['en2hi']['bleu']:.2f}  chrF={rm['en2hi']['chrf']:.2f}")
    print(f"    Hindi→English : BLEU={rm['hi2en']['bleu']:.2f}  chrF={rm['hi2en']['chrf']:.2f}")

# Save metrics JSON
metrics_path = OUTPUT_DIR / "metrics_summary.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\n  Metrics saved: {metrics_path}")

# ZIP all outputs
print("\n[ZIP] Archiving results...")
zip_path = OUTPUT_DIR / "ibt_results.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for p in OUTPUT_DIR.glob("*.jsonl"):
        zipf.write(p, arcname=p.name)
    zipf.write(metrics_path, arcname=metrics_path.name)
print(f"  ✓ ZIP saved: {zip_path}")
print(f"\n  Checkpoints : {WORK_DIR}")
print(f"  JSONL outputs: {OUTPUT_DIR}")
