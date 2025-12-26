import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sacrebleu.metrics import BLEU, CHRF

# --- LangDetect Guard (Prevents ModuleNotFoundError) ---
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langdetect"])
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42

# ============================================================
# CONFIGURATION
# ============================================================
BASE_MODEL_ID = "google/gemma-3-270m-it"
ADAPTER_PATH = "./gemma3-270m-hindi-ft"
DATASET_NAME = "ai4bharat/Pralekha"
MAX_EVAL_SAMPLES = 50  # Number of samples from TRAIN to verify
OUT_DIR = Path("./gemma3_post_ft_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map for LID check
LANG_MAP = {"HINDI": "hi", "ENGLISH": "en"}

# ============================================================
# ANALYTICS UTILS
# ============================================================
def get_lid_score(text: str, target_iso: str) -> int:
    if not text.strip() or len(text) < 3: return 0
    try:
        return 1 if detect(text) == target_iso else 0
    except:
        return 0

def get_semantic_copy_rate(src: str, pred: str) -> float:
    """Measures how much the model 'cheated' by copying the source."""
    return SequenceMatcher(None, src.lower(), pred.lower()).ratio()

def make_strict_prompt(src_text: str, src_lang: str, tgt_lang: str) -> str:
    """Strictly aligned with Gemma-3 Technical Report prompt format."""
    return (
        f"<start_of_turn>user\n"
        f"Translate from {src_lang.upper()} to {tgt_lang.upper()}:\n{src_text}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )

# ============================================================
# LOAD & MERGE MODEL (Strict Float32)
# ============================================================
print("Merging weights for strict Float32 inference...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID, 
    torch_dtype=torch.float32, 
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()
model.eval()

# ============================================================
# EVALUATION LOOP
# ============================================================
directions = [("ENGLISH", "HINDI"), ("HINDI", "ENGLISH")]
summary_results = []

for src_l, tgt_l in directions:
    print(f"\n--- Evaluating {src_l} -> {tgt_l} (Train Split) ---")
    
    # Use the train split as requested
    ds = load_dataset(DATASET_NAME, "train", split="eng_hin")
    if MAX_EVAL_SAMPLES:
        ds = ds.shuffle(seed=42).select(range(min(len(ds), MAX_EVAL_SAMPLES)))

    # Handle column mapping based on direction
    src_col, tgt_col = ("src_txt", "tgt_txt") if src_l == "ENGLISH" else ("tgt_txt", "src_txt")
    
    results = []
    for i in tqdm(range(len(ds))):
        source_text = ds[i][src_col]
        reference_text = ds[i][tgt_col]
        
        prompt = make_strict_prompt(source_text, src_l, tgt_l)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        pred_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        # Scoring
        lid = get_lid_score(pred_text, LANG_MAP[tgt_l])
        copy_rate = get_semantic_copy_rate(source_text, pred_text)
        
        results.append({
            "source": source_text,
            "reference": reference_text,
            "prediction": pred_text,
            "lid_accuracy": lid,
            "semantic_copy_rate": copy_rate
        })

    # Metrics Calculation
    df = pd.DataFrame(results)
    bleu = BLEU().corpus_score(df["prediction"].tolist(), [df["reference"].tolist()]).score
    chrf = CHRF().corpus_score(df["prediction"].tolist(), [df["reference"].tolist()]).score
    
    avg_lid = df["lid_accuracy"].mean()
    avg_copy = df["semantic_copy_rate"].mean()

    # Save Directional CSV
    tag = f"{src_l.lower()}_{tgt_l.lower()}"
    df.to_csv(OUT_DIR / f"analysis_{tag}.csv", index=False)
    
    summary_results.append({
        "direction": f"{src_l}->{tgt_l}",
        "BLEU": bleu,
        "ChrF": chrf,
        "LID_Accuracy": avg_lid,
        "Copy_Rate": avg_copy
    })

# ============================================================
# FINAL SUMMARY
# ============================================================
final_df = pd.DataFrame(summary_results)
final_df.to_csv(OUT_DIR / "final_summary_metrics.csv", index=False)

print("\n" + "="*50)
print("FINAL EVALUATION RESULTS (TRAIN SPLIT)")
print("="*50)
print(final_df.to_string(index=False))
print("="*50)
print(f"Results saved to: {OUT_DIR}")
