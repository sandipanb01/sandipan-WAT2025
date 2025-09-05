#!/usr/bin/env python
"""
Modified for Gemma 3-1B models with document chunking and sentence-level translation
FOR ALL LANGUAGES - FOR ALL PAIRS - INFERENCE CODE
"""

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from pathlib import Path
from typing import List, Tuple
import re

from datasets import load_dataset
from vllm import LLM, SamplingParams

import os, re, subprocess, sys

# Add authentication at the very beginning
def setup_huggingface_auth():
    """Set up HuggingFace authentication"""
    # Try to get token from environment variable
    hf_token = os.environ.get("HF_TOKEN", "")

    if not hf_token:
        # Try to get from Colab secrets
        try:
            from google.colab import userdata
            hf_token = userdata.get('HF_TOKEN')
        except:
            pass

    if hf_token:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print("✓ HuggingFace token set from environment")
    else:
        print("⚠ No HF_TOKEN found. You may need to run: huggingface-cli login")

setup_huggingface_auth()

def _fix_cuda_visible_devices():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd.startswith("GPU-"):
        # Already numeric -> nothing to do
        return

    # 1) Build a map {UUID -> index} from `nvidia-smi -L`
    try:
        smi = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except FileNotFoundError:
        sys.exit("!! nvidia-smi not found—cannot map GPU UUIDs to indices")

    uuid2idx = {}
    for line in smi.splitlines():
        # Example: "GPU 2: NVIDIA H100 (UUID: GPU-b1b8e0d1-5d40-8a62-d5da-393bca3fd881)"
        m = re.match(r"GPU\s+(\d+):.*\(UUID:\s+(GPU-[0-9a-f\-]+)\)", line)
        if m:
            idx, uuid = m.groups()
            uuid2idx[uuid] = idx

    # 2) Translate the job-allocated UUID list to indices
    try:
        new_ids = ",".join(uuid2idx[uuid] for uuid in cvd.split(","))
    except KeyError as e:
        missing = str(e).strip("'")
        sys.exit(f"!! UUID {missing} not found in `nvidia-smi -L` output.\n"
                 f"   CUDA_VISIBLE_DEVICES was: {cvd}")

    os.environ["CUDA_VISIBLE_DEVICES"] = new_ids
    print(f"[fix-gpu] CUDA_VISIBLE_DEVICES  {cvd}  →  {new_ids}")

_fix_cuda_visible_devices()

# ---------------------------------------------------------------------------
# 1. DATA LOADING (UNCHANGED)
# ---------------------------------------------------------------------------
def load_pralekha_split(
    src_lang: str = "eng",
    tgt_lang: str = "hin",
    subset: str = "dev",
    max_rows: int | None = 100,
):
    ds = load_dataset("ai4bharat/Pralekha", f"{subset}", split=f"{src_lang}_{tgt_lang}")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    return ds

# ---------------------------------------------------------------------------
# 2. TEXT PROCESSING FUNCTIONS
# ---------------------------------------------------------------------------
def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using simple regex"""
    # Basic sentence splitting - can be enhanced with NLTK if needed
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
    return [s.strip() for s in sentences if s.strip()]

def chunk_document(text: str, max_chars: int = 2000) -> List[str]:
    """Split document into manageable chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(word)
        current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# ---------------------------------------------------------------------------
# 3. PROMPT TEMPLATING (Gemma 3 format for sentence-level translation)
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a professional translator. Translate the following text precisely "
    "from {src} to {tgt}, preserving meaning, tone, and formatting. "
    "Return only the translation without any additional text."
)

def make_sentence_prompts(sentences: List[str], src: str, tgt: str) -> List[str]:
    """Create prompts for individual sentence translation"""
    sys_msg = _SYSTEM.format(src=src, tgt=tgt)
    return [
        f"<start_of_turn>user\n{sys_msg}\n\nText to translate: {s}<end_of_turn>\n<start_of_turn>model\n"
        for s in sentences
    ]

def make_chunk_prompts(chunks: List[str], src: str, tgt: str) -> List[str]:
    """Create prompts for document chunk translation"""
    sys_msg = _SYSTEM.format(src=src, tgt=tgt)
    return [
        f"<start_of_turn>user\n{sys_msg}\n\nText to translate: {chunk}<end_of_turn>\n<start_of_turn>model\n"
        for chunk in chunks
    ]

# ---------------------------------------------------------------------------
# 4. MODEL INSTANTIATION (Gemma 3-1B with auth handling)
# ---------------------------------------------------------------------------
def init_gemma(model_type: str = "it") -> LLM:
    """
    Load Gemma 3-1B model with authentication
    model_type: "pt" for pretrained, "it" for instruction-tuned
    """
    checkpoint = f"google/gemma-3-1b-{model_type}" if model_type == "it" else "google/gemma-3-1b"

    # Check if we have authentication
    if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("⚠ Warning: No HuggingFace token found. You may need to:")
        print("  1. Run: huggingface-cli login")
        print("  2. Or set HF_TOKEN environment variable")
        print("  3. Accept model terms at: https://huggingface.co/google/gemma-3-1b-it")

    try:
        return LLM(
            model=checkpoint,
            dtype="float16", # Changed from bfloat16 to float16
            tokenizer=checkpoint,
            max_model_len=4096,
            gpu_memory_utilization=0.6, # Reduced gpu_memory_utilization
            trust_remote_code=True,
        )
    except Exception as e:
        if "gated" in str(e).lower() or "401" in str(e):
            print("\n❌ Authentication failed! You need to:")
            print("  1. Accept the model terms at: https://huggingface.co/google/gemma-3-1b-it")
            print("  2. Get a token from: https://huggingface.co/settings/tokens")
            print("  3. Either:")
            print("   - Run: huggingface-cli login")
            print("   - Or set HF_TOKEN environment variable")
        raise e

# ---------------------------------------------------------------------------
# 5. BATCH TRANSLATION
# ---------------------------------------------------------------------------
def translate(
    llm: LLM,
    prompts: List[str],
    temperature: float = 0.0,
    max_tokens: int = 512,  # Reduced for sentence-level translation
) -> List[str]:
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    return [out.outputs[0].text.strip() for out in outputs]

# ---------------------------------------------------------------------------
# 6. DOCUMENT TRANSLATION PIPELINE
# ---------------------------------------------------------------------------
def translate_document(
    llm: LLM,
    document: str,
    src_lang: str,
    tgt_lang: str,
    strategy: str = "sentence"  # "sentence" or "chunk"
) -> str:
    """
    Translate a document using either sentence-level or chunk-level approach
    """
    if strategy == "sentence":
        # Split into sentences and translate individually
        sentences = split_into_sentences(document)
        print(f"  Split into {len(sentences)} sentences")

        prompts = make_sentence_prompts(sentences, src_lang, tgt_lang)
        translations = translate(llm, prompts, max_tokens=256)

        # Reconstruct document
        return " ".join(translations)

    else:  # chunk strategy
        # Split into manageable chunks
        chunks = chunk_document(document, max_chars=1500)
        print(f"  Split into {len(chunks)} chunks")

        prompts = make_chunk_prompts(chunks, src_lang, tgt_lang)
        translations = translate(llm, prompts, max_tokens=1024)

        # Reconstruct document
        return " ".join(translations)

# ---------------------------------------------------------------------------
# 7. LANGUAGE CONFIGURATION - ALL SUPPORTED PAIRS
# ---------------------------------------------------------------------------
# All language pairs to process
LANGUAGE_PAIRS = [
    # English to Indic languages
    ("eng", "ben"),  # English to Bengali
    ("eng", "guj"),  # English to Gujarati
    ("eng", "hin"),  # English to Hindi
    ("eng", "kan"),  # English to Kannada
    ("eng", "mal"),  # English to Malayalam
    ("eng", "mar"),  # English to Marathi
    ("eng", "ori"),  # English to Odia
    ("eng", "pan"),  # English to Punjabi
    ("eng", "tam"),  # English to Tamil
    ("eng", "tel"),  # English to Telugu
    ("eng", "urd"),  # English to Urdu
    
    # Indic languages to English (reverse direction)
    ("ben", "eng"),  # Bengali to English
    ("guj", "eng"),  # Gujarati to English
    ("hin", "eng"),  # Hindi to English
    ("kan", "eng"),  # Kannada to English
    ("mal", "eng"),  # Malayalam to English
    ("mar", "eng"),  # Marathi to English
    ("ori", "eng"),  # Odia to English
    ("pan", "eng"),  # Punjabi to English
    ("tam", "eng"),  # Tamil to English
    ("tel", "eng"),  # Telugu to English
    ("urd", "eng"),  # Urdu to English
]

# Language code to name mapping for better prompts
LANG_NAMES = {
    "eng": "English",
    "ben": "Bengali",
    "guj": "Gujarati",
    "hin": "Hindi",
    "kan": "Kannada",
    "mal": "Malayalam",
    "mar": "Marathi",
    "ori": "Odia",
    "pan": "Punjabi",
    "tam": "Tamil",
    "tel": "Telugu",
    "urd": "Urdu",
}

# ---------------------------------------------------------------------------
# 8. PROCESS ALL LANGUAGES
# ---------------------------------------------------------------------------
def process_all_languages():
    """Process translation for all language pairs"""
    SUBSET = "dev"
    N = 10  # Reduced for testing - increase after it works
    MODEL_TYPE = "it"
    STRATEGY = "sentence"  # "sentence" or "chunk"

    print("Initializing Gemma model...")
    try:
        llm = init_gemma(MODEL_TYPE)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Process each language pair
    for src_lang, tgt_lang in LANGUAGE_PAIRS:
        print(f"\n{'='*60}")
        print(f"Processing: {LANG_NAMES[src_lang]} → {LANG_NAMES[tgt_lang]}")
        print(f"{'='*60}")

        try:
            # Load dataset for this language pair
            ds = load_pralekha_split(src_lang, tgt_lang, SUBSET, N)
            print(f"Loaded {len(ds):,} rows from {SUBSET}/{src_lang}_{tgt_lang}")

            if len(ds) == 0:
                print("⚠ No documents found, skipping...")
                continue

            translations = []
            for i, doc in enumerate(ds["src_txt"]):
                print(f"Translating document {i+1}/{len(ds['src_txt'])}...")
                translated_doc = translate_document(llm, doc, LANG_NAMES[src_lang], LANG_NAMES[tgt_lang], STRATEGY)
                translations.append(translated_doc)
                print(f"  ✓ Completed document {i+1}")

            # Add predictions & persist
            ds = ds.add_column("pred_txt", translations)
            out_file = Path(f"gemma3_1b_{MODEL_TYPE}_{STRATEGY}_{src_lang}_to_{tgt_lang}_{SUBSET}_{N}.csv")
            ds.to_pandas().to_csv(out_file, index=False)
            print(f"✓ Saved translations to {out_file.resolve()}")

        except Exception as e:
            print(f"❌ Error processing {src_lang}→{tgt_lang}: {e}")
            print("Skipping to next language pair...")
            continue

    print(f"\n{'='*60}")
    print("✓ Translation completed for ALL language pairs!")
    print(f"{'='*60}")

# ---------------------------------------------------------------------------
# 9. MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    process_all_languages()
