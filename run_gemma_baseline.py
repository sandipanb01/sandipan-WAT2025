"""
run_gemma4_mathdoc.py
=====================
End-to-end HuggingFace inference + full evaluation for MathDoc-ENHI.

Model  : google/gemma-4-31b-it  (local HuggingFace weights — NO API calls)
Task   : English → Hindi mathematical textbook translation
Metrics:
    Standard  — BLEU (flores101), chrF++  [sacrebleu / IBM-origin stack]
    SFS Suite — MSP, MCP, TSP, TCP        [structure-fidelity, no ML required]
    Optional  — COMET-22                  [Unbabel; needs ~2 GB download + GPU]

Masking: LaTeX equations ($, $$, \\begin{}, <math>) and HTML/MD tables are
         replaced with [EQ_k] / [TB_k] placeholders before translation and
         restored verbatim afterwards — the same strategy as run_sarvam.py /
         run_gpt4o.py in the collaborator pipeline.

Features:
    - CONFIG dict (no argparse) — tweak everything at the top
    - dtype toggle  : bfloat16 / float16 / float32
    - quant toggle  : 4-bit / 8-bit / none  (bitsandbytes; use on Colab A100)
    - SANITY_CHECK  : run on first N instances only, print side-by-side diffs
    - Batch size 4, greedy decoding (do_sample=False, no beam search)
    - Category breakdown: pure_text / math_only / math_tables
    - Outputs compatible with the team's evaluate.py (JSONL + JSON results)

Usage:
    python run_gemma4_mathdoc.py

Requirements (install once):
    pip install torch transformers accelerate sacrebleu tqdm
    pip install bitsandbytes          # for 4-bit / 8-bit quantisation
    pip install unbabel-comet         # only if SKIP_COMET = False

Author: advisor-grade research script — auto-generated
"""

# =============================================================================
# SECTION 0 — Imports
# =============================================================================

import gc
import json
import math
import os
import re
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# SECTION 1 — CONFIG  (edit here; no command-line args needed)
# =============================================================================

CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────────
    "DATA_PATHS": [
        "ncert_class11_math_en_hi_test_instances_curated_FIXED.json",
        "ncert_class12_math_en_hi_test_instances_curated.json",
    ],
    "OUTPUT_DIR": "outputs/gemma4",
    "HYPOTHESIS_JSONL": "outputs/gemma4/gemma4_hypotheses.jsonl",
    "RESULTS_JSON":     "outputs/gemma4/gemma4_results.json",

    # ── Model ──────────────────────────────────────────────────────────────────
    "MODEL_ID": "google/gemma-4-31b-it",

    # ── Compute dtype: "bfloat16" | "float16" | "float32"
    # bfloat16 → recommended for Ampere+ GPUs (A100, A6000, H100)
    # float16  → safe fallback for older GPUs (V100, T4)
    # float32  → CPU-only or debugging
    "DTYPE": "bfloat16",

    # ── Quantisation: "4bit" | "8bit" | "none"
    # 4bit ≈ 16 GB VRAM (fits on Colab A100 40 GB or 2× consumer 24 GB)
    # 8bit ≈ 32 GB VRAM
    # none → full precision (requires ≥ 64 GB VRAM for 31B bfloat16)
    "QUANTISATION": "4bit",

    # ── Inference ──────────────────────────────────────────────────────────────
    "BATCH_SIZE":        4,         # reduce to 2 if you hit OOM
    "MAX_NEW_TOKENS":    3072,      # generous budget for long math passages
    "DO_SAMPLE":         False,     # greedy decoding — no beam search
    "TEMPERATURE":       1.0,       # only used if DO_SAMPLE=True
    "TOP_P":             0.95,
    "TOP_K":             64,

    # ── Sanity check ───────────────────────────────────────────────────────────
    # Set to True to run only on SANITY_N instances (print src/ref/hyp diffs)
    "SANITY_CHECK":      False,
    "SANITY_N":          5,

    # ── Metrics ────────────────────────────────────────────────────────────────
    "SKIP_COMET":        True,      # COMET-22 downloads ~2 GB; set False to use
    "COMET_MODEL":       "Unbabel/wmt22-comet-da",
    "COMET_BATCH_SIZE":  8,
    "COMET_GPUS":        1,
    "SACREBLEU_TOKENIZER": "flores101",  # handles Devanagari script correctly

    # ── Misc ───────────────────────────────────────────────────────────────────
    "SEED":              42,
    "SLEEP_BETWEEN_BATCHES": 0.0,   # seconds; 0 = no throttle
    "MAX_RETRIES":       3,
    "EXPECTED_TOTAL":    821,       # 393 class11 + 428 class12
}

# =============================================================================
# SECTION 2 — Logging
# =============================================================================

def log(msg: str, level: str = "INFO") -> None:
    prefix = {"INFO": "ℹ", "WARN": "⚠", "ERROR": "✗", "OK": "✓", "HEAD": "═"}
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {prefix.get(level, '·')} {msg}", flush=True)


def section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)

# =============================================================================
# SECTION 3 — GPU utilities
# =============================================================================

def free_gpu() -> None:
    """Release cached GPU memory between operations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_device_info() -> str:
    if not torch.cuda.is_available():
        return "CPU only"
    gpus = [
        f"{torch.cuda.get_device_name(i)} "
        f"({torch.cuda.get_device_properties(i).total_memory // (1024**3)} GB)"
        for i in range(torch.cuda.device_count())
    ]
    return "  |  ".join(gpus)


def resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose: {list(mapping)}")
    return mapping[dtype_str]

# =============================================================================
# SECTION 4 — Data loading & validation  (inlined from data_io.py)
# =============================================================================

class DataValidationError(ValueError):
    """Raised when dataset or hypothesis file is malformed."""


def _coerce_record_list(obj, path: str) -> List[Dict]:
    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict) and isinstance(obj.get("instances"), list):
        records = obj["instances"]
    elif isinstance(obj, dict):
        records = [obj]
    else:
        raise DataValidationError(
            f"{path}: expected JSON array, object, or JSONL — got {type(obj)}")
    bad = [i for i, r in enumerate(records) if not isinstance(r, dict)]
    if bad:
        raise DataValidationError(
            f"{path}: records must be objects; first bad index = {bad[0]}")
    return records


def _load_one_file(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise DataValidationError(f"{path}: file is empty")
    try:
        return _coerce_record_list(json.loads(text), path)
    except json.JSONDecodeError as whole_err:
        records = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as le:
                raise DataValidationError(
                    f"{path}:{lineno}: not valid JSON/JSONL "
                    f"(whole-file err: {whole_err}; line err: {le})"
                ) from le
            if not isinstance(obj, dict):
                raise DataValidationError(
                    f"{path}:{lineno}: JSONL records must be objects")
            records.append(obj)
        if not records:
            raise DataValidationError(f"{path}: no records found")
        return records


def load_dataset_records(paths, expected_total: Optional[int] = None) -> List[Dict]:
    """Load + validate MathDoc-ENHI dataset records."""
    required = ("id", "content_en", "content_hi")
    all_records: List[Dict] = []

    for p in ([paths] if isinstance(paths, (str, Path)) else paths):
        path_str = str(p)
        if not os.path.exists(path_str):
            raise DataValidationError(f"Dataset file not found: {path_str}")
        batch = _load_one_file(path_str)
        log(f"Loaded {len(batch):4d} records from {path_str}")
        all_records.extend(batch)

    # Field check
    for idx, rec in enumerate(all_records):
        missing = [f for f in required if f not in rec]
        if missing:
            raise DataValidationError(
                f"Record {idx} missing fields: {missing}")
        empty = [f for f in required if not str(rec.get(f, "")).strip()]
        if empty:
            raise DataValidationError(
                f"Record {rec.get('id', idx)!r} has empty fields: {empty}")

    # Duplicate id check
    seen, dupes = set(), []
    for rec in all_records:
        rid = str(rec["id"])
        if rid in seen:
            dupes.append(rid)
        seen.add(rid)
    if dupes:
        raise DataValidationError(
            f"Duplicate id(s): {dupes[:10]} ...")

    if expected_total is not None and len(all_records) != expected_total:
        raise DataValidationError(
            f"Expected {expected_total} records, found {len(all_records)}")

    return all_records

# =============================================================================
# SECTION 5 — Masking  (extended from run_gemma.py to include <math> tags)
# =============================================================================

# Order is critical: more specific patterns first so they are not
# subsumed by more general ones (e.g., $$ before $).
_MASK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Display LaTeX: $$...$$
    (re.compile(r'\$\$[^$]+?\$\$', re.DOTALL), "EQ"),
    # Named environments: \begin{equation}…\end{equation}, align, matrix, etc.
    (re.compile(
        r'\\begin\{(equation|align|gather|eqnarray|'
        r'bmatrix|pmatrix|vmatrix|matrix|smallmatrix)\*?\}'
        r'.*?'
        r'\\end\{\1\*?\}',
        re.DOTALL), "EQ"),
    # MathML tags: <math>...</math>  (present in 33 records of this dataset)
    (re.compile(r'<math[^>]*>.*?</math>', re.DOTALL | re.IGNORECASE), "EQ"),
    # Inline LaTeX: $...$  (no newlines inside; matched LAST to avoid $$)
    (re.compile(r'\$[^$\n]+?\$'), "EQ"),
    # HTML tables: <table>...</table>
    (re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE), "TB"),
    # Markdown pipe tables (2+ rows of |...|...|)
    (re.compile(r'(?:^|\n)((?:\|[^\n]+\|[\t ]*\n){2,})', re.MULTILINE), "TB"),
]


def mask_spans(text: str) -> Tuple[str, List[str]]:
    """
    Replace math and table spans with positional placeholders.
    Returns (masked_text, ordered_list_of_original_spans).
    """
    spans: List[str] = []
    masked = text
    for pat, prefix in _MASK_PATTERNS:
        def _repl(m, prefix=prefix):
            original = m.group(0)
            idx = len(spans)
            spans.append(original)
            return f"[{prefix}_{idx}]"
        masked = pat.sub(_repl, masked)
    return masked, spans


def unmask_spans(text: str, spans: List[str]) -> str:
    """Restore masked spans from the placeholder list."""
    result = text
    # Reverse order to avoid index collision on short strings
    for i in range(len(spans) - 1, -1, -1):
        for prefix in ("EQ", "TB"):
            placeholder = f"[{prefix}_{i}]"
            if placeholder in result:
                result = result.replace(placeholder, spans[i])
    return result

# =============================================================================
# SECTION 6 — Structure Fidelity Suite  (inlined from structure_fidelity.py)
# =============================================================================

# ── Equation extraction ──────────────────────────────────────────────────────

_EQ_PATTERNS_SFS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\$\$([^$]+?)\$\$', re.DOTALL), "display_dollar"),
    (re.compile(r'\$([^$\n]+?)\$'), "inline_dollar"),
    (re.compile(
        r'\\begin\{(equation|align|gather|eqnarray|'
        r'bmatrix|pmatrix|vmatrix|matrix|smallmatrix)\*?\}'
        r'(.+?)'
        r'\\end\{\1\*?\}',
        re.DOTALL), "environment"),
    (re.compile(r'<math[^>]*>(.*?)</math>', re.DOTALL | re.IGNORECASE), "mathml"),
]


def extract_equations(text: str) -> List[str]:
    occupied: List[Tuple[int, int]] = []
    spans: List[Tuple[int, str]] = []
    for pat, kind in _EQ_PATTERNS_SFS:
        for m in pat.finditer(text):
            s, e = m.span()
            if any(s < oe and e > os for os, oe in occupied):
                continue
            if kind == "environment":
                content = m.group(2)
            elif kind == "mathml":
                content = m.group(1)
            else:
                content = m.group(1)
            spans.append((s, content.strip()))
            occupied.append((s, e))
    return [c for _, c in sorted(spans)]


def count_equations(text: str) -> int:
    return len(extract_equations(text))


def normalise_latex(s: str) -> str:
    return re.sub(r"\s+", "", s).strip()


# ── MSP — Math Structure Preservation ────────────────────────────────────────

def metric_msp(ref: str, hyp: str) -> Optional[float]:
    """min(1, |EQ(hyp)| / |EQ(ref)|);  None if ref has no equations."""
    n_ref = count_equations(ref)
    if n_ref == 0:
        return None
    return min(1.0, count_equations(hyp) / n_ref)


# ── MCP — Math Content Preservation ──────────────────────────────────────────

def metric_mcp(ref: str, hyp: str) -> Optional[float]:
    """Fraction of ref equations that appear verbatim in hyp."""
    ref_eqs = extract_equations(ref)
    if not ref_eqs:
        return None
    hyp_norm = normalise_latex(hyp)
    hits = sum(1 for eq in ref_eqs if normalise_latex(eq) in hyp_norm)
    return hits / len(ref_eqs)


# ── Table extraction ──────────────────────────────────────────────────────────

_HTML_TABLE_RE = re.compile(
    r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
_HTML_ROW_RE   = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
_HTML_CELL_RE  = re.compile(
    r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
_HTML_TAG_RE   = re.compile(r'<[^>]+>')
_MD_TABLE_RE   = re.compile(
    r'(?:^|\n)((?:\|[^\n]+\|[\t ]*\n){2,})', re.MULTILINE)


def _parse_html_table(inner_html: str) -> Dict:
    rows: List[List[str]] = []
    for row_html in _HTML_ROW_RE.findall(inner_html):
        cells = [_HTML_TAG_RE.sub("", c).strip()
                 for c in _HTML_CELL_RE.findall(row_html)]
        rows.append(cells)
    if not rows:
        return {"rows": 0, "cols": 0, "cells": [], "format": "html"}
    cols = max(len(r) for r in rows)
    return {
        "rows": len(rows),
        "cols": cols,
        "cells": [c for r in rows for c in r],
        "format": "html",
    }


def _parse_md_table(block: str) -> Optional[Dict]:
    lines = [ln for ln in block.strip().split("\n") if ln.strip()]
    if len(lines) < 2:
        return None

    def cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]

    rows = [cells(lines[0])]
    start = 1
    if start < len(lines) and re.match(r'^[\s|:\-]+$', lines[start]):
        start += 1
    rows.extend(cells(ln) for ln in lines[start:])
    if not rows:
        return None
    return {
        "rows": len(rows),
        "cols": max(len(r) for r in rows),
        "cells": [c for r in rows for c in r],
        "format": "md",
    }


def extract_tables(text: str) -> List[Dict]:
    tables = [_parse_html_table(m.group(1)) for m in _HTML_TABLE_RE.finditer(text)]
    for m in _MD_TABLE_RE.finditer(text):
        parsed = _parse_md_table(m.group(1))
        if parsed:
            tables.append(parsed)
    return tables


# ── TSP — Table Structure Preservation ───────────────────────────────────────

def _shape_score(t_ref: Dict, t_hyp: Dict) -> float:
    if t_ref["rows"] == 0 or t_ref["cols"] == 0:
        return 0.0
    row_r = min(t_ref["rows"], t_hyp["rows"]) / max(t_ref["rows"], t_hyp["rows"], 1)
    col_r = min(t_ref["cols"], t_hyp["cols"]) / max(t_ref["cols"], t_hyp["cols"], 1)
    return (row_r + col_r) / 2.0


def metric_tsp(ref: str, hyp: str) -> Optional[float]:
    ref_tables = extract_tables(ref)
    if not ref_tables:
        return None
    hyp_tables = extract_tables(hyp)
    scores = [
        _shape_score(t_ref, hyp_tables[i]) if i < len(hyp_tables) else 0.0
        for i, t_ref in enumerate(ref_tables)
    ]
    return sum(scores) / len(scores)


# ── TCP — Table Content Preservation ─────────────────────────────────────────

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _numeric_cells(table: Dict) -> List[str]:
    return [n for cell in table["cells"] for n in _NUMBER_RE.findall(cell)]


def metric_tcp(ref: str, hyp: str) -> Optional[float]:
    ref_tables = extract_tables(ref)
    if not ref_tables:
        return None
    ref_nums = [n for t in ref_tables for n in _numeric_cells(t)]
    if not ref_nums:
        return None
    hyp_tables = extract_tables(hyp)
    hyp_nums = [n for t in hyp_tables for n in _numeric_cells(t)]
    ref_c, hyp_c = Counter(ref_nums), Counter(hyp_nums)
    preserved = sum(min(ref_c[n], hyp_c.get(n, 0)) for n in ref_c)
    return preserved / sum(ref_c.values())


# ── Aggregate SFS ─────────────────────────────────────────────────────────────

def compute_sfs(refs: List[str], hyps: List[str]) -> Dict:
    """Compute MSP, MCP, TSP, TCP over aligned ref/hyp lists."""
    assert len(refs) == len(hyps), "Length mismatch in compute_sfs"
    buckets: Dict[str, List[float]] = {
        "MSP": [], "MCP": [], "TSP": [], "TCP": []
    }
    for ref, hyp in zip(refs, hyps):
        for key, fn in [
            ("MSP", metric_msp), ("MCP", metric_mcp),
            ("TSP", metric_tsp), ("TCP", metric_tcp)
        ]:
            v = fn(ref, hyp)
            if v is not None:
                buckets[key].append(v)

    def _mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "MSP":      _mean(buckets["MSP"]),
        "MCP":      _mean(buckets["MCP"]),
        "TSP":      _mean(buckets["TSP"]),
        "TCP":      _mean(buckets["TCP"]),
        "n_math":   len(buckets["MSP"]),
        "n_tables": len(buckets["TSP"]),
        "n_tcp":    len(buckets["TCP"]),
    }

# =============================================================================
# SECTION 7 — Standard metrics (BLEU, chrF++, COMET-22)
# =============================================================================

def compute_bleu_chrf(refs: List[str], hyps: List[str], cfg: Dict) -> Dict[str, float]:
    """
    Corpus-level BLEU (flores101 tokenizer) and chrF++ (word_order=2).
    The flores101 tokenizer is the correct choice for Devanagari/Hindi text
    and is the one used by the official evaluation pipeline.

    IBM sacreBLEU (https://github.com/mjpost/sacrebleu) — MIT licence.
    """
    import sacrebleu  # lazy import so script loads even without it installed

    tokenizer = cfg.get("SACREBLEU_TOKENIZER", "flores101")
    bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tokenizer)
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)   # ++ = order 2

    return {
        "BLEU":   round(float(bleu.score), 4),
        "chrF++": round(float(chrf.score), 4),
    }


def compute_comet(
    srcs: List[str], hyps: List[str], refs: List[str], cfg: Dict
) -> Dict[str, float]:
    """
    COMET-22 (reference-based, Unbabel/wmt22-comet-da).
    Downloads ~2 GB on first run; requires GPU for reasonable speed.
    """
    from comet import download_model, load_from_checkpoint  # lazy import

    log("Downloading / loading COMET-22 model (~2 GB first run)...", "WARN")
    model_path = download_model(cfg["COMET_MODEL"])
    model = load_from_checkpoint(model_path)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    output = model.predict(
        data,
        batch_size=cfg["COMET_BATCH_SIZE"],
        gpus=cfg["COMET_GPUS"],
    )
    return {"COMET-22": round(float(output.system_score), 6)}


def categorise_instance(content_en: str, content_hi: str) -> str:
    """
    Assign content category based on reference translation content:
      math_tables → has HTML/MD tables (with or without math)
      math_only   → has math, no tables
      pure_text   → neither
    """
    has_math   = count_equations(content_hi) > 0
    has_tables = len(extract_tables(content_hi)) > 0
    if has_tables:
        return "math_tables"
    if has_math:
        return "math_only"
    return "pure_text"

# =============================================================================
# SECTION 8 — Model loading
# =============================================================================

def load_model_and_tokenizer(cfg: Dict):
    """
    Load google/gemma-4-31b-it with the configured dtype and quantisation.

    Quantisation map:
        "4bit" → BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        "8bit" → BitsAndBytesConfig(load_in_8bit=True)
        "none" → no quantisation (full precision in configured dtype)
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    model_id   = cfg["MODEL_ID"]
    quant_mode = cfg["QUANTISATION"].lower()
    dtype      = resolve_dtype(cfg["DTYPE"])

    log(f"Model    : {model_id}")
    log(f"dtype    : {cfg['DTYPE']}")
    log(f"quant    : {quant_mode}")
    log(f"GPU(s)   : {get_device_info()}")

    # ── Quantisation config ──────────────────────────────────────────────────
    bnb_config = None
    if quant_mode == "4bit":
        try:
            import bitsandbytes as bnb  # noqa: F401
        except ImportError:
            raise ImportError(
                "bitsandbytes not found. "
                "Install with: pip install bitsandbytes"
            )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_mode == "8bit":
        try:
            import bitsandbytes as bnb  # noqa: F401
        except ImportError:
            raise ImportError(
                "bitsandbytes not found. "
                "Install with: pip install bitsandbytes"
            )
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # ── Load tokenizer ───────────────────────────────────────────────────────
    log("Loading tokenizer...", "INFO")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # causal-LM: pad on the left

    # ── Load model ───────────────────────────────────────────────────────────
    log("Loading model weights (this may take several minutes)...", "INFO")

    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    log("Model loaded successfully.", "OK")
    return model, tokenizer

# =============================================================================
# SECTION 9 — Prompt building
# =============================================================================

SYSTEM_PROMPT = (
    "You are a professional mathematical translator specialised in Indian NCERT "
    "mathematics textbooks. Translate the following English text into Hindi.\n\n"
    "CRITICAL RULES:\n"
    "1. You will see placeholders like [EQ_0], [EQ_1], [TB_0], [TB_1], etc.\n"
    "2. Do NOT translate, modify, or remove these placeholders.\n"
    "3. Keep every placeholder exactly as written, in its correct relative position.\n"
    "4. Translate only the surrounding natural-language text.\n"
    "5. Output ONLY the translated Hindi text — no preamble, no explanations."
)


def build_chat_prompt(masked_src: str, tokenizer) -> str:
    """
    Build a Gemma-4 instruction-tuned chat prompt for one instance.
    Uses tokenizer.apply_chat_template when available.
    """
    messages = [
        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\nEnglish:\n{masked_src}"},
    ]
    # apply_chat_template adds the model's special tokens (BOS, turn markers, etc.)
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # Fallback for tokenizers that don't support chat templates
        prompt = (
            f"<start_of_turn>user\n"
            f"{SYSTEM_PROMPT}\n\nEnglish:\n{masked_src}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
    return prompt

# =============================================================================
# SECTION 10 — Batched inference
# =============================================================================

def run_inference_on_batch(
    model,
    tokenizer,
    prompts: List[str],
    cfg: Dict,
) -> List[str]:
    """
    Tokenize, pad, and generate translations for one batch of prompts.
    Returns raw decoded strings (no post-processing yet).
    """
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,          # gemma-4 context window
        padding_side="left",      # align for causal generation
    )

    input_ids      = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    gen_kwargs = {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": cfg["MAX_NEW_TOKENS"],
        "do_sample":      cfg["DO_SAMPLE"],
        "use_cache":      True,
        "pad_token_id":   tokenizer.pad_token_id,
        "eos_token_id":   tokenizer.eos_token_id,
    }
    # Greedy decoding — no beam search, no sampling
    # (temperature / top_p only apply when do_sample=True)
    if cfg["DO_SAMPLE"]:
        gen_kwargs["temperature"] = cfg["TEMPERATURE"]
        gen_kwargs["top_p"]       = cfg["TOP_P"]
        gen_kwargs["top_k"]       = cfg["TOP_K"]

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    # Slice off the input tokens — keep only newly generated tokens
    new_tokens = outputs[:, input_ids.shape[1]:]
    decoded = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return decoded


def translate_all_instances(
    model,
    tokenizer,
    instances: List[Dict],
    cfg: Dict,
) -> List[Dict]:
    """
    Translate every instance with masking → inference → unmasking.
    Returns list of output dicts: {id, source, masked_source, hypothesis, error?}
    """
    batch_size = cfg["BATCH_SIZE"]
    n = len(instances)
    outputs: List[Dict] = []

    log(f"Starting inference: {n} instances, batch_size={batch_size}, "
        f"greedy={'yes' if not cfg['DO_SAMPLE'] else 'no'}")

    # Pre-build all prompts (mask each instance)
    prompts:  List[str] = []
    span_maps: List[List[str]] = []   # one span list per instance

    log("Pre-building prompts and masking equations/tables...")
    for inst in instances:
        masked, spans = mask_spans(inst["content_en"])
        prompt = build_chat_prompt(masked, tokenizer)
        prompts.append(prompt)
        span_maps.append(spans)

    log(f"All prompts built. Starting batched generation...", "OK")
    free_gpu()

    for batch_start in tqdm(
        range(0, n, batch_size),
        desc="Translating",
        unit="batch",
        dynamic_ncols=True,
    ):
        batch_end   = min(batch_start + batch_size, n)
        batch_idx   = list(range(batch_start, batch_end))
        batch_prompts = [prompts[i] for i in batch_idx]

        hyps_raw: List[str] = []
        last_err = None

        for attempt in range(cfg["MAX_RETRIES"]):
            try:
                hyps_raw = run_inference_on_batch(
                    model, tokenizer, batch_prompts, cfg)
                last_err = None
                break
            except RuntimeError as e:
                last_err = str(e)
                if "out of memory" in last_err.lower():
                    log(f"OOM on batch {batch_start}–{batch_end} "
                        f"(attempt {attempt+1}/{cfg['MAX_RETRIES']}). "
                        "Reduce BATCH_SIZE if this persists.", "WARN")
                    free_gpu()
                    time.sleep(2)
                else:
                    log(f"Runtime error: {last_err}", "ERROR")
                    break

        for local_i, global_i in enumerate(batch_idx):
            inst   = instances[global_i]
            spans  = span_maps[global_i]

            if last_err and local_i >= len(hyps_raw):
                # Failed batch — record error, keep going
                rec = {
                    "id":         inst["id"],
                    "source":     inst["content_en"],
                    "hypothesis": "",
                    "error":      last_err,
                }
            else:
                raw_hyp  = hyps_raw[local_i].strip()
                final_hyp = unmask_spans(raw_hyp, spans)
                rec = {
                    "id":            inst["id"],
                    "source":        inst["content_en"],
                    "hypothesis":    final_hyp,
                }

            outputs.append(rec)

        if cfg["SLEEP_BETWEEN_BATCHES"] > 0:
            time.sleep(cfg["SLEEP_BETWEEN_BATCHES"])

    log(f"Inference complete. {n} outputs generated.", "OK")
    return outputs

# =============================================================================
# SECTION 11 — Evaluation
# =============================================================================

def evaluate_subset(
    ids:    List[str],
    srcs:   Dict[str, str],
    refs:   Dict[str, str],
    hyps:   Dict[str, str],
    cats:   Dict[str, str],
    cfg:    Dict,
    label:  str = "OVERALL",
) -> Dict:
    """Run all metrics on a subset of instance ids."""
    if not ids:
        return {"n": 0}

    ref_list = [refs[i] for i in ids]
    hyp_list = [hyps[i] for i in ids]
    src_list = [srcs[i] for i in ids]

    result: Dict = {"n": len(ids)}

    # ── Standard: BLEU + chrF++ ──────────────────────────────────────────────
    try:
        bleu_chrf = compute_bleu_chrf(ref_list, hyp_list, cfg)
        result.update(bleu_chrf)
        log(f"  [{label}] BLEU={bleu_chrf['BLEU']:.2f}  "
            f"chrF++={bleu_chrf['chrF++']:.2f}")
    except Exception as exc:
        log(f"  [{label}] BLEU/chrF++ error: {exc}", "ERROR")
        result["BLEU"]   = None
        result["chrF++"] = None

    # ── Optional: COMET-22 ───────────────────────────────────────────────────
    if not cfg["SKIP_COMET"]:
        try:
            comet = compute_comet(src_list, hyp_list, ref_list, cfg)
            result.update(comet)
            log(f"  [{label}] COMET-22={comet['COMET-22']:.4f}")
        except Exception as exc:
            log(f"  [{label}] COMET-22 error: {exc}", "ERROR")
            result["COMET-22"] = None

    # ── SFS: MSP, MCP, TSP, TCP ──────────────────────────────────────────────
    try:
        sfs = compute_sfs(ref_list, hyp_list)
        result.update({
            "MSP": round(sfs["MSP"] * 100, 4),
            "MCP": round(sfs["MCP"] * 100, 4),
            "TSP": round(sfs["TSP"] * 100, 4),
            "TCP": round(sfs["TCP"] * 100, 4),
            "n_math":   sfs["n_math"],
            "n_tables": sfs["n_tables"],
            "n_tcp":    sfs["n_tcp"],
        })
        log(f"  [{label}] MSP={result['MSP']:.2f}  MCP={result['MCP']:.2f}  "
            f"TSP={result['TSP']:.2f}  TCP={result['TCP']:.2f}")
    except Exception as exc:
        log(f"  [{label}] SFS error: {exc}", "ERROR")

    return result


def run_full_evaluation(
    instances: List[Dict],
    outputs:   List[Dict],
    cfg:       Dict,
) -> Dict:
    """
    Align dataset + hypothesis outputs, run all metrics, return results dict.
    """
    section("EVALUATION PHASE")

    # Build lookup dicts
    srcs: Dict[str, str] = {str(inst["id"]): inst["content_en"]
                             for inst in instances}
    refs: Dict[str, str] = {str(inst["id"]): inst["content_hi"]
                             for inst in instances}
    hyps: Dict[str, str] = {str(out["id"]): out.get("hypothesis", "")
                             for out in outputs}
    cats: Dict[str, str] = {
        str(inst["id"]): categorise_instance(inst["content_en"], inst["content_hi"])
        for inst in instances
    }

    # Coverage check
    all_ids   = [str(inst["id"]) for inst in instances]
    missing   = [i for i in all_ids if not hyps.get(i)]
    if missing:
        log(f"{len(missing)} instance(s) have empty/missing hypothesis — "
            f"they will score 0 on all metrics.", "WARN")
        for mid in missing:
            hyps[mid] = ""

    results = {"model": cfg["MODEL_ID"], "dtype": cfg["DTYPE"],
                "quantisation": cfg["QUANTISATION"]}

    # ── Overall ──────────────────────────────────────────────────────────────
    log("Running OVERALL metrics...")
    results["overall"] = evaluate_subset(
        all_ids, srcs, refs, hyps, cats, cfg, "OVERALL")

    # ── Per category ─────────────────────────────────────────────────────────
    for cat_label in ("pure_text", "math_only", "math_tables"):
        subset = [i for i in all_ids if cats[i] == cat_label]
        log(f"Running {cat_label} subset (n={len(subset)})...")
        results[cat_label] = evaluate_subset(
            subset, srcs, refs, hyps, cats, cfg, cat_label.upper())

    return results

# =============================================================================
# SECTION 12 — Output saving
# =============================================================================

def save_hypotheses(outputs: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in outputs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    log(f"Hypotheses saved → {path}  ({len(outputs)} records)", "OK")


def save_results(results: Dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"Results saved    → {path}", "OK")


def print_results_table(results: Dict) -> None:
    """Pretty-print the final results in a readable table."""
    section("FINAL RESULTS")
    model_name = results.get("model", "unknown")
    print(f"  Model: {model_name}")
    print(f"  dtype: {results.get('dtype')}  |  quant: {results.get('quantisation')}")
    print()

    header = (
        f"{'Category':<14}  {'n':>5}  "
        f"{'BLEU':>7}  {'chrF++':>7}  "
        f"{'MSP':>7}  {'MCP':>7}  {'TSP':>7}  {'TCP':>7}"
    )
    if not results.get("overall", {}).get("COMET-22") is None:
        header += f"  {'COMET':>8}"
    print(header)
    print("-" * len(header))

    cats = ["overall", "pure_text", "math_only", "math_tables"]
    for cat in cats:
        r = results.get(cat)
        if not r or r.get("n", 0) == 0:
            continue
        row = (
            f"{cat:<14}  {r['n']:>5}  "
            f"{r.get('BLEU', 0) or 0:>7.2f}  "
            f"{r.get('chrF++', 0) or 0:>7.2f}  "
            f"{r.get('MSP', 0) or 0:>7.2f}  "
            f"{r.get('MCP', 0) or 0:>7.2f}  "
            f"{r.get('TSP', 0) or 0:>7.2f}  "
            f"{r.get('TCP', 0) or 0:>7.2f}"
        )
        if r.get("COMET-22") is not None:
            row += f"  {r['COMET-22']:>8.4f}"
        print(row)

    print("-" * len(header))
    print()

# =============================================================================
# SECTION 13 — Sanity check mode
# =============================================================================

def run_sanity_check(model, tokenizer, instances: List[Dict], cfg: Dict) -> None:
    """
    Translate SANITY_N instances, print a side-by-side diff of
    source / reference / hypothesis for quick visual inspection.
    """
    section(f"SANITY CHECK MODE  (first {cfg['SANITY_N']} instances)")
    log("Running sanity check — NOT saving to disk.")

    subset = instances[: cfg["SANITY_N"]]
    outputs = translate_all_instances(model, tokenizer, subset, cfg)

    for i, (inst, out) in enumerate(zip(subset, outputs)):
        src_text = inst["content_en"][:600]
        ref_text = inst["content_hi"][:600]
        hyp_text = out.get("hypothesis", "[EMPTY]")[:600]

        print(f"\n{'─'*72}")
        print(f"[{i+1}]  ID: {inst['id']}")
        print(f"  CATEGORY  : {categorise_instance(inst['content_en'], inst['content_hi'])}")
        print(f"  SOURCE    : {src_text}")
        print(f"  REFERENCE : {ref_text}")
        print(f"  HYPOTHESIS: {hyp_text}")
        if out.get("error"):
            print(f"  ERROR     : {out['error']}")

        # Quick per-instance SFS
        ref_full = inst["content_hi"]
        hyp_full = out.get("hypothesis", "")
        msp_v = metric_msp(ref_full, hyp_full)
        mcp_v = metric_mcp(ref_full, hyp_full)
        tsp_v = metric_tsp(ref_full, hyp_full)
        tcp_v = metric_tcp(ref_full, hyp_full)
        fmt = lambda v, name: f"{name}={v:.3f}" if v is not None else f"{name}=N/A"
        print(f"  SFS (inst): "
              f"{fmt(msp_v,'MSP')}  {fmt(mcp_v,'MCP')}  "
              f"{fmt(tsp_v,'TSP')}  {fmt(tcp_v,'TCP')}")

    print(f"\n{'─'*72}")
    log("Sanity check done. Set SANITY_CHECK=False to run full inference.", "OK")

# =============================================================================
# SECTION 14 — Main
# =============================================================================

def main() -> None:

    # ── Banner ────────────────────────────────────────────────────────────────
    section("MathDoc-ENHI  |  Gemma-4 31B  |  End-to-End Pipeline")
    log(f"Model    : {CONFIG['MODEL_ID']}")
    log(f"dtype    : {CONFIG['DTYPE']}")
    log(f"quant    : {CONFIG['QUANTISATION']}")
    log(f"batch_sz : {CONFIG['BATCH_SIZE']}")
    log(f"sanity   : {CONFIG['SANITY_CHECK']} (N={CONFIG['SANITY_N']})")
    log(f"COMET    : {'SKIP' if CONFIG['SKIP_COMET'] else 'ENABLED'}")
    log(f"Device(s): {get_device_info()}")

    torch.manual_seed(CONFIG["SEED"])

    # ── Load data ─────────────────────────────────────────────────────────────
    section("LOADING DATASET")
    try:
        instances = load_dataset_records(
            CONFIG["DATA_PATHS"],
            expected_total=CONFIG["EXPECTED_TOTAL"],
        )
    except DataValidationError as exc:
        log(f"Data validation failed: {exc}", "ERROR")
        sys.exit(1)

    log(f"Dataset validated: {len(instances)} instances total", "OK")

    # Category breakdown
    cat_counts: Dict[str, int] = Counter(
        categorise_instance(inst["content_en"], inst["content_hi"])
        for inst in instances
    )
    for cat, cnt in sorted(cat_counts.items()):
        log(f"  {cat:<14}: {cnt}")

    # ── Load model ────────────────────────────────────────────────────────────
    section("LOADING MODEL")
    try:
        model, tokenizer = load_model_and_tokenizer(CONFIG)
    except Exception as exc:
        log(f"Model loading failed: {exc}", "ERROR")
        traceback.print_exc()
        sys.exit(1)

    # ── Sanity check or full inference ────────────────────────────────────────
    if CONFIG["SANITY_CHECK"]:
        run_sanity_check(model, tokenizer, instances, CONFIG)
        log("Exiting after sanity check. Set SANITY_CHECK=False to run all.", "OK")
        return

    # ── Full inference ────────────────────────────────────────────────────────
    section("INFERENCE PHASE")

    # Check for existing outputs (resume support)
    hyp_path = CONFIG["HYPOTHESIS_JSONL"]
    existing_outputs: List[Dict] = []
    existing_ids: set = set()

    if os.path.exists(hyp_path):
        log(f"Found existing outputs at {hyp_path}. Loading for resume...", "WARN")
        try:
            with open(hyp_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if rec.get("hypothesis") and not rec.get("error"):
                            existing_outputs.append(rec)
                            existing_ids.add(str(rec["id"]))
            log(f"Resuming: {len(existing_ids)} already done, "
                f"{len(instances) - len(existing_ids)} remaining.", "OK")
        except Exception as exc:
            log(f"Could not read existing outputs: {exc}. Starting fresh.", "WARN")
            existing_outputs = []
            existing_ids     = set()

    # Filter to only un-processed instances
    remaining = [inst for inst in instances
                 if str(inst["id"]) not in existing_ids]
    log(f"Instances to translate: {len(remaining)}")

    new_outputs: List[Dict] = []
    if remaining:
        new_outputs = translate_all_instances(
            model, tokenizer, remaining, CONFIG)

    all_outputs = existing_outputs + new_outputs

    # ── Save hypotheses ───────────────────────────────────────────────────────
    save_hypotheses(all_outputs, hyp_path)

    # ── Free model memory before running COMET (which also loads a model) ─────
    del model
    free_gpu()

    # ── Evaluation ────────────────────────────────────────────────────────────
    results = run_full_evaluation(instances, all_outputs, CONFIG)

    # ── Save results ──────────────────────────────────────────────────────────
    save_results(results, CONFIG["RESULTS_JSON"])
    print_results_table(results)

    section("PIPELINE COMPLETE")
    log(f"Hypotheses : {hyp_path}", "OK")
    log(f"Results    : {CONFIG['RESULTS_JSON']}", "OK")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
