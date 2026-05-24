"""
Gemma 4 31B inference + advisor-style evaluation on MathDoc-ENHI.

Keeps your exact interface (like code C):
    python run_gemma_eval_mathdoc_enhi.py \
        --input  class11.json class12.json \
        --output outputs/gemma.jsonl

Modes:
- DEFAULT (no API key): LOCAL HuggingFace inference (advisor way)
- Optional: Google AI Studio API inference (--use-api + GEMMA_API_KEY)

Masking:
- LaTeX -> [EQ_k]
- HTML tables -> [TB_k]
Restore verbatim after translation.

Outputs:
1) --output (JSONL): id, source, hypothesis, error?
2) outputs/gemma.metrics.json : BLEU, CHRF + debug metrics
3) outputs/gemma.metrics.csv  : one-row table
4) outputs/gemma.lengths.json : length stats (chars)

NOTE:
Local Gemma-4-31B may require a strong GPU + HF access depending on model gating.
"""

import os
import re
import json
import time
import math
import argparse
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import sacrebleu
from tqdm import tqdm

from data_io import DataValidationError, load_dataset

# Optional imports (only used in certain modes)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


# ============================================================
# Masking patterns (same as your C)
# ============================================================
_MASK_PATTERNS = [
    (re.compile(r"\$\$[^$]+?\$\$", re.DOTALL), "EQ"),
    (re.compile(r"\\\[.+?\\\]", re.DOTALL), "EQ"),
    (re.compile(
        r"\\begin\{(equation|align|bmatrix|pmatrix|vmatrix|matrix)\*?\}.*?"
        r"\\end\{\1\*?\}", re.DOTALL
    ), "EQ"),
    (re.compile(r"\$[^$\n]+?\$"), "EQ"),
    (re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE), "TB"),
]

_PLACEHOLDER_RE = re.compile(r"\[(EQ|TB)_(\d+)\]")


def mask_spans(text: str) -> Tuple[str, List[str]]:
    spans = []
    masked = text
    for pat, prefix in _MASK_PATTERNS:
        def repl(m, prefix=prefix):
            spans.append(m.group(0))
            return f"[{prefix}_{len(spans) - 1}]"
        masked = pat.sub(repl, masked)
    return masked, spans


def unmask_spans(text: str, spans: List[str]) -> str:
    result = text
    for i, span in enumerate(spans):
        for prefix in ("EQ", "TB"):
            result = result.replace(f"[{prefix}_{i}]", span)
    return result


def extract_placeholders(s: str) -> List[str]:
    return [m.group(0) for m in _PLACEHOLDER_RE.finditer(s)]


def placeholder_ok(masked_input: str, model_output: str) -> Tuple[bool, str]:
    inp = extract_placeholders(masked_input)
    out = extract_placeholders(model_output)
    if inp == out:
        return True, "ok"
    if set(inp) != set(out):
        missing = [p for p in inp if p not in out]
        extra = [p for p in out if p not in inp]
        return False, f"set_mismatch missing={missing[:10]} extra={extra[:10]}"
    return False, "order_mismatch"


# ============================================================
# Metrics (advisor) + extra debug metrics
# ============================================================
def calc_metrics(preds: List[str], refs: List[str]) -> Tuple[float, float]:
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(float(bleu), 2), round(float(chrf), 2)


def summarize_lengths(arr: List[int]) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.int64)
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "min": int(a.min()),
        "max": int(a.max()),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


# ============================================================
# Inference: API mode (Google GenAI) — mirrors your C
# ============================================================
def translate_one_api(client, text: str, model: str, max_new_tokens: int) -> Tuple[str, Optional[str], bool, str]:
    masked, spans = mask_spans(text)

    system_instruction = (
        "You are a professional mathematical translator. Translate the following English text into Hindi. "
        "CRITICAL RULE: You will encounter placeholders like [EQ_0], [EQ_1], [TB_0], etc. "
        "Do NOT translate, alter, omit, duplicate, or reorder these placeholders. Keep them exactly as they are "
        "in their correct relative positions in the translated Hindi sentence. Output ONLY the translated text."
    )

    try:
        resp = client.models.generate_content(
            model=model,
            contents=masked,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                max_output_tokens=max_new_tokens,  # GenAI uses max_output_tokens
            ),
        )
        out_masked = (resp.text or "").strip()
        ok, reason = placeholder_ok(masked, out_masked)
        hyp = unmask_spans(out_masked, spans)
        err = None if hyp.strip() else "EmptyOutput"
        if not ok:
            err = err or f"PlaceholderIntegrityError: {reason}"
        return hyp, err, ok, reason
    except Exception as e:
        return "", str(e), False, "api_exception"


# ============================================================
# Inference: LOCAL mode (advisor way, no API key)
# ============================================================
def build_prompt_local(masked_src: str, tokenizer) -> str:
    # Advisor-style: apply_chat_template with a user message
    messages = [
        {
            "role": "user",
            "content": (
                "Translate the following sentence from English to Hindi.\n\n"
                "CRITICAL RULE: You will see placeholders like [EQ_0], [EQ_1], [TB_0]. "
                "Do NOT modify, translate, remove, duplicate, or reorder these placeholders. "
                "Keep them exactly as they are.\n\n"
                f"English: {masked_src}"
            ),
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def translate_one_local(model, tokenizer, text: str, max_new_tokens: int) -> Tuple[str, Optional[str], bool, str]:
    masked, spans = mask_spans(text)

    prompt = build_prompt_local(masked, tokenizer)
    toks = tokenizer(prompt, return_tensors="pt", truncation=True)

    input_ids = toks["input_ids"].to(model.device)
    attention_mask = toks.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )

    new_tokens = out[:, input_ids.shape[1]:]
    out_masked = (tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0] or "").strip()

    ok, reason = placeholder_ok(masked, out_masked)
    hyp = unmask_spans(out_masked, spans)
    err = None if hyp.strip() else "EmptyOutput"
    if not ok:
        err = err or f"PlaceholderIntegrityError: {reason}"
    return hyp, err, ok, reason


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--api-key", default=os.environ.get("GEMMA_API_KEY"))
    p.add_argument("--use-api", action="store_true", help="Use Google AI Studio API instead of local HF")
    p.add_argument("--model", default="gemma-4-31b-it")
    p.add_argument("--sleep", type=float, default=0.5)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--expected-total", type=int)
    # Hard requirement you kept repeating:
    p.add_argument("--max-new-tokens", type=int, default=3072)
    args = p.parse_args()

    # Load dataset EXACTLY like your C expects
    try:
        instances = load_dataset(args.input, expected_total=args.expected_total)
    except DataValidationError as e:
        raise SystemExit(f"Data validation failed: {e}")
    print(f"Validated {len(instances)} instances.")

    # Init inference backend
    use_api = bool(args.use_api)
    if use_api:
        if genai is None or types is None:
            raise SystemExit("google-genai not available. pip install google-genai")
        if not args.api_key:
            raise SystemExit("No API key found. Set GEMMA_API_KEY or pass --api-key, or remove --use-api.")
        client = genai.Client(api_key=args.api_key)
        print(f"Using API inference via Google GenAI. Model={args.model}")
        local_model = None
        local_tokenizer = None
    else:
        if torch is None or AutoTokenizer is None:
            raise SystemExit("transformers/torch not available for local mode.")
        print(f"Using LOCAL HF inference. Model repo/id={args.model}")
        # dtype choice: bfloat16 preferred if supported
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        local_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=dtype,
        )
        local_model.eval()
        local_tokenizer = AutoTokenizer.from_pretrained(args.model)
        if local_tokenizer.pad_token is None:
            local_tokenizer.pad_token = local_tokenizer.eos_token
        client = None

    outputs = []
    preds, refs = [], []

    # Debug tracking (prevents “everything zero” mysteries)
    empty_outputs = 0
    error_count = 0
    placeholder_violations = 0
    in_char_lens, out_char_lens = [], []

    for inst in tqdm(instances):
        hyp = ""
        err = None
        ph_ok = True
        ph_reason = "ok"

        for attempt in range(args.max_retries):
            try:
                if use_api:
                    hyp, err, ph_ok, ph_reason = translate_one_api(
                        client, inst["content_en"], model=args.model, max_new_tokens=args.max_new_tokens
                    )
                else:
                    hyp, err, ph_ok, ph_reason = translate_one_local(
                        local_model, local_tokenizer, inst["content_en"], max_new_tokens=args.max_new_tokens
                    )
                # break if no hard errors
                if err is None or (err and "PlaceholderIntegrityError" not in err):
                    break
            except Exception as e:
                err = str(e)
                time.sleep(2 * (attempt + 1))

        record = {
            "id": inst["id"],
            "source": inst["content_en"],
            "hypothesis": hyp,
            "placeholder_ok": ph_ok,
            "placeholder_reason": ph_reason,
        }
        if err:
            record["error"] = err

        outputs.append(record)

        ref = inst.get("content_hi", inst.get("reference", ""))
        refs.append(ref)
        preds.append(hyp)

        in_char_lens.append(len(inst["content_en"] or ""))
        out_char_lens.append(len(hyp or ""))

        if not (hyp or "").strip():
            empty_outputs += 1
        if err:
            error_count += 1
        if not ph_ok:
            placeholder_violations += 1

        time.sleep(args.sleep)

    # Write output JSONL exactly at args.output (like your C)
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    # Compute metrics (advisor)
    bleu, chrf = calc_metrics(preds, refs)

    # Save metrics next to your output (same folder)
    base = Path(args.output)
    metrics_json = base.with_suffix(".metrics.json")
    metrics_csv = base.with_suffix(".metrics.csv")
    lengths_json = base.with_suffix(".lengths.json")

    summary = {
        "model": args.model,
        "mode": "api" if use_api else "local_hf",
        "num_records": len(instances),
        "generation": {
            "max_new_tokens": args.max_new_tokens,  # (local)
            "max_output_tokens": args.max_new_tokens,  # (api equivalent)
            "temperature": 1.0 if use_api else 0.0,   # local is deterministic
            "top_p": 0.95 if use_api else None,
            "top_k": 64 if use_api else None,
            "do_sample": False,
        },
        "BLEU": bleu,
        "CHRF": chrf,
        "debug": {
            "empty_output_rate_pct": round(100.0 * empty_outputs / max(1, len(instances)), 4),
            "error_rate_pct": round(100.0 * error_count / max(1, len(instances)), 4),
            "placeholder_violation_rate_pct": round(100.0 * placeholder_violations / max(1, len(instances)), 4),
        },
    }

    lengths = {
        "input_char_lengths": summarize_lengths(in_char_lens),
        "output_char_lengths": summarize_lengths(out_char_lens),
    }

    metrics_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame([{
        "model": summary["model"],
        "mode": summary["mode"],
        "num_records": summary["num_records"],
        "BLEU": summary["BLEU"],
        "CHRF": summary["CHRF"],
        "max_new_tokens": args.max_new_tokens,
        "empty_output_rate_pct": summary["debug"]["empty_output_rate_pct"],
        "error_rate_pct": summary["debug"]["error_rate_pct"],
        "placeholder_violation_rate_pct": summary["debug"]["placeholder_violation_rate_pct"],
    }]).to_csv(metrics_csv, index=False)
    lengths_json.write_text(json.dumps(lengths, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n============================================================")
    print("✅ DONE")
    print("============================================================")
    print("Predictions JSONL :", args.output)
    print("Metrics JSON      :", metrics_json)
    print("Metrics CSV       :", metrics_csv)
    print("Lengths JSON      :", lengths_json)
    print("BLEU              :", bleu)
    print("CHRF              :", chrf)
    print("============================================================")


if __name__ == "__main__":
    main()
