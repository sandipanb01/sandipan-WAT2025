"""
gpt-oss-120b  MathDoc EN→HI  Translation Pipeline  (batched)
=============================================================
Model card : https://huggingface.co/openai/gpt-oss-120b
License    : Apache 2.0
Arch       : 117B params · 5.1B active (MoE) · MXFP4 quant · single H100/MI300X

Model-card specifics baked in
──────────────────────────────
* Harmony response format  — first line of system prompt MUST be "Reasoning: <level>"
* Configurable reasoning   — low / medium / high (flag: --reasoning)
* CoT stripping            — <think>…</think> removed before writing hypothesis
* max_tokens = 3072        — HARD-CODED, never overridden; matches model card guidance
* Batched inference        — --batch-size N fires N concurrent threads (ThreadPoolExecutor)

Providers (OpenAI-compatible)
──────────────────────────────
  vLLM self-hosted  →  --base-url http://localhost:8000/v1  --model openai/gpt-oss-120b
  OpenRouter        →  --base-url https://openrouter.ai/api/v1  --model openai/gpt-oss-120b:free
  OpenAI API        →  --base-url https://api.openai.com/v1  --model gpt-oss-120b

Env vars
────────
  GPT_OSS_API_KEY           required
  OPENROUTER_HTTP_REFERER   optional
  OPENROUTER_X_TITLE        optional  (default: mathdoc-enhi)

Quick-start
───────────
  # OpenRouter, batch=8, medium reasoning
  export GPT_OSS_API_KEY=sk-or-...
  python run_gpt_oss_120b.py \\
      --input  ncert_class11_math_en_hi_test_instances_curated.json \\
               ncert_class12_math_en_hi_test_instances_curated.json \\
      --output outputs/gpt_oss_120b.jsonl \\
      --base-url https://openrouter.ai/api/v1 \\
      --model   openai/gpt-oss-120b:free \\
      --reasoning medium \\
      --batch-size 8 \\
      --enforce-placeholders

  # Self-hosted vLLM, high reasoning, large batch
  vllm serve openai/gpt-oss-120b
  python run_gpt_oss_120b.py \\
      --input *.json \\
      --output outputs/gpt_oss_120b.jsonl \\
      --base-url http://localhost:8000/v1 \\
      --model openai/gpt-oss-120b \\
      --reasoning high \\
      --batch-size 32 \\
      --enforce-placeholders
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import re
import json
import time
import random
import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kw):          # noqa: E302
        return iterable

try:
    from openai import OpenAI
except ImportError:
    sys.exit("openai package not found.  Run:  pip install openai")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

MAX_TOKENS: int = 3072          # STRICT — never override per model card
DEFAULT_TEMPERATURE: float = 0.15
DEFAULT_TOP_P: float = 1.0
DEFAULT_BATCH_SIZE: int = 8
DEFAULT_MAX_RETRIES: int = 5
DEFAULT_BASE_SLEEP: float = 1.0
DEFAULT_SLEEP_BETWEEN: float = 0.0   # per-item sleep; usually 0 in batch mode


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING  (self-contained — no external data_io required)
# ══════════════════════════════════════════════════════════════════════════════

class DataValidationError(Exception):
    pass


def load_dataset(
    paths: List[str],
    expected_total: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load one or more JSON-array or JSONL files into a flat list of dicts."""
    records: List[Dict[str, Any]] = []

    for path in paths:
        if not os.path.exists(path):
            raise DataValidationError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read().strip()
        if raw.startswith("["):
            try:
                batch = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DataValidationError(f"JSON parse error in {path}: {exc}")
            if not isinstance(batch, list):
                raise DataValidationError(f"Expected JSON array in {path}")
            records.extend(batch)
        else:
            for lineno, line in enumerate(raw.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DataValidationError(
                        f"JSONL parse error {path}:{lineno}: {exc}"
                    )

    if expected_total is not None and len(records) != expected_total:
        raise DataValidationError(
            f"Expected {expected_total} records, got {len(records)}"
        )
    return records


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LaTeX / TABLE MASKING
# ══════════════════════════════════════════════════════════════════════════════

# Display math before inline math — order is critical.
_MASK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\$\$[^$]+?\$\$", re.DOTALL), "EQ"),
    (re.compile(r"\\\[.+?\\\]",    re.DOTALL), "EQ"),
    (re.compile(
        r"\\begin\{(equation|align|bmatrix|pmatrix|vmatrix|matrix)\*?\}"
        r".*?\\end\{\1\*?\}", re.DOTALL), "EQ"),
    (re.compile(r"\$[^$\n]+?\$"), "EQ"),
    (re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE), "TB"),
]


def mask_spans(text: str) -> Tuple[str, List[str]]:
    spans: List[str] = []
    masked = text
    for pat, prefix in _MASK_PATTERNS:
        def _repl(m, _p=prefix):
            spans.append(m.group(0))
            return f"[{_p}_{len(spans) - 1}]"
        masked = pat.sub(_repl, masked)
    return masked, spans


def unmask_spans(text: str, spans: List[str]) -> str:
    result = text
    for i, span in enumerate(spans):
        for prefix in ("EQ", "TB"):
            result = result.replace(f"[{prefix}_{i}]", span)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PLACEHOLDER INTEGRITY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

_PH_RE = re.compile(r"\[(EQ|TB)_(\d+)\]")


def _extract_ph(s: str) -> List[str]:
    return [m.group(0) for m in _PH_RE.finditer(s)]


def validate_placeholders(masked_input: str, model_output: str) -> Tuple[bool, str]:
    inp = _extract_ph(masked_input)
    out = _extract_ph(model_output)
    if inp == out:
        return True, "ok"
    missing = [p for p in inp if p not in out]
    extra   = [p for p in out if p not in inp]
    if missing or extra:
        return False, f"set_mismatch  missing={missing[:6]}  extra={extra[:6]}"
    return False, "order_mismatch"


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CHAIN-OF-THOUGHT STRIPPING  (gpt-oss-120b harmony format)
# ══════════════════════════════════════════════════════════════════════════════

_COT_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_cot(text: str) -> str:
    """Remove <think>…</think> reasoning blocks — per model card guidance."""
    return _COT_RE.sub("", text).strip()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PARAGRAPH-SAFE CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return parts or [""]


def _chunk_by_budget(paragraphs: List[str], budget: int) -> List[str]:
    chunks: List[str] = []
    cur = ""
    for p in paragraphs:
        if not cur:
            cur = p if len(p) <= budget else ""
            if len(p) > budget:
                for i in range(0, len(p), budget):
                    chunks.append(p[i:i + budget])
            continue
        candidate = cur + "\n\n" + p
        if len(candidate) <= budget:
            cur = candidate
        else:
            chunks.append(cur)
            if len(p) <= budget:
                cur = p
            else:
                for i in range(0, len(p), budget):
                    chunks.append(p[i:i + budget])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# 6.  SYSTEM PROMPT  (harmony-aware: "Reasoning: <level>" MUST be first line)
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM_TEMPLATE = """\
Reasoning: {level}

You are an expert English-to-Hindi translator specialising in NCERT \
mathematics textbooks (classes 11 and 12). Your translations must be accurate, \
natural, and preserve all mathematical structure exactly.

RULES — follow without exception:
1. Translate ONLY the English prose into fluent, grammatically correct Hindi.
2. Placeholders such as [EQ_0], [EQ_1], [TB_0] represent equations or tables. \
   Do NOT modify, translate, remove, duplicate, or reorder them. \
   Position each placeholder correctly within the Hindi sentence structure.
3. Output the Hindi translation ONLY — no preamble, no explanation, \
   no markdown fences."""


def build_system_prompt(reasoning_level: str) -> str:
    level = reasoning_level.lower()
    if level not in ("low", "medium", "high"):
        raise ValueError(f"reasoning must be low/medium/high — got '{level}'")
    return _SYSTEM_TEMPLATE.format(level=level)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  OPENAI-COMPATIBLE CLIENT
# ══════════════════════════════════════════════════════════════════════════════

def build_client(
    base_url: str,
    api_key: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=extra_headers or {},
    )


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SINGLE API CALL  (max_tokens=3072 STRICT, temperature=0.15, top_p=1.0)
# ══════════════════════════════════════════════════════════════════════════════

def _call_api(
    client: OpenAI,
    masked_text: str,
    model: str,
    system_prompt: str,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
) -> str:
    """
    Single chat completion.

    max_tokens is STRICTLY 3072 as specified in the model card guidance.
    temperature=0.15  → deterministic enough for translation, small creativity margin.
    top_p=1.0         → full nucleus (temperature does the narrowing).
    CoT blocks are stripped before returning.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": masked_text},
        ],
        max_tokens=MAX_TOKENS,          # ← 3072, HARD-CODED
        temperature=temperature,        # 0.15 default
        top_p=top_p,                    # 1.0  default
    )
    raw = (resp.choices[0].message.content or "").strip()
    return strip_cot(raw)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  RETRY LOGIC
# ══════════════════════════════════════════════════════════════════════════════

_RETRYABLE = [
    "429", "rate limit", "too many requests",
    "timeout", "timed out",
    "502", "503", "504",
    "bad gateway", "service unavailable", "gateway timeout",
    "connection reset", "connection aborted",
    "temporarily unavailable", "try again", "overloaded",
]


def _is_retryable(exc: Exception) -> bool:
    s = str(exc).lower()
    return any(k in s for k in _RETRYABLE)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  PER-RECORD TRANSLATION  (chunked + retry + placeholder enforcement)
# ══════════════════════════════════════════════════════════════════════════════

def translate_record(
    client: OpenAI,
    text: str,
    model: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_retries: int,
    base_sleep: float,
    enforce_placeholders: bool,
    chunk_chars: int,
) -> Tuple[str, Optional[str]]:
    """
    Returns (hypothesis, error_or_None).

    Parameters
    ──────────
    client              : pre-built OpenAI-compatible client
    text                : raw English source text
    model               : provider model ID string
    system_prompt       : harmony-formatted system prompt (includes Reasoning level)
    temperature         : 0.15 recommended for translation determinism
    top_p               : 1.0 (full nucleus; temperature does the steering)
    max_retries         : total attempts per chunk before giving up (default 5)
    base_sleep          : base seconds for exponential backoff (default 1.0)
    enforce_placeholders: if True, retries when model corrupts [EQ_k]/[TB_k]
    chunk_chars         : 0 = no chunking; >0 = paragraph-safe split budget
                          (recommended 8000–16000 for very long documents)
    """
    # ── optional chunking ─────────────────────────────────────────────────────
    if chunk_chars > 0 and len(text) > chunk_chars:
        chunks = _chunk_by_budget(_split_paragraphs(text), chunk_chars)
    else:
        chunks = [text]

    translated_chunks: List[str] = []
    any_error: Optional[str] = None

    for ci, chunk in enumerate(chunks):
        masked, spans = mask_spans(chunk)
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                out_masked = _call_api(
                    client=client,
                    masked_text=masked,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                )

                if enforce_placeholders:
                    ok, reason = validate_placeholders(masked, out_masked)
                    if not ok:
                        raise RuntimeError(f"PlaceholderIntegrityError: {reason}")

                translated_chunks.append(unmask_spans(out_masked, spans))
                last_exc = None
                break

            except Exception as exc:
                last_exc = exc
                if attempt == max_retries - 1:
                    break

                # Exponential backoff with ±30 % jitter
                backoff = min(64.0, base_sleep * (2 ** (attempt + 1)))
                jitter  = random.random() * 0.3 * backoff
                if (not _is_retryable(exc)
                        and "placeholderintegrityerror" not in str(exc).lower()):
                    backoff, jitter = min(2.0, base_sleep + 0.2 * attempt), 0.0
                time.sleep(backoff + jitter)

        if last_exc is not None:
            any_error = f"chunk_{ci}: {last_exc}"
            translated_chunks.append("")

    hypothesis = "\n\n".join(c for c in translated_chunks if c).strip()
    return hypothesis, any_error


# ══════════════════════════════════════════════════════════════════════════════
# 11.  BATCHED WORKER  (ThreadPoolExecutor — concurrent API calls)
# ══════════════════════════════════════════════════════════════════════════════

def _process_instance(
    inst: Dict[str, Any],
    client: OpenAI,
    model: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_retries: int,
    base_sleep: float,
    enforce_placeholders: bool,
    chunk_chars: int,
) -> Dict[str, Any]:
    """Worker function — called concurrently inside the thread pool."""
    rec_id = inst.get("id", "")
    source = inst.get("content_en", inst.get("source", ""))

    if not source.strip():
        return {"id": rec_id, "source": source,
                "hypothesis": "", "error": "empty_source"}

    hyp, err = translate_record(
        client=client,
        text=source,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_retries=max_retries,
        base_sleep=base_sleep,
        enforce_placeholders=enforce_placeholders,
        chunk_chars=chunk_chars,
    )

    record: Dict[str, Any] = {"id": rec_id, "source": source, "hypothesis": hyp}
    if err:
        record["error"] = err
    return record


def run_batched(
    instances: List[Dict[str, Any]],
    client: OpenAI,
    model: str,
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_retries: int,
    base_sleep: float,
    enforce_placeholders: bool,
    chunk_chars: int,
    batch_size: int,
    output_path: str,
    write_lock: threading.Lock,
    done_ids: set,
) -> int:
    """
    Submit all pending instances to a thread pool of size `batch_size`.
    Results are written to `output_path` as soon as each future completes
    (streaming write — safe to interrupt).

    Returns the number of errored records.
    """
    pending = [inst for inst in instances
               if inst.get("id", "") not in done_ids]

    if not pending:
        print("Nothing to do — all records already in output.")
        return 0

    errors_total = 0

    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        future_map = {
            pool.submit(
                _process_instance,
                inst, client, model, system_prompt,
                temperature, top_p,
                max_retries, base_sleep,
                enforce_placeholders, chunk_chars,
            ): inst.get("id", "")
            for inst in pending
        }

        with tqdm(total=len(pending), desc="Translating", unit="ex") as pbar:
            for future in as_completed(future_map):
                try:
                    record = future.result()
                except Exception as exc:
                    rec_id = future_map[future]
                    record = {
                        "id": rec_id, "source": "",
                        "hypothesis": "",
                        "error": f"unhandled_future_exception: {exc}",
                    }

                if record.get("error"):
                    errors_total += 1

                with write_lock:
                    with open(output_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

                pbar.update(1)

    return errors_total


# ══════════════════════════════════════════════════════════════════════════════
# 12.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    p = argparse.ArgumentParser(
        description="gpt-oss-120b MathDoc EN→HI translation — batched",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--input",  nargs="+", required=True,
                   help="Input JSON / JSONL file(s)")
    p.add_argument("--output", required=True,
                   help="Output JSONL file")
    p.add_argument("--expected-total", type=int, default=None,
                   help="Assert dataset has exactly this many records")

    # Provider / auth
    p.add_argument("--api-key",
                   default=os.environ.get("GPT_OSS_API_KEY"),
                   help="API key  (env: GPT_OSS_API_KEY)")
    p.add_argument("--base-url",
                   default="https://openrouter.ai/api/v1",
                   help="OpenAI-compatible endpoint")
    p.add_argument("--model",
                   default="openai/gpt-oss-120b:free",
                   help="Model ID — openai/gpt-oss-120b[:free] | gpt-oss-120b | openai/gpt-oss-120b")

    # gpt-oss-120b model-card knobs
    p.add_argument("--reasoning",
                   default="medium", choices=["low", "medium", "high"],
                   help="Harmony reasoning effort: low=fast  medium=balanced  high=deep")

    # Generation  (max_tokens is HARD-CODED to 3072 — not exposed as a flag)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help="Sampling temperature (0.15 recommended for translation)")
    p.add_argument("--top-p",       type=float, default=DEFAULT_TOP_P,
                   help="Nucleus sampling top-p (1.0 = full nucleus)")

    # Batching
    p.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE,
                   help="Number of concurrent API calls (ThreadPoolExecutor workers). "
                        "Increase for self-hosted vLLM; keep ≤4 for free-tier OpenRouter.")

    # Reliability
    p.add_argument("--max-retries", type=int,   default=DEFAULT_MAX_RETRIES)
    p.add_argument("--base-sleep",  type=float, default=DEFAULT_BASE_SLEEP,
                   help="Base multiplier for exponential back-off (seconds)")
    p.add_argument("--enforce-placeholders", action="store_true",
                   help="Retry if model corrupts/drops/reorders [EQ_k]/[TB_k]")

    # Chunking
    p.add_argument("--chunk-chars", type=int, default=0,
                   help="Paragraph-safe chunking budget (chars). "
                        "0=disabled. Recommended 8000–16000 for long docs.")

    # OpenRouter headers (safe to ignore elsewhere)
    p.add_argument("--http-referer",
                   default=os.environ.get("OPENROUTER_HTTP_REFERER", ""))
    p.add_argument("--x-title",
                   default=os.environ.get("OPENROUTER_X_TITLE", "mathdoc-enhi"))

    # Resume
    p.add_argument("--resume", action="store_true",
                   help="Append to existing output, skip already-written IDs")

    args = p.parse_args()

    # ── guards ────────────────────────────────────────────────────────────────
    if not args.api_key:
        sys.exit(
            "Missing API key.\n"
            "  export GPT_OSS_API_KEY=<your_key>   or use --api-key"
        )

    # ── load data ─────────────────────────────────────────────────────────────
    try:
        instances = load_dataset(args.input, expected_total=args.expected_total)
    except DataValidationError as exc:
        sys.exit(f"Data validation failed: {exc}")
    print(f"Loaded {len(instances)} instances.")

    # ── resume: collect already-done IDs ─────────────────────────────────────
    done_ids: set = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        done_ids.add(json.loads(line).get("id", ""))
                    except json.JSONDecodeError:
                        pass
        print(f"Resume: {len(done_ids)} records already done.")

    # ── build client ──────────────────────────────────────────────────────────
    extra_headers: Dict[str, str] = {}
    if args.http_referer:
        extra_headers["HTTP-Referer"] = args.http_referer
    if args.x_title:
        extra_headers["X-Title"] = args.x_title

    client        = build_client(args.base_url, args.api_key, extra_headers)
    system_prompt = build_system_prompt(args.reasoning)

    # ── output dir ────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if not args.resume:
        # Truncate / create fresh
        open(args.output, "w", encoding="utf-8").close()

    # ── banner ────────────────────────────────────────────────────────────────
    W = 64
    print("=" * W)
    print("  gpt-oss-120b  ·  MathDoc EN→HI  ·  Batched Inference")
    print("=" * W)
    print(f"  Endpoint      : {args.base_url}")
    print(f"  Model         : {args.model}")
    print(f"  Reasoning     : {args.reasoning}  (harmony format)")
    print(f"  max_tokens    : {MAX_TOKENS}  [STRICT]")
    print(f"  temperature   : {args.temperature}    top_p: {args.top_p}")
    print(f"  Batch size    : {args.batch_size}  concurrent workers")
    print(f"  Retries       : {args.max_retries}    base_sleep: {args.base_sleep}s")
    print(f"  Enforce ph.   : {args.enforce_placeholders}")
    print(f"  Chunk chars   : {args.chunk_chars or 'disabled'}")
    print(f"  Resume        : {args.resume}")
    print(f"  Output        : {args.output}")
    print("=" * W + "\n")

    # ── run ───────────────────────────────────────────────────────────────────
    write_lock = threading.Lock()

    errors = run_batched(
        instances=instances,
        client=client,
        model=args.model,
        system_prompt=system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_retries=args.max_retries,
        base_sleep=args.base_sleep,
        enforce_placeholders=args.enforce_placeholders,
        chunk_chars=args.chunk_chars,
        batch_size=args.batch_size,
        output_path=args.output,
        write_lock=write_lock,
        done_ids=done_ids,
    )

    # ── summary ───────────────────────────────────────────────────────────────
    total = len(instances) - len(done_ids)
    print(f"\n{'=' * W}")
    print(f"  Done.  Written → {args.output}")
    print(f"  Processed : {total}  |  Errors: {errors}")
    print("=" * W)


if __name__ == "__main__":
    main()
