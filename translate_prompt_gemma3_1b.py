#!/usr/bin/env python
"""
Quick-start: translate a slice of the Pralekha dev-set with vLLM + Gemma-3-1b-pt.
"""
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # Removing this line
from pathlib import Path
from typing import List

from datasets import load_dataset
from vllm import LLM, SamplingParams

# Removing imports related to fixing cuda visible devices: import os, re, subprocess, sys

# Removing the _fix_cuda_visible_devices function
# def _fix_cuda_visible_devices():
#     cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
#     if not cvd.startswith("GPU-"):
#         # Already numeric -> nothing to do
#         return

#     # 1) Build a map {UUID -> index} from `nvidia-smi -L`
#     try:
#         smi = subprocess.check_output(["nvidia-smi", "-L"], text=True)
#     except FileNotFoundError:
#         sys.exit("!! nvidia-smi not found—cannot map GPU UUIDs to indices")

#     uuid2idx = {}
#     for line in smi.splitlines():
#         # Example: "GPU 2: NVIDIA H100 (UUID: GPU-b1b8e0d1-5d40-8a62-d5da-393bca3fd881)"
#         m = re.match(r"GPU\s+(\d+):.*\(UUID:\s+(GPU-[0-9a-f\-]+)\)", line)
#         if m:
#             idx, uuid = m.groups()
#             uuid2idx[uuid] = idx

#     # 2) Translate the job-allocated UUID list to indices
#     try:
#         new_ids = ",".join(uuid2idx[uuid] for uuid in cvd.split(","))
#     except KeyError as e:
#         missing = str(e).strip("'")
#         sys.exit(f"!! UUID {missing} not found in `nvidia-smi -L` output.\n"
#                  f"   CUDA_VISIBLE_DEVICES was: {cvd}")

#     os.environ["CUDA_VISIBLE_DEVICES"] = new_ids
#     print(f"[fix-gpu] CUDA_VISIBLE_DEVICES  {cvd}  →  {new_ids}")

# Removing the call to _fix_cuda_visible_devices
# _fix_cuda_visible_devices()

# ---------------------------------------------------------------------------
# 1. DATA LOADING
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
# 2. PROMPT TEMPLATING (Gemma format)
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a professional translator. Translate the user message precisely "
    "from {src} to {tgt}, preserving meaning, tone, and any markup."
)

def make_prompts(sentences: List[str], src: str, tgt: str) -> List[str]:
    sys = _SYSTEM.format(src=src, tgt=tgt)
    return [
        (
            f"<start_of_turn>user\n"
            f"{sys}\n"
            f"Please translate the following text from {src} to {tgt}:\n"
            f"{s}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        for s in sentences
    ]


# ---------------------------------------------------------------------------
# 3. MODEL INSTANTIATION (vLLM)
# ---------------------------------------------------------------------------
def init_gemma(checkpoint: str = "google/gemma-3-1b-pt") -> LLM:
    """
    Loads the Gemma 3 1B PT checkpoint under vLLM.
    """
    return LLM(
        model=checkpoint,
        dtype="float16", # Changed from bfloat16 to float16 for Tesla T4 compatibility
        tokenizer=checkpoint,
        max_model_len=8192,  # Increased for document-level translation
    )


# ---------------------------------------------------------------------------
# 4. BATCH TRANSLATION
# ---------------------------------------------------------------------------
def translate(
    llm: LLM,
    prompts: List[str],
    temperature: float = 0.0,
    max_tokens: int = 4096,  # Changed to 4096
) -> List[str]:
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    # vLLM returns a list of RequestOutput; we take the first candidate
    return [out.outputs[0].text.strip() for out in outputs]


# ---------------------------------------------------------------------------
# 5. END-TO-END EXECUTION
# ---------------------------------------------------------------------------
def main():
    SRC, TGT = "eng", "hin"          # change here for other pairs
    SUBSET = "dev"                   # or "train" / "test"
    N = 50                           # quick smoke-test
    MODEL = "google/gemma-3-1b-it" # Changed model to instruct following cell NlbMmAKflT6n

    ds = load_pralekha_split(SRC, TGT, SUBSET, N)
    print(f"Loaded {len(ds):,} rows from {SUBSET}/{SRC}_{TGT}")

    llm = init_gemma(MODEL)
    prompts = make_prompts(ds["src_txt"], SRC, TGT)
    translations = translate(llm, prompts)

    # Add predictions & persist
    ds = ds.add_column("pred_txt", translations)
    out_file = Path(f"translations_{MODEL.split('/')[-1]}_{SRC}_{TGT}_{SUBSET}_{N}.csv")
    ds.to_pandas().to_csv(out_file, index=False)
    print(f"✓ Saved translations to {out_file.resolve()}")

if __name__ == "__main__":
    main()
