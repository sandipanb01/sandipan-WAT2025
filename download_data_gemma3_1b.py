import json
from pathlib import Path
from typing import List

from datasets import load_dataset
from tqdm.auto import tqdm

PAIRS: List[str] = [
    "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal",
    "eng_mar", "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
]

def save_jsonl(lines: List[str], path: Path) -> None:
    """Saves a list of strings to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            json.dump([ln], f, ensure_ascii=False)
            f.write("\n")

def dump_pair(subset: str, pair: str, out_root: Path) -> None:
    """Downloads and saves a specific language pair from a subset."""
    src_lang, tgt_lang = pair.split("_")
    ds = load_dataset("ai4bharat/Pralekha", subset, split=f"{src_lang}_{tgt_lang}")
    print(f"  {subset}/{pair}: {len(ds):,} docs")
    save_jsonl(ds["src_txt"], out_root / subset / pair / f"doc.{src_lang}.jsonl")
    save_jsonl(ds["tgt_txt"], out_root / subset / pair / f"doc.{tgt_lang}.jsonl")

def main() -> None:
    """Main function to parse arguments and download data."""
    # Define output directory and splits directly for Colab environment
    out_root = Path("./pralekha_data").expanduser()  # Example output directory
    splits = ["dev", "test"]  # Example splits

    print("Language pairs fixed to:", ", ".join(PAIRS))
    print("Splits:", ", ".join(splits))

    for subset in splits:
        for pair in tqdm(PAIRS, desc=f"[{subset}]"):
            dump_pair(subset, pair, out_root)

    print("\n✓ All data has been written to", out_root.resolve())

if __name__ == "__main__":
    main()
