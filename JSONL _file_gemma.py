# --------------------- JSONL + ZIP --------------------------------------------
import json
import zipfile
from pathlib import Path
from google.colab import files

OUTPUT_DIR = Path("./universal_output_best")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

directions = ["eng_hin", "hin_eng"]
max_samples = 150  # adjust as needed
batch_size = 8

# Save each direction as JSONL
jsonl_files = []
for split in directions:
    src, tgt = split.split("_")
    
    preds, refs = [], []
    ds = load_pralekha_split(src, tgt)
    ds_iter = iter(ds)
    processed = 0

    save_path = OUTPUT_DIR / f"{split}_pred_refs.jsonl"
    with open(save_path, "w", encoding="utf-8") as f:
        while processed < max_samples:
            batch_src, batch_refs, batch_ids = [], [], []

            for _ in range(batch_size):
                try:
                    ex = next(ds_iter)
                except StopIteration:
                    break

                if src == "eng" and tgt == "hin":
                    src_text = ex["src_txt"]
                    ref_text = ex["tgt_txt"]
                else:
                    src_text = ex["tgt_txt"]
                    ref_text = ex["src_txt"]

                fake_ex = {"src_txt": src_text}
                ids = build_eval_prompt_tokenized(fake_ex, tokenizer, src, tgt)

                batch_src.append(src_text)
                batch_refs.append(ref_text)
                batch_ids.append(ids)

            if not batch_ids:
                break

            outs = generate_batch(model, tokenizer, batch_ids)
            for p, r in zip(outs, batch_refs):
                json_line = json.dumps({"prediction": p, "reference": r}, ensure_ascii=False)
                f.write(json_line + "\n")

            processed += len(batch_ids)

    print(f"Saved {processed} examples to {save_path}")
    jsonl_files.append(save_path)

# Create ZIP of all JSONL files
zip_path = OUTPUT_DIR / "pred_refs_eng_hin.zip"
with zipfile.ZipFile(zip_path, 'w') as zipf:
    for jsonl_file in jsonl_files:
        zipf.write(jsonl_file, arcname=jsonl_file.name)

print(f"All JSONL files zipped at {zip_path}")

# Download the ZIP
files.download(str(zip_path))
