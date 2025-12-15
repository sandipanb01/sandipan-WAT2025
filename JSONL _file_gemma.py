# ------------------ JSONL EXPORT --------------------------------
OUTPUT_DIR = Path("./universal_output_best")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

directions = ["eng_hin", "hin_eng"]
max_samples_export = 100
batch_size = 8

jsonl_files = []

for split in directions:
    src, tgt = split.split("_")

    raw_ds = load_pralekha_split(src, tgt)
    eval_ds = EvalDataset(raw_ds, tokenizer, src, tgt)

    collate = partial(eval_collate_fn, tokenizer=tokenizer)
    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=0
    )

    save_path = OUTPUT_DIR / f"{split}_pred_refs.jsonl"
    processed = 0

    with open(save_path, "w", encoding="utf-8") as f:
        for input_ids, attention_mask, refs in loader:
            preds = generate_batch(model, tokenizer, input_ids, attention_mask)

            for p, r in zip(preds, refs):
                f.write(json.dumps(
                    {"prediction": p, "reference": r},
                    ensure_ascii=False
                ) + "\n")

            processed += len(refs)
            if processed >= max_samples_export:
                break

    jsonl_files.append(save_path)
    print(f"Saved {processed} examples to {save_path}")


# ------------------ ZIP -----------------------------------------
zip_path = OUTPUT_DIR / "pred_refs_eng_hin.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for f in jsonl_files:
        zipf.write(f, arcname=f.name)

print(f"ZIP saved at: {zip_path}")

# Optional (Colab only)
# from google.colab import files
# files.download(str(zip_path))

# ------------------ ZIP DOWNLOAD CELL -------------------------
from pathlib import Path

# Path to the ZIP created in your previous cell
zip_path = Path("./universal_output_best/pred_refs_eng_hin.zip")

if zip_path.exists():
    files.download(str(zip_path))
    print(f"Downloading ZIP: {zip_path}")
else:
    print(f"ZIP file not found at {zip_path}, please check previous export step.")
