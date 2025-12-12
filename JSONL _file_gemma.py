#---------------------JSONL--------------------------------------------        
# ======================================================
# ✅ Save predictions and references for ENG↔HIN # RUN THIS FOR CHECKS AND SAVE
# ======================================================
import json
from pathlib import Path

OUTPUT_DIR = Path("./universal_output_best")

# Assuming you still have your `evaluate_direction` function
# We will re-run it with a smaller sample size just to collect preds/refs

directions = ["eng_hin", "hin_eng"]
max_samples = 150  # or whatever you want
batch_size = 8

for split in directions:
    src, tgt = split.split("_")
    
    # Re-run evaluation to collect predictions and references
    preds, refs, _ = [], [], []
    ds = load_pralekha_split(src, tgt)
    ds_iter = iter(ds)
    processed = 0

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
        preds.extend(outs)
        refs.extend(batch_refs)
        processed += len(batch_ids)

    # Save to JSON
    save_path = OUTPUT_DIR / f"{split}_pred_refs.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump([{"prediction": p, "reference": r} for p, r in zip(preds, refs)],
                  f, ensure_ascii=False, indent=2)

    print(f"Saved {len(preds)} examples to {save_path}")
     
