# ------------------------------ EVALUATION HELPERS
def extract_langs_from_split(split):
    parts = split.split("_")
    return parts[0], parts[1]

def build_eval_prompt(example, tokenizer):
    prompt = (
        f"Translate this {example['src_lang']} text to "
        f"{example['tgt_lang']}:\n{example['src_txt']}"
    )
    messages = {"messages": [{"role": "user", "content": prompt}]}
    return apply_chat_template(messages, tokenizer=tokenizer)

def generate_batch(model, tok, prompts, batch_size):
    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LEN
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            **BEAM_KWARGS
        )

    outputs = []
    prompt_len = enc["input_ids"].shape[1]

    for i in range(len(prompts)):
        gen_ids = out[i][prompt_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True).strip()
        outputs.append(text)

    return outputs

# ------------------------------ EVALUATION
def evaluate_model(
    model,
    tok,
    max_samples=200,
    batch_size=8,
    save_json=True,
    save_zip=True
):
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"

    splits = get_dataset_split_names(dataset_name, config_name)
    split = splits[0]
    src_lang, tgt_lang = extract_langs_from_split(split)

    ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)

    preds, refs, srcs = [], [], []

    print("\n==============================")
    print(f"🔍 Evaluating: {src_lang} → {tgt_lang}")
    print("==============================\n")

    iterator = iter(ds)
    processed = 0
    pbar = tqdm(total=max_samples, desc="Evaluating")

    while processed < max_samples:
        batch_examples = []

        for _ in range(batch_size):
            try:
                batch_examples.append(next(iterator))
            except StopIteration:
                break

        if not batch_examples:
            break

        processed += len(batch_examples)
        pbar.update(len(batch_examples))

        prompts = [build_eval_prompt(ex, tok) for ex in batch_examples]
        outs = generate_batch(model, tok, prompts, batch_size)

        for ex, pred in zip(batch_examples, outs):
            preds.append(pred)
            refs.append(ex["tgt_txt"].strip())
            srcs.append(ex["src_txt"].strip())

    pbar.close()
    print(f"\nDone! Total evaluated = {len(preds)}")

    # ---------- BLEU & CHRF ----------
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    print(f"\n🌟 BLEU:   {bleu:.2f}")
    print(f"🌟 chrF++: {chrf:.3f}")

    # ---------- JSONL ----------
    jsonl_path = OUTPUT_DIR / "eval_predictions.jsonl"
    if save_json:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for p, r, s in zip(preds, refs, srcs):
                f.write(json.dumps(
                    {"pred": p, "ref": r, "src": s},
                    ensure_ascii=False
                ) + "\n")
        print(f"📄 Saved predictions → {jsonl_path}")

    # ---------- TOP-10 TABLE ----------
    scores = [
        sacrebleu.sentence_chrf(p, [r]).score
        for p, r in zip(preds, refs)
    ]

    rank = list(enumerate(scores))
    rank.sort(key=lambda x: -x[1])
    top10 = rank[:10]

    print("\n================ TOP-10 (chrF) ================")
    print(f"{'Rank':<6}{'Score':<8}{'Prediction'}")
    print("-----------------------------------------------")

    for i, (idx, sc) in enumerate(top10, 1):
        preview = preds[idx][:120].replace("\n", " ")
        print(f"{i:<6}{sc:<8.2f}{preview}")

    # ---------- ZIP BUNDLE ----------
    if save_zip:
        zip_path = OUTPUT_DIR / "eval_bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            if save_json:
                z.write(jsonl_path, arcname="eval_predictions.jsonl")
        print(f"\n📦 Saved ZIP bundle → {zip_path}")

    return bleu, chrf

# ------------------------------ MAIN
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES

    # 1️⃣ Train
    model, tok, trainer = train_model(max_samples=max_samples)

    # 2️⃣ Full Evaluation
    bleu, chrf = evaluate_model(
        model,
        tok,
        max_samples=150,
        batch_size=EVAL_BATCH_SIZE,
    )
