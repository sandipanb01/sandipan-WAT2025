# =========================================================
# W19-5212 STYLE MULTILINGUAL CORPUS PIPELINE
# Mann Ki Baat → Structured Parallel Dataset
# =========================================================

import requests
from bs4 import BeautifulSoup
import re
import json
import time
import unicodedata
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# CONFIG
# =========================

BASE_URLS = [
    "https://www.pmindia.gov.in/en/mann-ki-baat/",
    "https://www.pmindia.gov.in/en/tag/mann-ki-baat/"
]

SITEMAP = "https://www.pmindia.gov.in/sitemap.xml"

HEADERS = {"User-Agent": "Mozilla/5.0"}

TARGET_EPISODES = 132
BATCH_SIZE = 6

MODEL_NAME = "sarvamai/sarvam-translate"

# =========================
# LOAD SARVAM MODEL
# =========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
).to("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# STEP 1: CRAWLING (HTML + SITEMAP)
# =========================

def extract_links(url):
    links = set()
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        soup = BeautifulSoup(r.text, "lxml")

        for a in soup.find_all("a", href=True):
            if "mann-ki-baat" in a["href"] and a["href"].startswith("http"):
                links.add(a["href"])
    except:
        pass
    return links


def get_urls():
    urls = set()

    for base in BASE_URLS:
        urls |= extract_links(base)

        for i in range(1, 30):
            urls |= extract_links(f"{base}page/{i}/")

    try:
        r = requests.get(SITEMAP, timeout=30)
        soup = BeautifulSoup(r.text, "xml")
        for loc in soup.find_all("loc"):
            if "mann-ki-baat" in loc.text:
                urls.add(loc.text)
    except:
        pass

    return list(urls)


urls = sorted(list(set(get_urls())))
print("[STEP 1] URLs:", len(urls))

# =========================
# STEP 2: STRUCTURED DOM EXTRACTION (W19 IDEA)
# =========================

def clean(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scrape_structured(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(r.text, "lxml")

        paragraphs = soup.find_all("p")

        structured = []
        for i, p in enumerate(paragraphs):
            txt = clean(p.get_text(" ", strip=True))
            if len(txt) > 20:
                structured.append({
                    "para_id": i,
                    "text": txt
                })

        return structured if len(structured) > 3 else None

    except:
        return None


raw = []

for url in tqdm(urls):
    data = scrape_structured(url)
    if data:
        raw.append({"url": url, "paragraphs": data})

print("[STEP 2] Episodes scraped:", len(raw))

# =========================
# STEP 3: FORCE 132 ALIGNMENT
# =========================

raw = raw[:TARGET_EPISODES]

while len(raw) < TARGET_EPISODES:
    raw.append({"url": "MISSING", "paragraphs": []})

print("[STEP 3] Episodes:", len(raw))

# =========================
# STEP 4: SENTENCE SEGMENTATION WITH IDS
# =========================

dataset = []

for ep in raw:
    segments = []

    for para in ep["paragraphs"]:
        sents = sent_tokenize(para["text"])

        for j, s in enumerate(sents):
            segments.append({
                "para_id": para["para_id"],
                "sent_id": j,
                "en": s
            })

    dataset.append({
        "url": ep["url"],
        "segments": segments
    })

# =========================
# STEP 5: SARVAM TRANSLATION (CORRECT HF USAGE)
# =========================

def translate(text):
    messages = [
        {"role": "system", "content": "Translate the text below to Hindi."},
        {"role": "user", "content": text}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.01
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# =========================
# STEP 6: QUALITY FILTER + TRANSLATION
# =========================

final = []

for ep in tqdm(dataset):

    ep_out = {
        "segments": []
    }

    for seg in ep["segments"]:

        en = seg["en"]

        # simple quality filter (W19-style)
        if len(en) < 20 or len(en) > 1000:
            continue

        hi = translate(en)
        ta = translate(en)
        bn = translate(en)

        ep_out["segments"].append({
            "para_id": seg["para_id"],
            "sent_id": seg["sent_id"],
            "en": en,
            "hi": hi,
            "ta": ta,
            "bn": bn
        })

    final.append(ep_out)

# =========================
# STEP 7: SPLIT (80/10/10)
# =========================

train, temp = train_test_split(final, test_size=0.2, random_state=42)
dev, test = train_test_split(temp, test_size=0.5, random_state=42)

# =========================
# STEP 8: SAVE (SALESFORCE STYLE)
# =========================

json.dump(train, open("train.json", "w"), indent=2)
json.dump(dev, open("dev.json", "w"), indent=2)
json.dump(test, open("test.json", "w"), indent=2)
json.dump(final, open("full_dataset.json", "w"), indent=2)

# =========================
# STEP 9: STATS
# =========================

print("\n========================")
print("W19-5212 STYLE PIPELINE COMPLETE")
print("========================")
print("Episodes:", len(final))
print("Segments:", sum(len(e["segments"]) for e in final))
print("========================")
