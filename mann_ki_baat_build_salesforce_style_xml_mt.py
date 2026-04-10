!pip install requests beautifulsoup4 tqdm datasets regex
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import random
import os
import re
from datasets import Dataset, DatasetDict

random.seed(42)

BASE = "https://www.pmindia.gov.in"
HI_INDEX = "https://www.pmindia.gov.in/hi/tag/mann-ki-baat/"
EN_INDEX = "https://www.pmindia.gov.in/en/tag/mann-ki-baat/"

HEADERS = {"User-Agent": "Mozilla/5.0"}


# ---------------------------------------------------
# STEP 1 — Fetch HTML
# ---------------------------------------------------

def fetch(url):
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.text


# ---------------------------------------------------
# STEP 2 — Collect ALL episode links (pagination)
# ---------------------------------------------------

def collect_links(index_url, max_pages=25):

    links = set()

    for page in range(1, max_pages + 1):

        url = index_url + f"page/{page}/"
        html = fetch(url)
        soup = BeautifulSoup(html, "html.parser")

        for a in soup.select("h3.entry-title a"):

            href = a["href"]

            if "mann-ki-baat" in href.lower():
                links.add(href)

    return list(links)


# ---------------------------------------------------
# STEP 3 — Extract clean paragraphs
# ---------------------------------------------------

def extract_paragraphs(url):

    html = fetch(url)
    soup = BeautifulSoup(html, "html.parser")

    content = soup.find("div", class_="entry-content") \
              or soup.find("div", class_="td-post-content")

    if content is None:
        return []

    paras = []

    for p in content.find_all("p"):
        txt = p.get_text(" ", strip=True)

        if len(txt) > 25:
            paras.append(txt)

    return paras


# ---------------------------------------------------
# STEP 4 — Align Hindi ↔ English by slug
# ---------------------------------------------------

def align_docs(hi_links, en_links):

    pairs = []

    hi_map = {l.split("/")[-2]: l for l in hi_links}
    en_map = {l.split("/")[-2]: l for l in en_links}

    common = set(hi_map.keys()) & set(en_map.keys())

    for slug in tqdm(common):

        hi_paras = extract_paragraphs(hi_map[slug])
        en_paras = extract_paragraphs(en_map[slug])

        n = min(len(hi_paras), len(en_paras))

        for i in range(n):

            pairs.append({
                "id": f"{slug}_{i}",
                "en": en_paras[i],
                "hi": hi_paras[i]
            })

    return pairs


# ---------------------------------------------------
# STEP 5 — Inject XML tags (Salesforce-style)
# ---------------------------------------------------

def inject_xml(text):

    # mimic inline tags like <ph>
    # rule-based tagging (IMPORTANT for realism)

    # tag numbers
    text = re.sub(r"\b\d+\b", r"<ph>\g<0></ph>", text)

    # tag keywords
    keywords = ["India", "भारत", "Modi", "सरकार"]

    for kw in keywords:
        text = text.replace(kw, f"<ph>{kw}</ph>")

    return text


# ---------------------------------------------------
# STEP 6 — Build Salesforce JSON format
# ---------------------------------------------------

def build_json(data, split, lang):

    out = {}

    for i, row in enumerate(data):

        idx = f"salesforce_xml:enhi_{split}_{i:010d}"

        text = inject_xml(row[lang])

        out[idx] = text

    return {
        "lang": lang,
        "type": "source" if lang == "en" else "target",
        "text": out
    }


# ---------------------------------------------------
# STEP 7 — Split dataset
# ---------------------------------------------------

def split_data(data):

    random.shuffle(data)

    n = len(data)

    train = data[: int(0.8*n)]
    dev   = data[int(0.8*n): int(0.9*n)]
    test  = data[int(0.9*n):]

    return train, dev, test


# ---------------------------------------------------
# STEP 8 — Save JSON files
# ---------------------------------------------------

def save_json(train, dev, test):

    os.makedirs("dataset/enhi", exist_ok=True)

    splits = {"train": train, "dev": dev, "test": test}

    for split, data in splits.items():

        en_json = build_json(data, split, "en")
        hi_json = build_json(data, split, "hi")

        with open(f"dataset/enhi/enhi_en_{split}.json","w",encoding="utf8") as f:
            json.dump(en_json, f, indent=2, ensure_ascii=False)

        with open(f"dataset/enhi/enhi_hi_{split}.json","w",encoding="utf8") as f:
            json.dump(hi_json, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------
# STEP 9 — Build HuggingFace dataset
# ---------------------------------------------------

def build_hf():

    def load_split(split):

        en = json.load(open(f"dataset/enhi/enhi_en_{split}.json"))
        hi = json.load(open(f"dataset/enhi/enhi_hi_{split}.json"))

        ids = list(en["text"].keys())

        data = []

        for i in ids:
            data.append({
                "id": i,
                "translation": {
                    "en": en["text"][i],
                    "hi": hi["text"][i]
                }
            })

        return Dataset.from_list(data)

    ds = DatasetDict({
        "train": load_split("train"),
        "validation": load_split("dev"),
        "test": load_split("test")
    })

    ds.save_to_disk("hf_dataset")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

def main():

    print("Collecting links...")

    hi_links = collect_links(HI_INDEX)
    en_links = collect_links(EN_INDEX)

    print("HI:", len(hi_links), "EN:", len(en_links))

    print("Aligning...")

    pairs = align_docs(hi_links, en_links)

    print("Total segments:", len(pairs))

    train, dev, test = split_data(pairs)

    print("Train:", len(train))
    print("Dev:", len(dev))
    print("Test:", len(test))

    save_json(train, dev, test)

    build_hf()

    print("DONE")


if __name__ == "__main__":
    main()
