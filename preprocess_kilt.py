# preprocess_kilt.py
# this is a faster way to preprocess kilt, but need to download kilt wiki knowledge source json first 
# streaming takes a lot longer through build_popqa_data.py so just preprocess the manually downloaded kilt wiki using this script
#
# converts:
#   raw file  ->  data/kilt_knowledgesource.json          (~29 GB)
#   output    ->  data/hf_popqa_kilt/kilt_corpus.jsonl    (~30 GB)

import json, os
from tqdm.auto import tqdm

RAW_FILE = "data/hf_popqa_kilt/kilt_knowledgesource.json"          # <-- adjust if different
DST_FILE = "data/hf_popqa_kilt/kilt_corpus2.jsonl"

os.makedirs(os.path.dirname(DST_FILE), exist_ok=True)

with open(RAW_FILE, encoding="utf-8") as fin, \
     open(DST_FILE, "w", encoding="utf-8") as fout:

    for line in tqdm(fin, desc="Flatten KILT"):
        art = json.loads(line)
        fout.write(
            json.dumps(
                {
                    "doc_id": art["_id"],              # unique id used later
                    "title" : art["wikipedia_title"],
                    "text"  : " ".join(art["text"])    # paragraphs -> one string
                },
                ensure_ascii=False
            )
            + "\n"
        )
print("Done →", DST_FILE)
