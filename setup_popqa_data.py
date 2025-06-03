# setup_popqa_data.py
"""
Create local JSONL files that pipeline expects

Paths:
  data/hf_popqa_kilt/popqa_test.jsonl    PopQA test split 
  data/hf_popqa_kilt/kilt_corpus.jsonl   KILT Wikipedia

functions are idempotent, if the file already exists they just
return the path
"""

import os, json, uuid, hashlib, itertools, datetime, typing as T
from datasets import load_dataset
from tqdm.auto import tqdm

ROOT = "data/hf_popqa_kilt"
os.makedirs(ROOT, exist_ok=True)

POPQA_FILE  = f"{ROOT}/popqa_test.jsonl"
CORPUS_FILE = f"{ROOT}/kilt_corpus2.jsonl"
LOG_FILE    = f"{ROOT}/build.log"


# helper 
def _sha1(path: str, max_bytes: int | None = 1_000_000) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
            if max_bytes:
                max_bytes -= len(chunk)
                if max_bytes <= 0:
                    break
    return h.hexdigest()[:10]


def _write_log(msg: str) -> None:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    with open(LOG_FILE, "a", encoding="utf-8") as f:   # add encoding
        f.write(f"[{ts}] {msg}\n")
    print(msg)


#  PopQA download and preprocess for pipeline
def ensure_popqa() -> str:
    if os.path.exists(POPQA_FILE):
        _write_log(f"PopQA already present  –  {POPQA_FILE}  (sha1={_sha1(POPQA_FILE)})")
        return POPQA_FILE

    _write_log("Downloading PopQA (test split) from HF …")
    ds = load_dataset("akariasai/PopQA", split="test")
    with open(POPQA_FILE, "w", encoding="utf-8") as f:
        for row in tqdm(ds, desc="Writing PopQA jsonl"):
            obj = {
                "task"            : "qa", 
                "query"           : row["question"],
                "query_id"        : f"popqa-{row['id']}",
                "prop_id"         : int(row["prop_id"]),
                "possible_answers": row["possible_answers"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    _write_log(f"PopQA written  –  {POPQA_FILE}  (records={len(ds)}, sha1={_sha1(POPQA_FILE)})")
    return POPQA_FILE


# KILT wiki corpus 
# if already present, then returns path to the data
# if not, streams articles from the corpus and flattens them into 3-column JSONL format for pipeline retrieval
def ensure_kilt_corpus(max_docs: int | None = None) -> str:
    """
    Streams articles from KILT Wikipedia dump and flattens them into the
    3 column JSONL (doc_id,title,text) format expected by retrieval_main().

    Set `max_docs` to a small int for quick debugging.
    """
    if os.path.exists(CORPUS_FILE):
        _write_log(f"KILT corpus already present  –  {CORPUS_FILE}  (sha1={_sha1(CORPUS_FILE)})")
        return CORPUS_FILE

    _write_log("Streaming KILT‑Wikipedia from HF...")
    stream = load_dataset(
        "facebook/kilt_wikipedia", "2019-08-01",
        split="full", streaming=True,
        trust_remote_code=True
    )

    count = 0
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for art in tqdm(stream, desc="Writing KILT corpus", total=max_docs):
            if max_docs and count >= max_docs:
                break
            f.write(json.dumps({
                "doc_id": art["kilt_id"],
                "title" : art["wikipedia_title"],
                "text"  : " ".join(art["text"]["paragraph"]),
            }, ensure_ascii=False) + "\n")
            count += 1

    _write_log(f"KILT corpus written  –  {CORPUS_FILE}  (records={count}, sha1={_sha1(CORPUS_FILE)})")
    return CORPUS_FILE