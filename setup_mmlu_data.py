# setup_mmlu_data.py
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from datasets import load_dataset
import json
import os

def setup_msmarco():
    """Download MS MARCO via BEIR and convert to pipeline format"""
    print("Downloading MS MARCO via BEIR...")
    
    # download MS MARCO using BEIR's download utility
    dataset = "msmarco"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, "beir/datasets")
    
    # load the data
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
    
    # convert to pipeline format
    os.makedirs("data/msmarco", exist_ok=True)
    
    with open("data/msmarco/msmarco_corpus.jsonl", "w") as f:
        for doc_id, doc_data in corpus.items():
            processed_doc = {
                "docid": doc_id,
                "title": doc_data.get("title", ""),
                "text": doc_data["text"]
            }
            f.write(json.dumps(processed_doc) + "\n")
    
    print(f"Converted {len(corpus)} MS MARCO documents")
    return len(corpus)

def setup_mmlu():
    """Download MMLU and convert to pipeline format"""
    print("Downloading MMLU...")
    
    # download MMLU from HuggingFace
    mmlu_dataset = load_dataset("cais/mmlu", "all")
    
    os.makedirs("data/mmlu", exist_ok=True)
    
    # convert test set
    with open("data/mmlu/mmlu_test.jsonl", "w") as f:
        for idx, item in enumerate(mmlu_dataset["test"]):
            processed_item = {
                "task": "qa",
                "query": item["question"],
                "choices": item["choices"],
                "answer": item["answer"],  # keep as int (0,1,2,3)
                "subject": item["subject"],
                "query_id": f"mmlu_test_{idx}"
            }
            f.write(json.dumps(processed_item) + "\n")
    
    # convert dev set (for few-shot)
    with open("data/mmlu/mmlu_dev.jsonl", "w") as f:
        for idx, item in enumerate(mmlu_dataset["dev"]):
            processed_item = {
                "task": "qa", 
                "query": item["question"],
                "choices": item["choices"],
                "answer": item["answer"],
                "subject": item["subject"],
                "query_id": f"mmlu_dev_{idx}"
            }
            f.write(json.dumps(processed_item) + "\n")
    
    test_size = len(mmlu_dataset["test"])
    dev_size = len(mmlu_dataset["dev"])
    print(f"Converted {test_size} MMLU test + {dev_size} dev examples")
    return test_size, dev_size

if __name__ == "__main__":
    print("Setting up MMLU evaluation data...")
    
    # setup both datasets
    corpus_size = setup_msmarco()
    test_size, dev_size = setup_mmlu()
    
    print(f"\n setup complete")
    print(f"MS MARCO corpus: {corpus_size:,} documents")
    print(f"MMLU test: {test_size:,} questions")
    print(f"MMLU dev: {dev_size:,} questions")
    print(f"\n ready to run eval_mmlu.py for evaluation") 