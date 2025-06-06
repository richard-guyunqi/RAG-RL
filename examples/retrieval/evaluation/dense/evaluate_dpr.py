import logging
import os
import pathlib
import random

from beir import LoggingHandler, util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

dataset = "nfcorpus"

#### Download NFCorpus dataset and unzip the dataset
url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
data_path = util.download_and_unzip(url, out_dir)

#### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
# data folder would contain these files:
# (1) nfcorpus/corpus.jsonl  (format: jsonlines)
# (2) nfcorpus/queries.jsonl (format: jsonlines)
# (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

#### Dense Retrieval using Dense Passage Retriever (DPR) ####
# DPR implements a two-tower strategy i.e. encoding the query and document seperately.
# The DPR model was fine-tuned using dot-product (dot) function.

#########################################################
#### 1. Loading DPR model using SentenceTransformers ####
#########################################################
# You need to provide a ' [SEP] ' to seperate titles and passages in documents
# Ref: (https://www.sbert.net/docs/pretrained-models/dpr.html)

model = DRES(
    models.SentenceBERT(
        (
            "facebook-dpr-question_encoder-multiset-base",
            "facebook-dpr-ctx_encoder-multiset-base",
            " [SEP] ",
        ),
        batch_size=128,
    )
)

################################################################
#### 2. Loading Original HuggingFace DPR models by Facebook ####
################################################################
# If you do not have your saved model on Sentence Transformers,
# You can load HF-based DPR models in BEIR.
# No need to provide seperator token, the model handles automatically!

# model = DRES(models.DPR((
#     "facebook/dpr-question_encoder-multiset-base",
#     "facebook/dpr-ctx_encoder-multiset-base"), batch_size=128))

# You can also load similar trained DPR models available on Hugging Face.
# For eg. GermanDPR (https://deepset.ai/germanquad)

# model = DRES(models.DPR((
#     "deepset/gbert-base-germandpr-question_encoder",
#     "deepset/gbert-base-germandpr-ctx_encoder"), batch_size=128))


retriever = EvaluateRetrieval(model, score_function="dot")

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info(f"Retriever evaluation for k in: {retriever.k_values}")
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")

### If you want to save your results and runfile (useful for reranking)
results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
os.makedirs(results_dir, exist_ok=True)

#### Save the evaluation runfile & results
util.save_runfile(os.path.join(results_dir, f"{dataset}.run.trec"), results)
util.save_results(os.path.join(results_dir, f"{dataset}.json"), ndcg, _map, recall, precision, mrr)

#### Print top-k documents retrieved ####
top_k = 10

query_id, ranking_scores = random.choice(list(results.items()))
scores_sorted = sorted(ranking_scores.items(), key=lambda item: item[1], reverse=True)
logging.info(f"Query : {queries[query_id]}\n")

for rank in range(top_k):
    doc_id = scores_sorted[rank][0]
    # Format: Rank x: ID [Title] Body
    logging.info(f"Rank {rank + 1}: {doc_id} [{corpus[doc_id].get('title')}] - {corpus[doc_id].get('text')}\n")
