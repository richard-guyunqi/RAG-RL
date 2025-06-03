# Evaluation Pipeline Configuration README

This doc provides a comprehensive overview of all configuration arguments for the PopQA and MMLU evaluation pipelines

---

## Start

### PopQA Evaluation
```bash
python eval_popqa.py --query_encoder intfloat/e5-base-v2 --model_name_or_path google/flan-t5-large
```

### MMLU Evaluation  
```bash
python eval_mmlu.py --query_encoder intfloat/e5-base-v2 --model_name_or_path google/flan-t5-large
```

---

## PopQA Evaluation (`eval_popqa.py`)

### PopQA-Specific Arguments

| Argument | Default | Possible Values | Description |
|----------|---------|-----------------|-------------|
| `output_dir` | `"data/results/popqa_hf_kilt"` | Any valid path | Output directory for results |
| `eval_data` | `ensure_popqa()` | File path (.jsonl) | Auto-downloads PopQA test set to `data/hf_popqa_kilt/popqa_test.jsonl` |
| `corpus` | `ensure_kilt_corpus()` | File path (.jsonl) | Auto-downloads KILT corpus to `data/hf_popqa_kilt/kilt_corpus2.jsonl` |
| `few_shot` | `15` | 0-50+ (integer) | Number of few-shot examples in prompts |
| `hits` | `10` | 1-1000+ (integer) | Number of documents retrieved per query |
| `key_num` | `3` | 1-20+ (integer) | Number of documents used in final prompt |
| `key_template` | `"{title} {text}"` | Format string with {field} | How to format corpus documents |
| `key_max_length` | `128` | 32-512+ (integer) | Maximum tokens per document |
| `metrics` | `["collate_key"]` | `["collate_key", "mrr", "recall", "ndcg"]` | Evaluation metrics to compute |
| `save_to_output` | `True` | `True`, `False` | Save results to output_dir |
| `log_path` | `"data/results/popqa/popqa.log"` | Any valid path | Log file path |

### PopQA Generation Arguments

| Argument | Default | Possible Values | Description |
|----------|---------|-----------------|-------------|
| `max_new_tokens` | `16` | 1-100+ (integer) | Maximum tokens to generate |
| `eos_token_id` | `13` | Any token ID (integer) | End-of-sequence token ID |

---

## MMLU Evaluation (`eval_mmlu.py`)

### MMLU-Specific Arguments

| Argument | Default | Possible Values | Description |
|----------|---------|-----------------|-------------|
| `output_dir` | `"data/results/mmlu"` | Any valid path | Output directory for results |
| `eval_data` | `"data/mmlu/mmlu_test.jsonl"` | File path (.jsonl) | Path to MMLU test questions |
| `corpus` | `"data/msmarco/msmarco_corpus.jsonl"` | File path (.jsonl) | Path to MS MARCO corpus |
| `train_data` | `"data/mmlu/mmlu_dev.jsonl"` | File path (.jsonl) | Path to MMLU dev set (for few-shot) |
| `few_shot` | `0` | 0-10+ (integer) | Number of few-shot examples (0 = zero-shot) |
| `hits` | `10` | 1-1000+ (integer) | Number of documents retrieved per query |
| `key_num` | `3` | 1-20+ (integer) | Number of documents used in final prompt |
| `key_template` | `"{title} {text}"` | Format string with {field} | How to format corpus documents |
| `key_max_length` | `128` | 32-512+ (integer) | Maximum tokens per document |
| `lm_batch_size` | `2` | 1-64+ (integer) | Batch size for language model |
| `metrics` | `["collate_key"]` | `["collate_key", "mrr", "recall", "ndcg"]` | Evaluation metrics to compute |
| `save_to_output` | `True` | `True`, `False` | Save results to output_dir |
| `log_path` | `"data/results/mmlu/mmlu.log"` | Any valid path | Log file path |

---

## Common Arguments

Both evaluation scripts inherit from `LMArgs` and `RetrievalArgs`, providing these shared configurations:

### Language Model Arguments (`LMArgs`)

| Argument | Default | Possible Values | Description |
|----------|---------|-----------------|-------------|
| `model_name_or_path` | `"meta-llama/Llama-2-7b-chat-hf"` | HuggingFace model ID or local path | HuggingFace model identifier |
| `padding_side` | `"left"` | `"left"`, `"right"` | Tokenizer padding side |
| `truncation_side` | `"right"` | `"left"`, `"right"` | Tokenizer truncation side |
| `context_max_length` | `2048` | 512-32768+ (integer) | Maximum context length |
| `add_position_ids` | `False` | `True`, `False` | Add position embeddings |
| `lm_dtype` | `"bf16"` | `"fp16"`, `"bf16"`, `"fp32"`, `"auto"` | Model precision |
| `lm_device_map` | `None` | `"auto"`, `"cpu"`, `None`, custom dict | Device mapping strategy |
| `lm_batch_size` | `2` | 1-64+ (integer) | Evaluation batch size |
| `cpu` | `False` | `True`, `False` | Force CPU usage |
| `add_llama_inst` | `False` | `True`, `False` | Add Llama2-chat instructions |

### Retrieval Arguments (`RetrievalArgs`)

#### Dense Retrieval Settings
| Argument | Default | Possible Values | Description |
|----------|---------|-----------------|-------------|
| `retrieval_method` | `"dense"` | `"dense"`, `"bm25"`, `"no"`, `"random"` | Retrieval method |
| `query_encoder` | `"BAAI/bge-base-en"` | HuggingFace model ID or local path | Query encoder model |
| `key_encoder` | `"BAAI/bge-base-en"` | HuggingFace model ID or local path | Document encoder model |
| `add_instruction` | `True` | `True`, `False` | Use task-specific instructions |
| `version` | `"bge"` | `"bge"`, `"e5"`, `"llm-embedder"`, `"gtr-t5-rl"`, `"instructor"` | Instruction set version |
| `query_max_length` | `256` | 32-1024+ (integer) | Maximum query length |
| `key_max_length` | `256` | 32-1024+ (integer) | Maximum document length |
| `dense_metric` | `"cos"` | `"cos"`, `"ip"`, `"l2"` | Similarity metric (cosine, inner product, L2) |
| `hits` | `200` | 1-10000+ (integer) | Documents to retrieve (overridden by task-specific) |
| `batch_size` | `1000` | 1-5000+ (integer) | Encoding/indexing batch size |
| `faiss_index_factory` | `"Flat"` | `"Flat"`, `"IVFFlat"`, `"HNSW32"`, etc. | FAISS index type |
| `dtype` | `"fp16"` | `"fp16"`, `"fp32"`, `"bf16"` | Retrieval model precision |

#### BM25 Settings (when `retrieval_method="bm25"`)
| Argument | Default | Possible Values | Description |
|----------|---------|-----------------|-------------|
| `k1` | `0.82` | 0.1-5.0 (float) | BM25 k1 parameter |
| `b` | `0.68` | 0.0-1.0 (float) | BM25 b parameter |
| `threads` | `32` | 1-128+ (integer) | Number of threads for indexing |

#### Retrieval Settings
| Argument | Default | Possible Values | Description |
|----------|---------|-----------------|-------------|
| `corpus` | `None` | File path (.jsonl) | Path to corpus file |
| `eval_data` | `None` | File path (.jsonl) | Path to evaluation data |
| `metrics` | `["mrr", "recall", "ndcg"]` | `["mrr", "recall", "ndcg", "collate_key"]` | Retrieval metrics |
| `cutoffs` | `[1, 5, 10, 100]` | List of integers | Metric cutoff values |
| `save_result` | `True` | `True`, `False` | Save retrieval results |
| `load_result` | `False` | `True`, `False` | Load cached results |

---

## Argument Value Details

### FAISS Index Types (`faiss_index_factory`)

| Index Type | Use Case | Memory | Speed | 
|-----------|----------|---------|-------|
| `"Flat"` | Small corpus (<1M docs) | High | Fast 
| `"IVF1024,Flat"` | Medium corpus (1M-10M) | Medium | Medium 
| `"HNSW32"` | Large corpus (10M+) | Medium | Fast 
| `"IVF4096,PQ32"` | Very large corpus | Low | Fast

### Precision Settings (`lm_dtype`, `dtype`)

| Precision | Memory Usage | Speed | Quality |
|-----------|--------------|-------|---------|
| `"fp32"` | High | Slow | Best 
| `"bf16"` | Medium | Fast | Good 
| `"fp16"` | Medium | Fast | Good 
| `"auto"` | Variable | Variable | Variable 

### Similarity Metrics (`dense_metric`)

| Metric | Formula | Use Case | 
|--------|---------|----------|
| `"cos"` | cosine similarity | Most common, angle-based 
| `"ip"` | inner product | When vectors already normalized 
| `"l2"` | negative L2 distance | Euclidean distance 

---

## Configuration Examples

### Basic Retriever Comparison

```bash
# E5 baseline
python eval_popqa.py \
  --query_encoder intfloat/e5-base-v2 \
  --model_name_or_path google/flan-t5-large \
  --version e5

# BGE baseline  
python eval_popqa.py \
  --query_encoder BAAI/bge-base-en-v1.5 \
  --model_name_or_path google/flan-t5-large \
  --version bge

# GTR-T5 baseline
python eval_popqa.py \
  --query_encoder sentence-transformers/gtr-t5-xl \
  --model_name_or_path google/flan-t5-large \
  --version llm-embedder

# RL fine-tuned GTR-T5
python eval_popqa.py \
  --query_encoder ./path/to/your-gtr-t5-rl \
  --model_name_or_path google/flan-t5-large \
  --version gtr-t5-rl
```

### MMLU with Few-Shot Learning

```bash
python eval_mmlu.py \
  --query_encoder intfloat/e5-base-v2 \
  --model_name_or_path google/flan-t5-large \
  --few_shot 5 \
  --lm_batch_size 4 \
  --hits 20 \
  --key_num 5
```

### No Retrieval Baseline

```bash
python eval_popqa.py \
  --retrieval_method no \
  --model_name_or_path google/flan-t5-large
```


### Fast Testing Configuration

```bash
# Small corpus for quick testing
python eval_mmlu.py \
  --corpus data/test/small_corpus.jsonl \
  --eval_data data/test/small_test.jsonl \
  --lm_batch_size 8 \
  --hits 5 \
  --key_num 2
```

### Performance Optimization

```bash
# High throughput settings
python eval_popqa.py \
  --lm_batch_size 8 \
  --batch_size 2000 \
  --lm_dtype fp16 \
  --lm_device_map auto
```

---

## Instruction Templates

The `--version` flag controls which instruction templates are used:

- **`bge`**: Minimal instructions for BGE models
- **`e5`**: Query/passage prefixes for E5 models  
- **`llm-embedder`**: Detailed task-specific instructions
- **`gtr-t5-rl`**: Custom instructions for your RL model (same as llm-embedder)

---

## Data Preparation

### PopQA
- **Auto-download**: Uses `ensure_popqa()` and `ensure_kilt_corpus()` 
- **Manual**: Place files at `data/hf_popqa_kilt/`

### MMLU
- **Setup script**: Run `python setup_mmlu_data.py` 
- **Manual**: Place files at `data/mmlu/` and `data/msmarco/`

---

## Output Structure

Results are saved with this structure:
```
data/results/
├── popqa_hf_kilt/
│   ├── {model-name}/
│   │   ├── result.json          # Final scores
│   │   └── retrieval_results/   # Intermediate files
│   └── popqa.log               # Execution log
└── mmlu/
    ├── {model-name}/
    │   ├── result.json
    │   └── retrieval_results/
    └── mmlu.log
``` 