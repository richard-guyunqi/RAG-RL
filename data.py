#!/usr/bin/env python3
"""
data loading and preprocessing for ppo retriever training.
"""

import logging
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from utils import has_valid_ground_truth, is_textual

logger = logging.getLogger(__name__)


def load_ms_marco_with_splits(k_neg: int = 3, max_samples: int = None):
    """load ms marco with proper train/val/test splits for research"""
    
    def process_split(split_name, retry_count=3):
        for attempt in range(retry_count):
            try:
                logger.info(f"loading {split_name} split (attempt {attempt + 1}/{retry_count})")
                
                # try loading with streaming first for large datasets
                if split_name == "train" and max_samples:
                    ds = load_dataset("microsoft/ms_marco", "v2.1", split=f"train[:{max_samples}]", trust_remote_code=True)
                else:
                    ds = load_dataset("microsoft/ms_marco", "v2.1", split=split_name, trust_remote_code=True)
                
                logger.info(f"successfully loaded {split_name} split: {len(ds)} examples")
                
                def keep(ex):
                    try:
                        # first: check if ground truth is valid for training
                        answers = ex.get("answers", [])
                        if not has_valid_ground_truth(answers):
                            logger.debug("rejected example: no valid ground truth")
                            return None
                        
                        # handle ms marco v2.1 structure
                        is_selected = ex["passages"]["is_selected"]
                        passage_texts = ex["passages"]["passage_text"]
                        
                        pos = [i for i, s in enumerate(is_selected) if s == 1]
                        neg = [i for i, s in enumerate(is_selected) if s == 0][:k_neg]
                        
                        if not pos:
                            return None
                        
                        # extract passage text strings with validation
                        selected_passages = []
                        for i in pos + neg:
                            if i < len(passage_texts):
                                passage_text = passage_texts[i]
                                
                                # handle different data types
                                if isinstance(passage_text, list):
                                    passage_text = " ".join(str(item) for item in passage_text if str(item).strip())
                                elif isinstance(passage_text, dict):
                                    passage_text = " ".join(str(v) for v in passage_text.values() if str(v).strip())
                                elif not isinstance(passage_text, str):
                                    passage_text = str(passage_text)
                                
                                passage_text = passage_text.strip()
                                
                                if (len(passage_text) > 0 and 
                                    not passage_text.isdigit() and
                                    passage_text not in ["0", "1"] and
                                    is_textual(passage_text)):
                                    selected_passages.append(passage_text)
                        
                        if not selected_passages:
                            return None
                            
                        return dict(
                            query=str(ex["query"]),
                            answers=[str(ans) for ans in answers],
                            passages=selected_passages
                        )
                    except Exception as e:
                        logger.warning(f"error processing example: {e}")
                        return None
                
                # process examples safely
                def safe_keep(ex):
                    result = keep(ex)
                    if result is None:
                        return {
                            "query": "_REMOVE_ME_",
                            "answers": ["_REMOVE_ME_"],
                            "passages": ["_REMOVE_ME_"]
                        }
                    return result
                
                processed = ds.map(safe_keep)
                processed = processed.filter(lambda x: x["query"] != "_REMOVE_ME_")
                
                if len(processed) > 0:
                    columns_to_keep = ["query", "answers", "passages"]
                    columns_to_remove = [col for col in processed.column_names if col not in columns_to_keep]
                    if columns_to_remove:
                        processed = processed.remove_columns(columns_to_remove)
                
                logger.info(f"after filtering {split_name}: {len(processed)} examples")
                return processed
                
            except Exception as e:
                logger.error(f"error loading {split_name} split (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return None
                else:
                    import time
                    time.sleep(5)
    
    logger.info("loading ms marco dataset with enhanced error handling...")
    
    train_ds = process_split("train")
    val_ds = process_split("validation") 
    test_ds = process_split("test")
    
    return {
        "train": train_ds,
        "validation": val_ds, 
        "test": test_ds
    }


def load_ms_marco_subset(split: str = "train", max_samples: int = 10000, k_neg: int = 3):
    """load a small subset of ms marco for quick testing"""
    try:
        logger.info(f"loading ms marco subset: {max_samples} samples from {split}")
        
        ds = load_dataset("microsoft/ms_marco", "v2.1", split=f"{split}[:{max_samples}]", trust_remote_code=True)
        logger.info(f"loaded {len(ds)} examples")
        
        def keep(ex):
            try:
                answers = ex.get("answers", [])
                if not has_valid_ground_truth(answers):
                    return None
                
                is_selected = ex["passages"]["is_selected"]
                passage_texts = ex["passages"]["passage_text"]
                
                pos = [i for i, s in enumerate(is_selected) if s == 1]
                neg = [i for i, s in enumerate(is_selected) if s == 0][:k_neg]
                
                if not pos:
                    return None
                
                selected_passages = []
                for i in pos + neg:
                    if i < len(passage_texts):
                        passage_text = passage_texts[i]
                        
                        if isinstance(passage_text, list):
                            passage_text = " ".join(str(item) for item in passage_text if str(item).strip())
                        elif isinstance(passage_text, dict):
                            passage_text = " ".join(str(v) for v in passage_text.values() if str(v).strip())
                        elif not isinstance(passage_text, str):
                            passage_text = str(passage_text)
                        
                        passage_text = passage_text.strip()
                        
                        if (len(passage_text) > 0 and 
                            not passage_text.isdigit() and
                            passage_text not in ["0", "1"] and
                            is_textual(passage_text)):
                            selected_passages.append(passage_text)

                if not selected_passages:
                    return None

                return dict(
                    query=str(ex["query"]),
                    answers=[str(ans) for ans in answers],
                    passages=selected_passages
                )
            except Exception as e:
                logger.warning(f"error processing example: {e}")
                return None
        
        def safe_keep(ex):
            result = keep(ex)
            if result is None:
                return {
                    "query": "_REMOVE_ME_",
                    "answers": ["_REMOVE_ME_"],
                    "passages": ["_REMOVE_ME_"]
                }
            return result
        
        processed = ds.map(safe_keep)
        processed = processed.filter(lambda x: x["query"] != "_REMOVE_ME_")
        
        if len(processed) > 0:
            columns_to_keep = ["query", "answers", "passages"]
            columns_to_remove = [col for col in processed.column_names if col not in columns_to_keep]
            if columns_to_remove:
                processed = processed.remove_columns(columns_to_remove)
        
        logger.info(f"after filtering: {len(processed)} examples")
        return processed
        
    except Exception as e:
        logger.error(f"error loading ms marco subset: {e}")
        return None


def create_dataloaders(datasets, args):
    """create train/val/test dataloaders with proper sampling"""
    
    dataloaders = {}
    
    if datasets["train"] is not None:
        dataloaders["train"] = DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=Collate(),
            num_workers=2,
            pin_memory=True
        )
    
    if datasets["validation"] is not None:
        dataloaders["validation"] = DataLoader(
            datasets["validation"],
            batch_size=args.batch_size // 2,
            shuffle=False,
            collate_fn=Collate(),
            num_workers=2,
            pin_memory=True
        )
    
    if datasets["test"] is not None:
        dataloaders["test"] = DataLoader(
            datasets["test"],
            batch_size=args.batch_size // 2,
            shuffle=False,
            collate_fn=Collate(),
            num_workers=2,
            pin_memory=True
        )
    
    return dataloaders


class Collate:
    """enhanced collation ensuring all data is in correct format"""
    def __call__(self, batch):
        q = []
        p = []
        gt = []
        
        for x in batch:
            # handle queries - ensure string
            query = x["query"]
            if isinstance(query, list):
                query = " ".join(str(q) for q in query)
            q.append(str(query))
            
            # handle passages - ensure list of strings
            passages = x["passages"]
            if isinstance(passages, dict):
                passages = list(passages.values())
            elif not isinstance(passages, list):
                passages = [str(passages)]
            
            clean_passages = []
            for passage in passages:
                if isinstance(passage, list):
                    passage = " ".join(str(p) for p in passage)
                clean_passages.append(str(passage))
            p.append(clean_passages)
            
            # handle answers - ensure string
            answers = x["answers"]
            if isinstance(answers, list) and answers:
                answer = answers[0]
            elif isinstance(answers, str):
                answer = answers
            else:
                answer = ""
            
            if isinstance(answer, list):
                answer = " ".join(str(a) for a in answer)
            gt.append(str(answer))
        
        return q, p, gt 