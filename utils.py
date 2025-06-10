#!/usr/bin/env python3
"""
utility functions for ppo retriever training.
"""

import random
import re
import logging
from typing import List
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# setup logging
logger = logging.getLogger(__name__)

# regex for text validation
ALPHA_RE = re.compile(r"[A-Za-z]")

# global sbert model for reward computation
sbert = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


def set_seed(seed: int = 42):
    """set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)


def validate_ground_truth(answers):
    """clean and validate ground truth answers"""
    if not answers:
        return "No answer available"
    
    if isinstance(answers, list):
        # take first non-empty answer
        for ans in answers:
            if ans and str(ans).strip() and str(ans).strip().lower() not in ["no answer present", "none", "n/a"]:
                return str(ans).strip()
    elif isinstance(answers, str):
        ans = answers.strip()
        if ans and ans.lower() not in ["no answer present", "none", "n/a"]:
            return ans
    
    return "No answer available"


def has_valid_ground_truth(answers):
    """check if ground truth is valid for training/evaluation (returns true/false)
    
    keeps both real answers and valid 'no answer present' cases.
    only filters out empty/corrupted data.
    """
    if not answers:
        return False
    
    if isinstance(answers, list):
        # filter out completely empty lists or lists with only empty strings
        valid_answers = [str(ans).strip() for ans in answers if str(ans).strip()]
        if not valid_answers:
            return False
        
        # accept both real answers and "no answer present" (which is a valid training case)
        return True
        
    elif isinstance(answers, str):
        clean_ans = answers.strip()
        # only reject truly empty strings
        return bool(clean_ans)
    
    return False


def is_textual(passage: str) -> bool:
    """return true if passage looks like real text with better validation"""
    if not isinstance(passage, str):
        return False
    
    passage = passage.strip()
    
    # reject empty or very short passages
    if len(passage) < 10:
        return False
    
    # reject passages that are just numbers/digits/spaces (like "0 0 0 0 0 1 0 0 0 0")
    if passage.replace(" ", "").replace("0", "").replace("1", "") == "":
        return False
    
    # reject passages that are mostly urls without content
    if passage.startswith("http") and len(passage.split()) < 10:
        return False
    
    # reject passages that are mostly just urls
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    urls = re.findall(url_pattern, passage)
    if len(urls) > 3 and len(passage.split()) - len(urls) < 5:
        return False
    
    # must contain alphabetic characters
    if not bool(ALPHA_RE.search(passage)):
        return False
    
    # must have reasonable word count
    words = passage.split()
    if len(words) < 5:
        return False
    
    # reject passages that are mostly punctuation or special characters
    alpha_chars = sum(1 for c in passage if c.isalpha())
    if alpha_chars < len(passage) * 0.3:  # at least 30% alphabetic
        return False
    
    return True


def extract_answer_from_generation(generated_text: str, prompt: str, model_type: str = "causal") -> str:
    """extract answer with better parsing for both causal and seq2seq models"""
    
    if model_type == "seq2seq":
        # for seq2seq models like t5/flan-t5, the output is just the answer
        answer = generated_text.strip()
        
        # basic cleanup for seq2seq models
        if answer.lower().startswith("answer:"):
            answer = answer[7:].strip()
        
        # remove common prefixes
        prefixes_to_remove = ["the answer is", "based on the context", "according to"]
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # handle very short or empty answers
        if len(answer) < 3 or answer.lower() in ["no", "none", "n/a", ""]:
            return "No answer generated"
        
        return answer
    
    else:
        # original logic for causal models (dialogpt, etc.)
        # find the a: marker and extract everything after it (matches prompt format)
        if "A:" in generated_text:
            parts = generated_text.split("A:")
            if len(parts) > 1:
                answer = parts[-1].strip()  # take last part after a:
            else:
                answer = generated_text.strip()
        else:
            # fallback: remove prompt completely
            answer = generated_text.replace(prompt, "").strip()
        
        # clean up the answer
        answer = answer.split('\n')[0].strip()  # take first line
        
        # remove repeated question words
        query_words = set(prompt.lower().split())
        answer_words = answer.lower().split()
        
        # if answer is just repeating the question, mark as empty
        if len(set(answer_words) & query_words) > len(answer_words) * 0.7:
            return "No answer generated"
        
        if not answer or answer.lower().startswith("question"):
            return "No answer generated"
            
        return answer


@torch.no_grad()
def reward_fn(pred: List[str], ref: List[str]) -> torch.Tensor:
    """compute sbert cosine similarity rewards between predictions and references"""
    if not pred or not ref:
        return torch.zeros(max(len(pred), len(ref)))
    
    # check for non-answers
    filtered_pred = []
    penalties = []
    
    for p in pred:
        if (p.lower().startswith("question") or 
            p.lower().startswith("no answer") or 
            len(p.split()) < 2):
            # less harsh penalty for non-answers to allow learning
            filtered_pred.append("No valid answer provided")
            penalties.append(0.3)  # increased from 0.1 to allow learning
        else:
            filtered_pred.append(p)
            penalties.append(1.0)  # no penalty
    
    # compute sbert similarity
    emb_pred = sbert.encode(filtered_pred, convert_to_tensor=True, normalize_embeddings=True)
    emb_ref = sbert.encode(ref, convert_to_tensor=True, normalize_embeddings=True)
    
    if emb_pred.device != emb_ref.device:
        emb_ref = emb_ref.to(emb_pred.device)
    
    similarities = util.cos_sim(emb_pred, emb_ref).diagonal()
    
    # apply penalties
    penalties_tensor = torch.tensor(penalties, device=similarities.device)
    return similarities * penalties_tensor


def mb_indices(n: int, b: int):
    """generate minibatch indices with shuffling"""
    perm = np.random.permutation(n)
    for i in range(0, n, b):
        yield perm[i:i+b]


def verify_gradients(tensor, name="tensor"):
    """enhanced debug function to verify gradient flow"""
    if tensor.requires_grad:
        logger.debug(f"tensor {name} has gradients enabled")
        if tensor.grad_fn is not None:
            logger.debug(f"tensor {name} has gradient function: {tensor.grad_fn}")
        return True
    else:
        logger.error(f"tensor {name} missing gradients! shape: {tensor.shape}")
        return False


def monitor_gradients(model, step, logger_tb=None):
    """monitor gradient flow during training"""
    grad_norms = {}
    param_norms = {}
    total_grad_norm = 0
    total_param_norm = 0
    params_with_grads = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            param_norm = param.norm().item()
            param_norms[name] = param_norm
            total_param_norm += param_norm
            
            if param.grad is not None:
                params_with_grads += 1
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                total_grad_norm += grad_norm
            else:
                grad_norms[name] = 0.0
    
    # log summary statistics
    if logger_tb:
        logger_tb.add_scalar("gradients/total_grad_norm", total_grad_norm, step)
        logger_tb.add_scalar("gradients/total_param_norm", total_param_norm, step)
        logger_tb.add_scalar("gradients/params_with_grads_ratio", params_with_grads / max(total_params, 1), step)
        logger_tb.add_scalar("gradients/avg_grad_norm", total_grad_norm / max(params_with_grads, 1), step)
        logger_tb.add_scalar("gradients/avg_param_norm", total_param_norm / max(total_params, 1), step)
    
    # log detailed gradients for debugging (every 100 steps)
    if step % 100 == 0:
        logger.debug(f"gradient monitoring at step {step}:")
        logger.debug(f"  parameters with gradients: {params_with_grads}/{total_params}")
        logger.debug(f"  total gradient norm: {total_grad_norm:.6f}")
        logger.debug(f"  average gradient norm: {total_grad_norm / max(params_with_grads, 1):.6f}")
        
        # log worst performers (parameters with very small gradients)
        small_grad_threshold = 1e-8
        small_grad_params = [name for name, norm in grad_norms.items() if norm < small_grad_threshold]
        if small_grad_params:
            logger.debug(f"  parameters with very small gradients: {len(small_grad_params)}")
            for name in small_grad_params[:5]:  # show first 5
                logger.debug(f"    {name}: {grad_norms[name]:.2e}")
    
    return {
        'total_grad_norm': total_grad_norm,
        'params_with_grads': params_with_grads,
        'total_params': total_params,
        'avg_grad_norm': total_grad_norm / max(params_with_grads, 1)
    } 