#!/usr/bin/env python3
"""
evaluation functions for model assessment.
"""

import logging
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from models import embed
from utils import reward_fn, extract_answer_from_generation

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_model_with_research_logging(retriever, critic, llm, tok_llm, dataloader, args, device, 
                                       split_name="validation", research_logger=None, step=0):
    """comprehensive evaluation with research logging"""
    
    retriever.eval()
    critic.eval()
    
    total_reward = 0
    total_samples = 0
    all_rewards = []
    all_queries = []
    all_preds = []
    all_gt = []
    
    # metrics for analysis
    retrieval_metrics = {
        'rewards': [],
        'value_predictions': [],
        'answer_lengths': [],
        'context_lengths': []
    }
    
    logger.info(f"evaluating on {split_name} set...")
    
    for batch_idx, (queries, passages, gt_answers) in enumerate(dataloader):
        B = len(queries)
        
        # get embeddings with smaller chunk size for evaluation
        q_emb = embed(retriever, queries, device, 8)  # use smaller chunk for eval
        
        # ensure passages data structure is correct
        corrected_passages = []
        passage_lengths = []
        flat_passages = []
        
        for i, plist in enumerate(passages):
            if isinstance(plist, dict):
                plist = list(plist.values())
            elif not isinstance(plist, list):
                plist = [str(plist)]
            
            # ensure all items in the list are strings
            clean_plist = []
            for passage in plist:
                if isinstance(passage, list):
                    passage = " ".join(str(p) for p in passage)
                clean_plist.append(str(passage))
            
            corrected_passages.append(clean_plist)
            passage_lengths.append(len(clean_plist))
            flat_passages.extend(clean_plist)
        
        # update passages to use corrected structure
        passages = corrected_passages
        
        # encode passages with smaller chunk size for evaluation
        p_emb = embed(retriever, flat_passages, device, 8)  # use smaller chunk for eval
        
        # reconstruct batch structure
        max_passages = max(passage_lengths) if passage_lengths else 1
        batch_p_emb = torch.zeros(B, max_passages, q_emb.size(-1), device=p_emb.device)
        
        start_idx = 0
        for i, length in enumerate(passage_lengths):
            end_idx = start_idx + length
            batch_p_emb[i, :length] = p_emb[start_idx:end_idx]
            start_idx = end_idx
        
        p_emb = batch_p_emb
        
        # retrieval (deterministic for evaluation)
        sim = torch.einsum("bd,bkd->bk", q_emb, p_emb)
        
        # create mask for valid passages
        mask = torch.zeros(B, max_passages, device=sim.device)
        for i, length in enumerate(passage_lengths):
            mask[i, :length] = 1
        
        # apply mask to similarities
        sim = sim.masked_fill(mask == 0, -1e9)
        
        # deterministic action selection for evaluation (unless explicitly stochastic)
        if args.stochastic_actions:
            action_probs = torch.softmax(sim, dim=1)
            action_probs = action_probs.masked_fill(mask == 0, 0)
            action_probs = action_probs / (action_probs.sum(dim=1, keepdim=True) + 1e-7)
            actions = torch.multinomial(action_probs, args.retrieve_top_k)
        else:
            actions = sim.topk(args.retrieve_top_k, dim=1).indices
        
        # generate llm responses (passages are now guaranteed to be lists)
        ctx = []
        for b in range(B):
            selected_passages = []
            for action_idx in actions[b]:
                # convert tensor to integer and ensure bounds checking
                idx = action_idx.item()  # convert cuda tensor to python int
                if idx < len(passages[b]):
                    passage = passages[b][idx]
                    # extra safety: ensure passage is a string
                    if isinstance(passage, list):
                        passage = " ".join(str(p) for p in passage)
                    elif not isinstance(passage, str):
                        passage = str(passage)
                    selected_passages.append(passage)
                else:
                    # fallback to first passage if index is out of bounds
                    fallback = passages[b][0] if passages[b] else "No passage available"
                    if isinstance(fallback, list):
                        fallback = " ".join(str(p) for p in fallback)
                    elif not isinstance(fallback, str):
                        fallback = str(fallback)
                    selected_passages.append(fallback)
            
            # ensure all passages are strings
            selected_passages = [str(p) for p in selected_passages]
            ctx.append("\n".join(selected_passages))
            
        # create prompts based on model type
        if args.llm_type == "seq2seq":
            # for t5/flan-t5: simple question with context
            prompts = [f"Answer this question using the context: {q}\n\nContext: {c}" for q, c in zip(queries, ctx)]
        else:
            # for causal models (dialogpt, gpt): conversational format
            prompts = [f"Context: {c}\n\nQ: {q}\nA:" for q, c in zip(queries, ctx)]
        
        with torch.no_grad():
            toks = tok_llm(prompts, return_tensors="pt", padding=True, 
                          truncation=True, max_length=512).to(device)
            # use greedy decoding for consistent evaluation
            outs = llm.generate(**toks, max_new_tokens=32, do_sample=False, 
                               pad_token_id=tok_llm.eos_token_id)
        
        preds = tok_llm.batch_decode(outs, skip_special_tokens=True)
        preds_clean = [extract_answer_from_generation(pred, prompt, args.llm_type) 
                       for pred, prompt in zip(preds, prompts)]
        
        # compute sbert cosine similarity rewards
        rewards = reward_fn(preds_clean, gt_answers).to(device)
        values = critic(q_emb, p_emb)
        
        # accumulate metrics
        total_reward += rewards.sum().item()
        total_samples += B
        all_rewards.extend(rewards.cpu().tolist())
        all_queries.extend(queries)
        all_preds.extend(preds_clean)
        all_gt.extend(gt_answers)
        
        retrieval_metrics['rewards'].extend(rewards.cpu().tolist())
        retrieval_metrics['value_predictions'].extend(values.cpu().tolist())
        retrieval_metrics['answer_lengths'].extend([len(pred.split()) for pred in preds_clean])
        retrieval_metrics['context_lengths'].extend([len(c.split()) for c in ctx])
        
        # log progress
        if batch_idx % 20 == 0:
            logger.info(f"eval batch {batch_idx}: avg_sbert_similarity={rewards.mean().item():.4f}")
    
    # compute final metrics - handle empty dataset case
    if total_samples == 0:
        logger.warning(f"no samples in {split_name} dataset - skipping evaluation")
        return {
            f'{split_name}/reward_mean': 0.0,
            f'{split_name}/reward_std': 0.0,
            f'{split_name}/reward_median': 0.0,
            f'{split_name}/samples_evaluated': 0
        }, {}
    
    avg_reward = total_reward / total_samples
    reward_std = np.std(all_rewards)
    
    eval_metrics = {
        f'{split_name}/reward_mean': avg_reward,
        f'{split_name}/reward_std': reward_std,
        f'{split_name}/reward_median': np.median(all_rewards),
        f'{split_name}/value_mean': np.mean(retrieval_metrics['value_predictions']),
        f'{split_name}/answer_length_mean': np.mean(retrieval_metrics['answer_lengths']),
        f'{split_name}/context_length_mean': np.mean(retrieval_metrics['context_lengths']),
        f'{split_name}/samples_evaluated': total_samples
    }
    
    # research logging for evaluation
    if research_logger:
        research_logger.log_evaluation(
            split_name=split_name,
            eval_metrics=eval_metrics,
            queries=all_queries[:50],  # log subset for analysis
            preds=all_preds[:50],
            gt_answers=all_gt[:50],
            rewards=torch.tensor(all_rewards[:50]),
            step=step
        )
    
    logger.info(f"{split_name.capitalize()} Results:")
    logger.info(f"  avg sbert similarity: {avg_reward:.4f} ± {reward_std:.4f}")
    logger.info(f"  samples: {total_samples}")
    
    return eval_metrics, retrieval_metrics 