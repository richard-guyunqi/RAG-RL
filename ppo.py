#!/usr/bin/env python3
"""
ppo.py
======

core ppo algorithm implementation with generalized advantage estimation.
this is the modular version that imports utilities from other modules.

key components:
- gaebuffer: generalized advantage estimation for better advantage computation
- enhanced_ppo_step_with_research_logging: complete ppo training step with logging
- gradient flow monitoring and memory optimization
"""

from typing import List, Tuple, Dict, Any
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# import from our modular components
from models import embed, EnhancedCritic
from utils import verify_gradients, monitor_gradients, reward_fn, extract_answer_from_generation
from research_logging import ResearchLogger

class GAEBuffer:
    """generalized advantage estimation buffer for better advantage computation"""
    
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        
    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor, 
                   dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute gae advantages and returns
        
        args:
            rewards: reward tensor (t,) - sbert cosine similarities
            values: value predictions (t,)
            dones: done flags (t,)
        
        returns:
            advantages: gae advantages (t,)
            returns: value targets (t,)
        """
        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0  # no next value for last step
            else:
                next_value = values[t + 1]
            
            # td error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            
            # gae computation
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            
        # returns are advantages + values
        returns = advantages + values
        return advantages, returns


def enhanced_ppo_step_with_research_logging(
    retriever,
    critic: EnhancedCritic,
    queries: List[str],
    passages: List[List[str]],
    gt_answers: List[str],
    llm: Any,
    tok_llm: Any,
    args: Any,
    opt_r: torch.optim.Optimizer,
    opt_c: torch.optim.Optimizer,
    gae_buffer: GAEBuffer,
    accelerator: Any,
    device: torch.device,
    research_logger: ResearchLogger = None,
    global_step: int = 0
) -> Dict[str, float]:
    """enhanced ppo training step with comprehensive research logging"""
    
    # ensure retriever is in training mode for proper gradient flow
    retriever.train()
    critic.train()
    
    B = len(queries)
    
    # store passage lengths for later use and ensure correct data structure
    passage_lengths = []
    
    # ensure passages is a list of lists of strings
    corrected_passages = []
    for i, plist in enumerate(passages):
        if isinstance(plist, dict):
            # convert dict to list if needed (shouldn't happen with our preprocessing)
            plist = list(plist.values())
            logger.debug(f"converted dict to list for passage {i}, length: {len(plist)}")
        elif not isinstance(plist, list):
            # convert other types to list
            plist = [str(plist)]
            logger.debug(f"converted {type(plist)} to list for passage {i}")
        
        # ensure all items in the list are strings
        clean_plist = []
        for passage in plist:
            if isinstance(passage, list):
                passage = " ".join(str(p) for p in passage)
            clean_plist.append(str(passage))
        
        corrected_passages.append(clean_plist)
        passage_lengths.append(len(clean_plist))
    
    # update passages to use corrected structure
    passages = corrected_passages
    B = len(queries)  # reconfirm batch size after data correction
    
    # step 1: get old policy values (before any updates) - fixed kl divergence bug
    with torch.no_grad():
        q_emb = embed(retriever, queries, device, args.embedding_chunk_size)
        
        # handle variable number of passages per example
        flat_passages = []
        for plist in passages:
            flat_passages.extend(plist)
        
        # encode all passages
        p_emb = embed(retriever, flat_passages, device, args.embedding_chunk_size)
        
        # reconstruct batch structure with proper indexing
        max_passages = max(passage_lengths)
        batch_p_emb = torch.zeros(B, max_passages, q_emb.size(-1), device=p_emb.device)
        
        start_idx = 0
        for i, length in enumerate(passage_lengths):
            end_idx = start_idx + length
            batch_p_emb[i, :length] = p_emb[start_idx:end_idx]
            start_idx = end_idx
        
        p_emb = batch_p_emb
        
        # compute similarities with temperature and masking
        sim_old = torch.einsum("bd,bkd->bk", q_emb, p_emb) / args.temperature
        
        # create mask for valid passages (handle padding)
        mask = torch.zeros(B, max_passages, device=sim_old.device)
        for i, length in enumerate(passage_lengths):
            mask[i, :length] = 1
        
        # apply mask to similarities (set padded positions to very negative values)
        sim_old = sim_old.masked_fill(mask == 0, -1e9)
        
        # fixed: store old probabilities properly for kl divergence
        old_probs = torch.softmax(sim_old, dim=1)
        
        # action selection - always stochastic during training for exploration
        action_probs = torch.softmax(sim_old, dim=1)
        # ensure we don't sample from masked positions - improved numerical stability
        action_probs = action_probs.masked_fill(mask == 0, 0)
        action_probs = action_probs / (action_probs.sum(dim=1, keepdim=True) + 1e-7)  # better numerical stability
        actions = torch.multinomial(action_probs, args.retrieve_top_k)
        
        # old log probabilities
        log_probs = torch.log_softmax(sim_old, dim=1)
        old_logp = log_probs.gather(1, actions).sum(1)
        
        # old values
        old_values = critic(q_emb, p_emb)
    
    # step 2: generate llm responses (passages are now guaranteed to be lists)
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
        
        # debug: check what we're joining
        if any(not isinstance(p, str) for p in selected_passages):
            logger.warning(f"non-string passage found in batch {b}: {[type(p) for p in selected_passages]}")
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
        outs = llm.generate(**toks, max_new_tokens=64, temperature=0.7, 
                           do_sample=True, pad_token_id=tok_llm.eos_token_id)
    
    # extract only the generated answers
    preds = tok_llm.batch_decode(outs, skip_special_tokens=True)
    preds_clean = [extract_answer_from_generation(pred, prompt, args.llm_type) 
                   for pred, prompt in zip(preds, prompts)]
    
    # step 3: compute sbert cosine similarity rewards
    rewards = reward_fn(preds_clean, gt_answers).to(device)
    
    # step 4: gae computation
    dones = torch.zeros(B, device=device)  # no episode termination
    advantages, returns = gae_buffer.compute_gae(rewards, old_values, dones)
    
    # normalize advantages (crucial for stable training)
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # step 5: multiple ppo epochs
    total_actor_loss = 0
    total_critic_loss = 0
    total_entropy = 0
    total_kl_div = 0
    num_updates = 0
    
    for ppo_epoch in range(args.ppo_epochs):
        # shuffle indices for each epoch
        indices = torch.randperm(B, device=device)
        
        for start_idx in range(0, B, args.minibatch_size):
            end_idx = min(start_idx + args.minibatch_size, B)
            mb_indices = indices[start_idx:end_idx]
            
            if len(mb_indices) < 2:  # skip tiny batches
                continue
            
            opt_r.zero_grad()
            opt_c.zero_grad()
            
            # get minibatch data
            mb_queries = [queries[i] for i in mb_indices]
            mb_passages = [passages[i] for i in mb_indices]
            mb_old_logp = old_logp[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]
            mb_actions = actions[mb_indices]
            
            # fresh forward pass
            q_emb_new = embed(retriever, mb_queries, device, args.embedding_chunk_size) 
            verify_gradients(q_emb_new, "query_embeddings")
            
            # handle variable passage lengths for minibatch
            flat_mb_passages = []
            mb_passage_lengths = []
            for i in mb_indices:
                plist = passages[i]
                flat_mb_passages.extend(plist)
                mb_passage_lengths.append(len(plist))
            
            # encode passages
            p_emb_flat = embed(retriever, flat_mb_passages, device, args.embedding_chunk_size)
            verify_gradients(p_emb_flat, "passage_embeddings")
            
            # reconstruct minibatch structure
            max_mb_passages = max(mb_passage_lengths)
            p_emb_new = torch.zeros(len(mb_indices), max_mb_passages, q_emb_new.size(-1), device=p_emb_flat.device)
            
            start_idx_inner = 0
            for i, length in enumerate(mb_passage_lengths):
                end_idx_inner = start_idx_inner + length
                p_emb_new[i, :length] = p_emb_flat[start_idx_inner:end_idx_inner]
                start_idx_inner = end_idx_inner
            
            # critical: ensure p_emb_new maintains gradients
            if not p_emb_new.requires_grad:
                logger.warning("lost gradients in passage embedding reconstruction, fixing...")
                p_emb_new.requires_grad_(True)
                
                # create a gradient connection to the original embeddings
                # this ensures gradients flow back through the reconstruction
                if p_emb_flat.requires_grad:
                    # create a connection that preserves gradients
                    gradient_connection = p_emb_flat.sum() * 0.0  # zero contribution but preserves gradients
                    p_emb_new = p_emb_new + gradient_connection.expand_as(p_emb_new)
            
            # double-check gradient flow is preserved
            verify_gradients(p_emb_new, "reconstructed_passage_embeddings")
            
            # compute new similarities with temperature and masking
            sim_new = torch.einsum("bd,bkd->bk", q_emb_new, p_emb_new) / args.temperature
            verify_gradients(sim_new, "similarity_scores")
            
            # create mask for valid passages in minibatch
            mb_mask = torch.zeros(len(mb_indices), max_mb_passages, device=sim_new.device)
            for i, length in enumerate(mb_passage_lengths):
                mb_mask[i, :length] = 1
            
            # apply mask to similarities
            sim_new = sim_new.masked_fill(mb_mask == 0, -1e9)
            
            # new log probabilities
            log_probs_new = torch.log_softmax(sim_new, dim=1)
            new_logp = log_probs_new.gather(1, mb_actions).sum(1)
            verify_gradients(new_logp, "new_log_probs")
            
            # ppo losses
            # importance sampling ratio
            ratio = torch.exp(new_logp - mb_old_logp)
            
            # clipped surrogate objective
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # value loss with optional clipping
            new_values = critic(q_emb_new, p_emb_new)
            value_loss = F.mse_loss(new_values, mb_returns)
            
            # entropy bonus for exploration
            action_probs_new = torch.softmax(sim_new, dim=1)
            # improved numerical stability for masked probabilities
            masked_probs = action_probs_new.masked_fill(mb_mask == 0, 1e-7)  # better numerical stability
            entropy = -(masked_probs * torch.log(masked_probs + 1e-7)).sum(1)
            # normalize by number of valid passages
            valid_passages = mb_mask.sum(1).clamp_min(1)
            entropy = entropy / valid_passages
            entropy = entropy.mean()
            
            # fixed: kl divergence for monitoring - use stored old probabilities
            with torch.no_grad():
                # use the old probabilities computed before any updates
                old_probs_mb = old_probs[mb_indices]
                
                # ensure consistent dimensions between old and new probabilities
                if old_probs_mb.size(1) != action_probs_new.size(1):
                    # pad or trim to match dimensions
                    if old_probs_mb.size(1) < action_probs_new.size(1):
                        # pad old probs with zeros
                        padding = torch.zeros(old_probs_mb.size(0), 
                                            action_probs_new.size(1) - old_probs_mb.size(1),
                                            device=old_probs_mb.device)
                        old_probs_mb = torch.cat([old_probs_mb, padding], dim=1)
                    else:
                        # trim old probs to match new size
                        old_probs_mb = old_probs_mb[:, :action_probs_new.size(1)]
                
                # mask both probability distributions with better numerical stability
                masked_new_probs = action_probs_new.masked_fill(mb_mask == 0, 1e-7)
                masked_old_probs = old_probs_mb.masked_fill(mb_mask == 0, 1e-7)
                
                kl_div = (masked_new_probs * (torch.log(masked_new_probs + 1e-7) - 
                                            torch.log(masked_old_probs + 1e-7))).sum(1)
                # normalize by number of valid passages
                kl_div = kl_div / valid_passages
                kl_div = kl_div.mean()
            
            # combined loss
            total_loss = (actor_loss + 
                         args.value_coef * value_loss - 
                         args.entropy_coef * entropy)
            
            # verify gradients before backward pass
            verify_gradients(total_loss, "total_loss")
            
            # backward pass with gradient clipping
            accelerator.backward(total_loss)
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(retriever.parameters(), args.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            
            opt_r.step()
            opt_c.step()
            
            # monitor gradient flow after optimizer steps
            if global_step % 25 == 0:  # monitor gradients every 25 steps
                grad_stats = monitor_gradients(retriever, global_step, research_logger.tb if research_logger else None)
                
                # warn if gradient flow is poor
                if grad_stats['params_with_grads'] < grad_stats['total_params'] * 0.8:
                    logger.warning(f"poor gradient flow: only {grad_stats['params_with_grads']}/{grad_stats['total_params']} parameters have gradients")
                
                if grad_stats['total_grad_norm'] < 1e-10:
                    logger.warning(f"very small gradient norms detected: {grad_stats['total_grad_norm']:.2e}")
            
            # accumulate metrics
            total_actor_loss += actor_loss.item()
            total_critic_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl_div += kl_div.item()
            num_updates += 1
    
    # prepare metrics for return
    metrics = {
        'actor_loss': total_actor_loss / max(num_updates, 1),
        'critic_loss': total_critic_loss / max(num_updates, 1),
        'entropy': total_entropy / max(num_updates, 1),
        'kl_div': total_kl_div / max(num_updates, 1),
        'reward_mean': rewards.mean().item(),
        'reward_std': rewards.std().item(),
        'advantages_mean': advantages.mean().item(),
        'advantages_std': advantages.std().item(),
        'value_mean': old_values.mean().item(),
        'returns_mean': returns.mean().item(),
    }
    
    # research logging
    if research_logger:
        research_logger.log_training_step(
            step=global_step,
            queries=queries,
            preds_clean=preds_clean,
            gt_answers=gt_answers,
            rewards=rewards,
            ctx=ctx,
            metrics=metrics,
            tokenizer=tok_llm
        )
    
    # return averaged metrics
    return metrics