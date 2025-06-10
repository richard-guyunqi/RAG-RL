#!/usr/bin/env python3
"""
configuration module for ppo retriever training.
"""

import argparse


def parse_args():
    """enhanced argument parsing with all ppo hyperparameters and evaluation options"""
    ap = argparse.ArgumentParser(description="Enhanced PPO Retriever Training with Research Logging")
    
    # basic training args
    ap.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size (reduced default for memory)")
    ap.add_argument("--minibatch_size", type=int, default=16, help="PPO minibatch size (reduced default for memory)")
    ap.add_argument("--retrieve_top_k", type=int, default=3, help="Number of passages to retrieve")
    ap.add_argument("--lr", type=float, default=5e-6, help="Learning rate (reduced for stability)")
    ap.add_argument("--log_dir", type=str, default="runs/enhanced_ppo_research", help="Logging directory")
    ap.add_argument("--llm", type=str, default="google/flan-t5-base", help="LLM model")
    ap.add_argument("--llm_type", type=str, default="seq2seq", choices=["causal", "seq2seq"], help="LLM model type")
    
    # memory optimization args
    ap.add_argument("--embedding_chunk_size", type=int, default=8, help="Chunk size for embedding computation")
    ap.add_argument("--gradient_checkpointing", action='store_true', help="Use gradient checkpointing to save memory")
    
    # dataset args
    ap.add_argument("--use_subset", action='store_true', help="Use small subset for testing")
    ap.add_argument("--subset_size", type=int, default=10000, help="Size of subset if using --use_subset")
    ap.add_argument("--max_train_samples", type=int, default=None, help="Max training samples to use")
    
    # enhanced ppo args
    ap.add_argument("--ppo_epochs", type=int, default=4, help="PPO epochs per batch")
    ap.add_argument("--clip_eps", type=float, default=0.15, help="PPO clipping parameter (reduced for stability)")
    ap.add_argument("--value_coef", type=float, default=0.5, help="Value loss coefficient")
    ap.add_argument("--entropy_coef", type=float, default=0.01, help="Entropy regularization coefficient")
    ap.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping threshold")
    ap.add_argument("--temperature", type=float, default=1.0, help="Temperature for action selection")
    ap.add_argument("--stochastic_actions", action='store_true', help="Use stochastic action selection during eval (training always stochastic)")
    
    # gae args
    ap.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    ap.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    
    # learning rate scheduling
    ap.add_argument("--lr_decay", type=float, default=0.99, help="LR decay per epoch")
    ap.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    ap.add_argument("--scheduler_step_interval", type=int, default=100, help="Step scheduler every N batches")
    
    # critic architecture
    ap.add_argument("--critic_heads", type=int, default=8, help="Number of attention heads in critic")
    ap.add_argument("--critic_dropout", type=float, default=0.1, help="Dropout in critic")
    
    # logging
    ap.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    ap.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    
    # evaluation-specific args
    ap.add_argument("--eval_epochs", type=int, default=1, help="Evaluate every N epochs")
    ap.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    ap.add_argument("--save_best_model", action='store_true', help="Save model with best validation score")
    ap.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    ap.add_argument("--validation_split_size", type=float, default=0.1, help="Fraction of train to use as validation if no val split")
    
    return ap.parse_args() 