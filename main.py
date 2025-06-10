#!/usr/bin/env python3
"""
complete main training script for ppo retriever training.
"""

import logging
import json
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

from config import parse_args
from utils import set_seed, setup_retriever_for_training
from models import EnhancedCritic
from data import load_ms_marco_with_splits, load_ms_marco_subset, create_dataloaders
from ppo import GAEBuffer, enhanced_ppo_step_with_research_logging
from evaluation import evaluate_model_with_research_logging
from research_logging import ResearchLogger

# setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    set_seed()
    accelerator = Accelerator()
    
    # setup device
    device = accelerator.device
    
    # enhanced logging setup with research logger
    log_dir = Path(args.log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    if accelerator.is_main_process:
        log_dir.mkdir(parents=True, exist_ok=True)
        tb = SummaryWriter(log_dir)
        
        # initialize research logger
        research_logger = ResearchLogger(tb, device)
        
        # log hyperparameters to tensorboard
        hparams_dict = vars(args)
        metrics_dict = {
            'hparam/reward_final': 0,
            'hparam/actor_loss_final': 0,
            'hparam/critic_loss_final': 0
        }
        tb.add_hparams(hparams_dict, metrics_dict)
        
        # log text summary of configuration
        config_text = "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
        tb.add_text("config", config_text, 0)
        
        logger.info(f"starting enhanced ppo training with research logging")
        logger.info(f"config: {vars(args)}")
        logger.info(f"tensorboard logs will be saved to: {log_dir}")
    else:
        tb = None
        research_logger = None
    
    # enhanced data loading with proper splits and fallback options
    logger.info("loading ms marco dataset...")
    
    if args.use_subset:
        # use small subset for quick testing
        logger.info(f"using subset mode: {args.subset_size} samples")
        train_ds = load_ms_marco_subset("train", args.subset_size)
        val_ds = load_ms_marco_subset("validation", args.subset_size // 5)  # smaller validation set
        test_ds = load_ms_marco_subset("test", args.subset_size // 5) if train_ds else None
        
        datasets = {
            "train": train_ds,
            "validation": val_ds,
            "test": test_ds
        }
    else:
        # try full dataset loading with fallbacks
        datasets = load_ms_marco_with_splits()
        
        # fallback to subset if full loading fails
        if datasets["train"] is None:
            logger.warning("full dataset loading failed. falling back to subset mode...")
            train_ds = load_ms_marco_subset("train", 50000)  # 50k samples
            val_ds = load_ms_marco_subset("validation", 10000)
            test_ds = load_ms_marco_subset("test", 10000)
            
            datasets = {
                "train": train_ds,
                "validation": val_ds,
                "test": test_ds
            }
    
    # final fallback - create minimal dataset if everything fails
    if datasets["train"] is None:
        logger.error("all dataset loading methods failed. creating minimal synthetic dataset for testing...")
        # create a tiny synthetic dataset for testing the pipeline
        synthetic_data = []
        for i in range(100):
            synthetic_data.append({
                "query": f"what is the capital of country {i}?",
                "answers": [f"the capital is city {i}"],
                "passages": [
                    f"city {i} is the capital and largest city of country {i}.",
                    f"country {i} is located in continent {i//10}.",
                    f"the population of city {i} is about {i*10000} people."
                ]
            })
        
        from datasets import Dataset
        synthetic_dataset = Dataset.from_list(synthetic_data)
        datasets = {
            "train": synthetic_dataset,
            "validation": synthetic_dataset.select(range(20)),
            "test": synthetic_dataset.select(range(20, 40))
        }
        logger.info("created synthetic dataset for pipeline testing")
    
    # limit training samples if specified
    if args.max_train_samples and datasets["train"] is not None:
        original_size = len(datasets["train"])
        datasets["train"] = datasets["train"].select(range(min(args.max_train_samples, original_size)))
        logger.info(f"limited training set from {original_size} to {len(datasets['train'])} samples")
    
    # handle missing validation split
    if datasets["validation"] is None and datasets["train"] is not None:
        logger.info("no validation split found, creating one from training data...")
        train_size = int(len(datasets["train"]) * (1 - args.validation_split_size))
        val_size = len(datasets["train"]) - train_size
        
        train_split, val_split = torch.utils.data.random_split(
            datasets["train"], 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        datasets["train"] = train_split
        datasets["validation"] = val_split
        logger.info(f"created validation split: {len(val_split)} examples")
    
    # create dataloaders
    dataloaders = create_dataloaders(datasets, args)
    
    if "train" not in dataloaders:
        raise ValueError("no training data available!")
    
    logger.info(f"data splits loaded:")
    for split_name, dl in dataloaders.items():
        logger.info(f"  {split_name}: {len(dl.dataset)} examples")
    
    # models setup
    logger.info("loading models...")
    retriever = SentenceTransformer("sentence-transformers/gtr-t5-base")
    
    # critical: enhanced setup for gradient-based training
    gradient_flow_ok = setup_retriever_for_training(retriever)
    
    if not gradient_flow_ok:
        logger.error("critical: gradient flow test failed!")
        logger.error("ppo training will not work properly without gradient flow.")
        logger.error("please check the embedding function and model setup.")
        # don't exit - continue with warning as some issues might resolve during training
    
    logger.info(f"retriever embedding dimension: {retriever.get_sentence_embedding_dimension()}")
    
    # enable gradient checkpointing if requested (memory optimization)
    if args.gradient_checkpointing:
        try:
            if hasattr(retriever._first_module(), 'auto_model'):
                retriever._first_module().auto_model.gradient_checkpointing_enable()
                logger.info("enabled gradient checkpointing for retriever")
        except Exception as e:
            logger.warning(f"could not enable gradient checkpointing: {e}")
    
    critic = EnhancedCritic(
        d=retriever.get_sentence_embedding_dimension(),
        num_heads=args.critic_heads,
        dropout=args.critic_dropout
    )
    
    tok_llm = AutoTokenizer.from_pretrained(args.llm)
    if tok_llm.pad_token is None:
        tok_llm.pad_token = tok_llm.eos_token
    
    # configure padding side based on model type
    if args.llm_type == "causal":
        tok_llm.padding_side = 'left'  # for decoder-only models (dialogpt, gpt, etc.)
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
    else:  # seq2seq
        tok_llm.padding_side = 'right'  # for encoder-decoder models (t5, bart, etc.)
        llm = AutoModelForSeq2SeqLM.from_pretrained(
            args.llm,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()
    
    # optimizers with different learning rates
    # verify retriever parameters have gradients enabled
    retriever_params_with_grad = [p for p in retriever.parameters() if p.requires_grad]
    critic_params_with_grad = [p for p in critic.parameters() if p.requires_grad]
    
    # count actual parameters (elements), not just tensors
    retriever_param_count = sum(p.numel() for p in retriever_params_with_grad)
    critic_param_count = sum(p.numel() for p in critic_params_with_grad)
    
    logger.info(f"retriever trainable parameter tensors: {len(retriever_params_with_grad)}")
    logger.info(f"retriever trainable parameter count: {retriever_param_count:,}")
    logger.info(f"critic trainable parameter tensors: {len(critic_params_with_grad)}")
    logger.info(f"critic trainable parameter count: {critic_param_count:,}")
    
    if len(retriever_params_with_grad) == 0:
        raise ValueError("no retriever parameters have requires_grad=true! cannot train.")
    if len(critic_params_with_grad) == 0:
        raise ValueError("no critic parameters have requires_grad=true! cannot train.")
    
    opt_r = torch.optim.Adam(retriever_params_with_grad, lr=args.lr)
    opt_c = torch.optim.Adam(critic_params_with_grad, lr=args.lr * 2)  # critic learns faster
    
    # learning rate schedulers
    scheduler_r = torch.optim.lr_scheduler.ExponentialLR(opt_r, gamma=args.lr_decay)
    scheduler_c = torch.optim.lr_scheduler.ExponentialLR(opt_c, gamma=args.lr_decay)
    
    # gae buffer
    gae_buffer = GAEBuffer(gamma=args.gamma, lam=args.gae_lambda)
    
    # accelerator preparation
    dl_train, retriever, critic, llm, opt_r, opt_c = accelerator.prepare(
        dataloaders["train"], retriever, critic, llm, opt_r, opt_c
    )
    
    # training loop with validation and research logging
    global_step = 0
    best_val_reward = -float('inf')
    best_train_reward = -float('inf')
    early_stopping_counter = 0
    
    for epoch in range(args.epochs):
        logger.info(f"starting epoch {epoch + 1}/{args.epochs}")
        
        # training phase
        retriever.train()
        critic.train()
        
        for batch_idx, (queries, passages, gt_answers) in enumerate(dl_train):
            # enhanced ppo training step with research logging
            metrics = enhanced_ppo_step_with_research_logging(
                retriever=retriever,
                critic=critic,
                queries=queries,
                passages=passages,
                gt_answers=gt_answers,
                llm=llm,
                tok_llm=tok_llm,
                args=args,
                opt_r=opt_r,
                opt_c=opt_c,
                gae_buffer=gae_buffer,
                accelerator=accelerator,
                device=device,
                research_logger=research_logger,
                global_step=global_step
            )
            
            # standard tensorboard logging
            if accelerator.is_main_process and global_step % args.log_interval == 0:
                # training metrics
                for key, value in metrics.items():
                    tb.add_scalar(f"train/{key}", value, global_step)
                
                # learning rates
                tb.add_scalar("learning_rate/retriever", opt_r.param_groups[0]['lr'], global_step)
                tb.add_scalar("learning_rate/critic", opt_c.param_groups[0]['lr'], global_step)
                
                # training progress
                tb.add_scalar("progress/epoch", epoch, global_step)
                tb.add_scalar("progress/batch", batch_idx, global_step)
                
                # console logging
                logger.info(
                    f"step {global_step:6d} | "
                    f"epoch {epoch+1:2d} | "
                    f"batch {batch_idx:4d} | "
                    f"sbert similarity: {metrics['reward_mean']:.4f}±{metrics['reward_std']:.4f} | "
                    f"actor: {metrics['actor_loss']:.4f} | "
                    f"critic: {metrics['critic_loss']:.4f} | "
                    f"entropy: {metrics['entropy']:.4f} | "
                    f"kl: {metrics['kl_div']:.4f} | "
                    f"lr: {opt_r.param_groups[0]['lr']:.2e}"
                )
            
            # validation evaluation with research logging
            if (global_step % args.eval_steps == 0 and global_step > 0) or \
               (epoch % args.eval_epochs == 0 and batch_idx == 0):
                
                if "validation" in dataloaders:
                    val_metrics, _ = evaluate_model_with_research_logging(
                        retriever, critic, llm, tok_llm, 
                        dataloaders["validation"], args, device, "validation",
                        research_logger, global_step
                    )
                    
                    # tensorboard logging for validation
                    if accelerator.is_main_process:
                        for key, value in val_metrics.items():
                            tb.add_scalar(key, value, global_step)
                    
                    # check for improvement
                    current_val_reward = val_metrics['validation/reward_mean']
                    if current_val_reward > best_val_reward:
                        best_val_reward = current_val_reward
                        early_stopping_counter = 0
                        
                        # save best model
                        if args.save_best_model and accelerator.is_main_process:
                            checkpoint = {
                                'retriever': retriever.state_dict(),
                                'critic': critic.state_dict(),
                                'opt_r': opt_r.state_dict(),
                                'opt_c': opt_c.state_dict(),
                                'args': vars(args),
                                'step': global_step,
                                'val_reward': best_val_reward
                            }
                            try:
                                # clear cache before saving to free memory
                                torch.cuda.empty_cache()
                                
                                # check disk space (rough estimate: need ~500mb)
                                import shutil
                                try:
                                    free_bytes = shutil.disk_usage(str(log_dir)).free
                                    if free_bytes < 500_000_000:  # 500mb
                                        logger.warning(f"low disk space: {free_bytes/1e9:.1f}gb free")
                                except:
                                    pass  # continue if can't check
                                
                                torch.save(checkpoint, log_dir / "best_val_model.pt")
                                logger.info(f"new best validation sbert similarity: {best_val_reward:.4f}")
                            except (RuntimeError, OSError) as e:
                                logger.error(f"failed to save validation checkpoint: {e}")
                                logger.warning("continuing training without saving validation checkpoint")
                    else:
                        early_stopping_counter += 1
                        
                    # early stopping
                    if early_stopping_counter >= args.early_stopping_patience:
                        logger.info(f"early stopping triggered at step {global_step}")
                        break
                
                # return to training mode
                retriever.train()
                critic.train()
            
            # save best training model
            if accelerator.is_main_process and metrics['reward_mean'] > best_train_reward:
                best_train_reward = metrics['reward_mean']
                checkpoint = {
                    'retriever': retriever.state_dict(),
                    'critic': critic.state_dict(),
                    'opt_r': opt_r.state_dict(),
                    'opt_c': opt_c.state_dict(),
                    'args': vars(args),
                    'step': global_step,
                    'train_reward': best_train_reward
                }
                try:
                    torch.cuda.empty_cache()
                    torch.save(checkpoint, log_dir / "best_train_model.pt")
                    logger.info(f"saved best training model (sbert: {best_train_reward:.4f})")
                except (RuntimeError, OSError) as e:
                    logger.error(f"failed to save training checkpoint: {e}")
                    logger.warning("continuing training without saving training checkpoint")
            
            global_step += 1
            
            # memory cleanup every few steps
            if global_step % 10 == 0:
                torch.cuda.empty_cache()
            
            # more frequent scheduler updates for better learning rate control
            if global_step % args.scheduler_step_interval == 0:
                scheduler_r.step()
                scheduler_c.step()
        
        # check for early stopping at epoch level
        if early_stopping_counter >= args.early_stopping_patience:
            break
        
        # end of epoch validation with research logging
        if "validation" in dataloaders:
            val_metrics, _ = evaluate_model_with_research_logging(
                retriever, critic, llm, tok_llm,
                dataloaders["validation"], args, device, "validation",
                research_logger, global_step
            )
            
            if accelerator.is_main_process:
                tb.add_scalar("epoch/validation_reward", val_metrics['validation/reward_mean'], epoch)
        
        if accelerator.is_main_process:
            # log epoch summary to tensorboard
            tb.add_scalar("epoch/completed", epoch + 1, global_step)
            tb.add_scalar("epoch/best_val_reward", best_val_reward, global_step)
            tb.add_scalar("epoch/best_train_reward", best_train_reward, global_step)
            
            logger.info(f"epoch {epoch + 1} completed. best val sbert similarity: {best_val_reward:.4f}")
    
    # final test evaluation with research logging
    if "test" in dataloaders and accelerator.is_main_process:
        logger.info("running final test evaluation...")
        
        # load best model if available
        if args.save_best_model and (log_dir / "best_val_model.pt").exists():
            checkpoint = torch.load(log_dir / "best_val_model.pt", map_location=device)
            retriever.load_state_dict(checkpoint['retriever'])
            critic.load_state_dict(checkpoint['critic'])
            logger.info("loaded best validation model for test evaluation")
        
        test_metrics, test_details = evaluate_model_with_research_logging(
            retriever, critic, llm, tok_llm,
            dataloaders["test"], args, device, "test",
            research_logger, global_step
        )
        
        # log test results
        for key, value in test_metrics.items():
            tb.add_scalar(key, value, global_step)
        
        # save detailed test results
        test_results = {
            'metrics': test_metrics,
            'detailed_results': test_details,
            'final_step': global_step,
            'best_val_reward': best_val_reward,
            'best_train_reward': best_train_reward
        }
        
        with open(log_dir / "test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        logger.info("test evaluation completed and saved!")
        logger.info(f"final test sbert similarity: {test_metrics['test/reward_mean']:.4f}")
    
    # final logging and cleanup with research summary
    if accelerator.is_main_process:
        # create final research summary
        research_logger.create_final_summary(best_train_reward, best_val_reward, global_step)
        
        # update final hparams metrics
        final_val_reward = best_val_reward if best_val_reward > -float('inf') else 0
        final_test_reward = test_metrics.get('test/reward_mean', 0) if 'test_metrics' in locals() else 0
        
        tb.add_hparams(vars(args), {
            'hparam/reward_final': best_train_reward,
            'hparam/val_reward_final': final_val_reward,
            'hparam/test_reward_final': final_test_reward,
            'hparam/total_steps': global_step,
            'hparam/epochs_completed': min(epoch + 1, args.epochs)
        })
        
        tb.close()
        logger.info("training completed with comprehensive research logging!")
        logger.info(f"tensorboard logs saved to: {log_dir}")
        logger.info(f"to view logs, run: tensorboard --logdir {log_dir}")
        logger.info(f"best models saved to: {log_dir}")
        
        # print final summary
        print("\n" + "="*80)
        print("FINAL TRAINING SUMMARY")
        print("="*80)
        print(f"best training sbert similarity:   {best_train_reward:.4f}")
        print(f"best validation sbert similarity: {final_val_reward:.4f}")
        if 'test_metrics' in locals():
            print(f"final test sbert similarity:      {final_test_reward:.4f}")
        print(f"total training steps:             {global_step}")
        print(f"tensorboard command:              tensorboard --logdir {log_dir}")
        print("="*80)
        print("research insights available in tensorboard:")
        print("  - sbert similarity distributions and trends")
        print("  - llm generation quality analysis") 
        print("  - best/worst examples with detailed analysis")
        print("  - publication-ready plots and visualizations")
        print("="*80)


if __name__ == "__main__":
    main() 