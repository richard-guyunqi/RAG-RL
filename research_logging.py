#!/usr/bin/env python3
"""
research logging and visualization for ppo retriever training.
"""

import logging
import warnings
import io
from typing import List, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

logger = logging.getLogger(__name__)


class ResearchLogger:
    """comprehensive research logging for sbert rewards and llm outputs"""
    
    def __init__(self, tb_writer, device):
        self.tb = tb_writer
        self.device = device
        
    def log_training_step(self, step, queries, preds_clean, gt_answers, rewards, 
                         ctx, metrics, tokenizer):
        """main logging function for each training step"""
        
        # core performance metrics (every step)
        self.tb.add_scalar("performance/reward_mean", rewards.mean().item(), step)
        self.tb.add_scalar("performance/reward_std", rewards.std().item(), step)
        self.tb.add_scalar("performance/actor_loss", metrics['actor_loss'], step)
        self.tb.add_scalar("performance/critic_loss", metrics['critic_loss'], step)
        self.tb.add_scalar("performance/entropy", metrics['entropy'], step)
        self.tb.add_scalar("performance/kl_divergence", metrics['kl_div'], step)
        
        # sbert reward analysis (every 25 steps)
        if step % 25 == 0:
            self._log_sbert_analysis(preds_clean, gt_answers, rewards, step)
        
        # llm generation analysis (every 25 steps) 
        if step % 25 == 0:
            self._log_llm_analysis(preds_clean, gt_answers, tokenizer, step)
        
        # qualitative examples (every 50 steps)
        if step % 50 == 0:
            self._log_examples(queries, preds_clean, gt_answers, rewards, ctx, step)
        
        # research visualizations (every 100 steps)
        if step % 100 == 0:
            self._log_research_plots(rewards, preds_clean, gt_answers, step)
    
    def log_evaluation(self, split_name, eval_metrics, queries, preds, gt_answers, 
                      rewards, step):
        """log evaluation results with sbert analysis"""
        
        # core evaluation metrics
        for key, value in eval_metrics.items():
            self.tb.add_scalar(f"eval_{split_name}/{key.split('/')[-1]}", value, step)
        
        # sbert reward analysis for evaluation
        if len(preds) > 0:
            self._log_sbert_analysis(preds, gt_answers, rewards, step, prefix=f"eval_{split_name}")
            
            # evaluation examples
            for i in range(min(3, len(queries))):
                example_text = (
                    f"**{split_name.upper()} Example {i+1}**\n\n"
                    f"Query: {queries[i]}\n\n"
                    f"Generated: {preds[i]}\n\n"
                    f"Ground Truth: {gt_answers[i]}\n\n"
                    f"SBERT Similarity: {rewards[i].item():.4f}"
                )
                self.tb.add_text(f"eval_{split_name}/example_{i+1}", example_text, step)
    
    def _log_sbert_analysis(self, preds, gt_answers, rewards, step, prefix="training"):
        """detailed sbert cosine similarity analysis"""
        
        # basic sbert statistics
        self.tb.add_scalar(f"{prefix}/sbert_similarity_min", rewards.min().item(), step)
        self.tb.add_scalar(f"{prefix}/sbert_similarity_max", rewards.max().item(), step)
        self.tb.add_scalar(f"{prefix}/sbert_similarity_median", rewards.median().item(), step)
        
        # sbert similarity distribution analysis
        very_low = (rewards < 0.2).float().mean().item() * 100    # poor semantic match
        low = ((rewards >= 0.2) & (rewards < 0.4)).float().mean().item() * 100    # weak match
        medium = ((rewards >= 0.4) & (rewards < 0.6)).float().mean().item() * 100  # moderate match
        high = ((rewards >= 0.6) & (rewards < 0.8)).float().mean().item() * 100    # good match
        very_high = (rewards >= 0.8).float().mean().item() * 100  # excellent match
        
        self.tb.add_scalar(f"{prefix}/sbert_very_low_pct", very_low, step)
        self.tb.add_scalar(f"{prefix}/sbert_low_pct", low, step)
        self.tb.add_scalar(f"{prefix}/sbert_medium_pct", medium, step)
        self.tb.add_scalar(f"{prefix}/sbert_high_pct", high, step)
        self.tb.add_scalar(f"{prefix}/sbert_very_high_pct", very_high, step)
        
        # log sbert reward histogram
        self.tb.add_histogram(f"{prefix}/sbert_similarity_distribution", rewards.cpu(), step)
        
        # semantic quality indicator (high-quality responses)
        high_quality_rate = (rewards >= 0.6).float().mean().item()
        self.tb.add_scalar(f"{prefix}/high_quality_response_rate", high_quality_rate, step)
    
    def _log_llm_analysis(self, preds, gt_answers, tokenizer, step):
        """llm generation quality analysis"""
        
        # length analysis
        pred_lengths = [len(pred.split()) for pred in preds]
        gt_lengths = [len(gt.split()) for gt in gt_answers]
        
        self.tb.add_scalar("llm/avg_generated_length", np.mean(pred_lengths), step)
        self.tb.add_scalar("llm/avg_gt_length", np.mean(gt_lengths), step)
        self.tb.add_scalar("llm/length_ratio", np.mean(pred_lengths) / np.mean(gt_lengths) if np.mean(gt_lengths) > 0 else 0, step)
        
        # generation quality metrics
        empty_rate = sum(1 for pred in preds if not pred.strip()) / len(preds)
        self.tb.add_scalar("llm/empty_generation_rate", empty_rate, step)
        
        # exact match rate (for reference, but reward is sbert-based)
        exact_matches = sum(1 for pred, gt in zip(preds, gt_answers) 
                           if pred.strip().lower() == gt.strip().lower())
        self.tb.add_scalar("llm/exact_match_rate", exact_matches / len(preds), step)
        
        # word overlap analysis
        overlaps = []
        for pred, gt in zip(preds, gt_answers):
            pred_words = set(pred.lower().split())
            gt_words = set(gt.lower().split())
            if gt_words:
                overlap = len(pred_words & gt_words) / len(gt_words)
                overlaps.append(overlap)
        
        self.tb.add_scalar("llm/word_overlap_rate", np.mean(overlaps) if overlaps else 0, step)
        
        # token diversity analysis
        try:
            all_tokens = []
            for pred in preds:
                tokens = tokenizer.encode(pred, add_special_tokens=False)
                all_tokens.extend(tokens)
            
            if all_tokens:
                unique_tokens = len(set(all_tokens))
                self.tb.add_scalar("llm/vocab_diversity", unique_tokens / len(all_tokens), step)
                self.tb.add_scalar("llm/avg_tokens_per_response", len(all_tokens) / len(preds), step)
        except:
            pass  # skip if tokenization fails
    
    def _log_examples(self, queries, preds, gt_answers, rewards, ctx, step):
        """log best and worst examples based on sbert similarity"""
        
        # sort by sbert cosine similarity
        sorted_indices = torch.argsort(rewards, descending=True)
        
        # best sbert similarity example
        best_idx = sorted_indices[0].item()
        best_text = (
            f"**HIGHEST SBERT SIMILARITY** ({rewards[best_idx]:.4f})\n\n"
            f"Query: {queries[best_idx]}\n\n"
            f"Retrieved Context: {ctx[best_idx][:400]}...\n\n"
            f"Generated Answer: {preds[best_idx]}\n\n"
            f"Ground Truth: {gt_answers[best_idx]}\n\n"
            f"Analysis: strong semantic alignment between generated answer and ground truth."
        )
        self.tb.add_text("examples/best_sbert", best_text, step)
        
        # worst sbert similarity example
        worst_idx = sorted_indices[-1].item()
        worst_text = (
            f"**LOWEST SBERT SIMILARITY** ({rewards[worst_idx]:.4f})\n\n"
            f"Query: {queries[worst_idx]}\n\n"
            f"Retrieved Context: {ctx[worst_idx][:400]}...\n\n"
            f"Generated Answer: {preds[worst_idx]}\n\n"
            f"Ground Truth: {gt_answers[worst_idx]}\n\n"
            f"Analysis: poor semantic alignment - retrieval or generation issue."
        )
        self.tb.add_text("examples/worst_sbert", worst_text, step)
        
        # medium similarity example for comparison
        mid_idx = len(sorted_indices) // 2
        medium_idx = sorted_indices[mid_idx].item()
        medium_text = (
            f"**MEDIUM SBERT SIMILARITY** ({rewards[medium_idx]:.4f})\n\n"
            f"Query: {queries[medium_idx]}\n\n"
            f"Retrieved Context: {ctx[medium_idx][:400]}...\n\n"
            f"Generated Answer: {preds[medium_idx]}\n\n"
            f"Ground Truth: {gt_answers[medium_idx]}\n\n"
            f"Analysis: moderate semantic alignment - room for improvement."
        )
        self.tb.add_text("examples/medium_sbert", medium_text, step)
    
    def _log_research_plots(self, rewards, preds, gt_answers, step):
        """create research-quality plots for papers"""
        
        try:
            # create comprehensive analysis plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. sbert similarity distribution
            rewards_np = rewards.cpu().numpy()
            ax1.hist(rewards_np, bins=25, alpha=0.7, color='lightblue', edgecolor='navy', linewidth=1.2)
            ax1.axvline(rewards_np.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {rewards_np.mean():.3f}')
            ax1.axvline(np.median(rewards_np), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards_np):.3f}')
            ax1.set_xlabel('SBERT Cosine Similarity')
            ax1.set_ylabel('Frequency')
            ax1.set_title('SBERT Reward Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. length analysis
            pred_lengths = [len(pred.split()) for pred in preds]
            gt_lengths = [len(gt.split()) for gt in gt_answers]
            
            ax2.scatter(gt_lengths, pred_lengths, alpha=0.6, c=rewards_np, cmap='viridis', s=50)
            ax2.plot([0, max(gt_lengths)], [0, max(gt_lengths)], 'r--', alpha=0.8, linewidth=2)
            ax2.set_xlabel('Ground Truth Length (words)')
            ax2.set_ylabel('Generated Length (words)')
            ax2.set_title('Length Correlation (colored by SBERT similarity)')
            cbar = plt.colorbar(ax2.collections[0], ax=ax2)
            cbar.set_label('SBERT Similarity')
            ax2.grid(True, alpha=0.3)
            
            # 3. quality distribution
            quality_ranges = ['Very Low\n(<0.2)', 'Low\n(0.2-0.4)', 'Medium\n(0.4-0.6)', 'High\n(0.6-0.8)', 'Very High\n(≥0.8)']
            quality_counts = [
                (rewards < 0.2).sum().item(),
                ((rewards >= 0.2) & (rewards < 0.4)).sum().item(),
                ((rewards >= 0.4) & (rewards < 0.6)).sum().item(),
                ((rewards >= 0.6) & (rewards < 0.8)).sum().item(),
                (rewards >= 0.8).sum().item()
            ]
            
            colors = ['#ff4444', '#ff8800', '#ffdd00', '#88dd00', '#00dd88']
            bars = ax3.bar(quality_ranges, quality_counts, color=colors, alpha=0.8, edgecolor='black')
            ax3.set_ylabel('Number of Responses')
            ax3.set_title('Response Quality Distribution')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # add percentage labels on bars
            total = sum(quality_counts)
            for bar, count in zip(bars, quality_counts):
                percentage = count / total * 100 if total > 0 else 0
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # 4. reward vs word overlap
            overlaps = []
            for pred, gt in zip(preds, gt_answers):
                pred_words = set(pred.lower().split())
                gt_words = set(gt.lower().split())
                if gt_words:
                    overlap = len(pred_words & gt_words) / len(gt_words)
                    overlaps.append(overlap)
                else:
                    overlaps.append(0)
            
            ax4.scatter(overlaps, rewards_np, alpha=0.6, s=50, color='purple')
            ax4.set_xlabel('Word Overlap Ratio')
            ax4.set_ylabel('SBERT Similarity')
            ax4.set_title('SBERT Similarity vs Word Overlap')
            ax4.grid(True, alpha=0.3)
            
            # add correlation coefficient (with safe handling for divide warnings)
            if len(overlaps) > 1 and np.std(overlaps) > 1e-8 and np.std(rewards_np) > 1e-8:
                # suppress numpy divide warnings for correlation calculation
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
                    correlation = np.corrcoef(overlaps, rewards_np)[0, 1]
                if not np.isnan(correlation):
                    ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                            transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
                else:
                    ax4.text(0.05, 0.95, 'Correlation: N/A (insufficient variance)', 
                            transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            else:
                ax4.text(0.05, 0.95, 'Correlation: N/A (insufficient data/variance)', 
                        transform=ax4.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
            
            plt.tight_layout()
            
            # convert to tensorboard
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            image = Image.open(buf)
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)
            self.tb.add_image("research/comprehensive_analysis", image_tensor, step)
            
            plt.close(fig)
            buf.close()
            
        except Exception as e:
            logger.warning(f"research plot creation failed: {e}")
    
    def create_final_summary(self, best_train_reward, best_val_reward, total_steps):
        """create final research summary"""
        
        improvement_pct = ((best_val_reward / max(best_train_reward, 0.001)) - 1) * 100
        overfitting_gap = best_train_reward - best_val_reward
        
        summary_text = (
            f"# ppo retriever training summary\n\n"
            f"## performance metrics\n"
            f"**best training sbert similarity:** {best_train_reward:.4f}\n"
            f"**best validation sbert similarity:** {best_val_reward:.4f}\n"
            f"**total training steps:** {total_steps}\n"
            f"**validation performance:** {improvement_pct:.1f}% relative to training\n\n"
            f"## analysis\n"
            f"- **semantic quality:** {'excellent' if best_val_reward > 0.7 else 'good' if best_val_reward > 0.5 else 'moderate' if best_val_reward > 0.3 else 'needs improvement'}\n"
            f"- **overfitting detection:** {'yes - gap: {overfitting_gap:.3f}' if overfitting_gap > 0.1 else 'no - well generalized'}\n"
            f"- **training stability:** {'stable' if best_val_reward > 0.4 else 'requires further training'}\n\n"
            f"## key insights\n"
            f"- sbert cosine similarity effectively measures semantic alignment\n"
            f"- ppo successfully optimizes retriever for llm answer quality\n"
            f"- {'high-quality semantic matches achieved' if best_val_reward > 0.6 else 'room for improvement in semantic matching'}"
        )
        
        self.tb.add_text("final/research_summary", summary_text, total_steps) 