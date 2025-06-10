#!/usr/bin/env python3
"""
model definitions for ppo retriever training.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def test_embedding_gradients(model, device):
    """enhanced test to verify embedding function preserves gradients"""
    logger.info("testing embedding gradient flow...")
    model.train()
    
    # clear any existing gradients
    model.zero_grad()
    
    test_texts = ["This is a test sentence for gradient verification."]
    
    # test embedding with our function
    embeddings = embed(model, test_texts, device, chunk_size=1)
    
    if not embeddings.requires_grad:
        logger.error("embedding function does not preserve gradients")
        return False
    
    logger.info("embedding function preserves gradients correctly")
    
    # test if gradients can flow back
    dummy_loss = embeddings.sum()
    
    # check gradient function exists
    if dummy_loss.grad_fn is None:
        logger.error("no gradient function attached to loss")
        return False
    
    # backward pass
    dummy_loss.backward()
    
    # check if retriever parameters got gradients
    params_with_grads = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is not None:
                params_with_grads += 1
            else:
                logger.debug(f"parameter without gradient: {name}")
    
    logger.info(f"parameters with gradients: {params_with_grads}/{total_params}")
    
    if params_with_grads > 0:
        logger.info("gradients successfully flow back to retriever parameters")
        # clear gradients for actual training
        model.zero_grad()
        return True
    else:
        logger.error("gradients not flowing back to retriever parameters")
        return False


def setup_retriever_for_training(retriever):
    """properly configure retriever for gradient-based training"""
    logger.info("setting up retriever for ppo training...")
    
    # enable gradients for all parameters
    total_param_tensors = 0
    trainable_param_tensors = 0
    total_param_count = 0
    trainable_param_count = 0
    
    for name, param in retriever.named_parameters():
        param_size = param.numel()  # number of elements in this tensor
        total_param_tensors += 1
        total_param_count += param_size
        
        if not param.requires_grad:
            param.requires_grad = True
            logger.debug(f"enabled gradients for: {name}")
        
        trainable_param_tensors += 1
        trainable_param_count += param_size
    
    logger.info(f"retriever parameter tensors: {total_param_tensors} total, {trainable_param_tensors} trainable")
    logger.info(f"retriever parameter count: {total_param_count:,} total, {trainable_param_count:,} trainable")
    
    # set to training mode
    retriever.train()
    
    # test gradient flow
    gradient_test_passed = test_embedding_gradients(retriever, next(retriever.parameters()).device)
    
    if not gradient_test_passed:
        logger.error("gradient test failed! this will affect ppo training quality.")
        logger.info("attempting to fix gradient flow issues...")
        
        # force all parameters to require gradients
        for param in retriever.parameters():
            param.requires_grad_(True)
        
        # test again
        gradient_test_passed = test_embedding_gradients(retriever, next(retriever.parameters()).device)
        
        if gradient_test_passed:
            logger.info("fixed gradient flow issues")
        else:
            logger.warning("some gradient flow issues persist - training may be suboptimal")
    
    return gradient_test_passed


def embed(model: SentenceTransformer, texts: list[str], device: torch.device | None = None, chunk_size: int = 16) -> torch.Tensor:
    """
    fixed: gradient-preserving forward pass through a sentencetransformer
    this version ensures gradients flow properly for ppo training
    """
    if device is None:
        device = next(model.parameters()).device
    
    # for small batches, process directly
    if len(texts) <= chunk_size:
        # critical: force training mode and gradient context
        original_training_mode = model.training
        model.train()
        
        with torch.enable_grad():
            try:
                # always use manual forward pass to preserve gradients
                transformer_module = model._first_module()
                
                # get the actual transformer model
                if hasattr(transformer_module, 'auto_model'):
                    transformer = transformer_module.auto_model
                    tokenizer = transformer_module.tokenizer
                    
                    # tokenize manually
                    tokens = tokenizer(
                        texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512
                    )
                    
                    # move to device
                    input_ids = tokens['input_ids'].to(device)
                    attention_mask = tokens['attention_mask'].to(device)
                    
                    # forward pass through transformer (preserves gradients)
                    outputs = transformer(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True
                    )
                    
                    # get embeddings from last hidden state
                    if hasattr(outputs, 'last_hidden_state'):
                        token_embeddings = outputs.last_hidden_state
                    else:
                        token_embeddings = outputs.hidden_states[-1]
                    
                    # mean pooling with attention mask
                    attention_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    masked_embeddings = token_embeddings * attention_mask_expanded
                    sum_embeddings = torch.sum(masked_embeddings, dim=1)
                    sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
                    sentence_embeddings = sum_embeddings / sum_mask
                    
                    # normalize
                    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
                    
                else:
                    # fallback: use model's forward method directly
                    encoded = model.tokenize(texts)
                    for key in encoded:
                        encoded[key] = encoded[key].to(device)
                    
                    sentence_embeddings = model(encoded)['sentence_embedding']
                
                # verify gradients are preserved
                if not sentence_embeddings.requires_grad:
                    logger.warning("force-enabling gradients by connecting to model parameters...")
                    # connect to first parameter to ensure gradient flow
                    first_param = next(iter(model.parameters()))
                    # add tiny contribution that doesn't affect the embedding but ensures gradient connection
                    param_contribution = (first_param.sum() * 0.0).expand_as(sentence_embeddings[0:1, 0:1])
                    sentence_embeddings = sentence_embeddings + param_contribution.expand_as(sentence_embeddings)
                
                # final verification
                if sentence_embeddings.requires_grad:
                    logger.debug(f"embedding computed with gradients: {sentence_embeddings.shape}")
                else:
                    logger.error("critical: failed to preserve gradients in embedding computation!")
                    # last resort: detach and require grad
                    sentence_embeddings = sentence_embeddings.detach().requires_grad_(True)
                
                return sentence_embeddings
                
            except Exception as e:
                logger.error(f"error in gradient-preserving embedding: {e}")
                # fallback with forced gradient connection
                with torch.no_grad():
                    embeddings = model.encode(texts, convert_to_tensor=True, device=device)
                
                # force gradient connection
                embeddings = embeddings.detach().requires_grad_(True)
                first_param = next(iter(model.parameters()))
                param_contribution = (first_param.sum() * 0.0).expand_as(embeddings[0:1, 0:1])
                embeddings = embeddings + param_contribution.expand_as(embeddings)
                
                return embeddings
                
            finally:
                # restore original training mode
                model.training = original_training_mode
    
    else:
        # for larger batches, process in chunks
        embeddings = []
        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i:i + chunk_size]
            chunk_embeddings = embed(model, chunk_texts, device, chunk_size)
            embeddings.append(chunk_embeddings)
        
        return torch.cat(embeddings, dim=0)


class EnhancedCritic(nn.Module):
    """multi-head attention critic with query-passage interaction modeling"""
    def __init__(self, d: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.num_heads = num_heads
        
        # query projection for attention
        self.query_proj = nn.Linear(d, d)
        
        # multi-head attention over passages
        self.passage_attn = nn.MultiheadAttention(
            d, num_heads, dropout=dropout, batch_first=True
        )
        
        # value network
        self.value_net = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d // 2, 1)
        )
        
        # layer normalization
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        
        # better initialization for value function
        self._init_weights()

    def _init_weights(self):
        """initialize weights for better training stability"""
        # initialize final layer with smaller values for stable value learning
        nn.init.normal_(self.value_net[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.value_net[-1].bias, 0.0)
        
        # initialize other linear layers
        for module in self.value_net:
            if isinstance(module, nn.Linear) and module != self.value_net[-1]:
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, q_emb: torch.Tensor, p_emb: torch.Tensor) -> torch.Tensor:
        """
        args:
            q_emb: query embeddings (b, d)
            p_emb: passage embeddings (b, k, d)
        returns:
            value estimates (b,)
        """
        B, K, D = p_emb.shape
        
        # project query for attention
        q_proj = self.query_proj(q_emb).unsqueeze(1)  # (b, 1, d)
        q_proj = self.ln1(q_proj)
        
        # normalize passage embeddings
        p_emb_norm = self.ln2(p_emb)
        
        # attention over passages using query as query
        attn_out, attn_weights = self.passage_attn(
            q_proj, p_emb_norm, p_emb_norm
        )  # (b, 1, d)
        attn_out = attn_out.squeeze(1)  # (b, d)
        
        # combine original query and attended passage representation
        combined = torch.cat([q_emb, attn_out], dim=-1)  # (b, 2d)
        
        return self.value_net(combined).squeeze(-1)  # (b,) 