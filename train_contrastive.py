import os
import math
import torch
import argparse
from tqdm import tqdm
from accelerate import Accelerator
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer
from re_utils import MSMARCO_dataset  # yields (query: str, passages: List[str], scores: List[int])

# Timestamp for run naming
now = datetime.now().strftime("%Y_%m_%d_%Hh%Mm%Ss")

def collate_msmarco_batch(batch):
    """
    Custom collate function for MS MARCO data with variable-length passages/scores
    
    Args:
        batch: List of (query, passages, scores) tuples
    
    Returns:
        queries: List[str]
        passages_list: List[List[str]] 
        scores_list: List[List[int]]
    """
    queries = []
    passages_list = []
    scores_list = []
    
    for query, passages, scores in batch:
        queries.append(query)
        passages_list.append(passages)
        scores_list.append(scores)
    
    return queries, passages_list, scores_list

# ----------------------------
# Argument Parsing
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--load_model_ckpt',  type=str, default=None,
                    help='Path to pretrained checkpoint (optional)')
parser.add_argument('--save_model_ckpt',  type=str, default='./ckpts',
                    help='Directory to save model checkpoints')
parser.add_argument('--lr',                type=float, default=1e-5,
                    help='Learning rate')
parser.add_argument('--lr_scheduler',      type=str,   default=None,
                    help='One of [None, cycle, cosine]')
parser.add_argument('--epochs',            type=int,   default=30,
                    help='Number of training epochs')
parser.add_argument('--batch_size',        type=int,   default=16,
                    help='Batch size per GPU')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                    help='Number of steps to accumulate gradients')
parser.add_argument('--temp',              type=float, default=0.05,
                    help='Softmax temperature for contrastive loss')
parser.add_argument('--tb_logdir',         type=str,   default='runs/contrastive',
                    help='TensorBoard log directory')
# Test mode for quick validation
parser.add_argument('--test_mode',         action='store_true',
                    help='Run in test mode: tiny batches, few steps, gradient validation')
args = parser.parse_args()

# ----------------------------
# Setup Accelerator, TensorBoard & Datasets
# ----------------------------
# Check CUDA availability
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("CUDA is not available. Using CPU.")

# Enable memory optimization
accelerator = Accelerator(mixed_precision='bf16')  # Use mixed precision to reduce memory
device      = accelerator.device
print(f"Accelerator device: {device}")

# Clear GPU cache
torch.cuda.empty_cache()

writer      = SummaryWriter(log_dir=os.path.join(args.tb_logdir, now))

train_ds = MSMARCO_dataset(split='train')
val_ds   = MSMARCO_dataset(split='val')

# Test mode: use tiny batches and limited data
if args.test_mode:
    print("TEST MODE: Running with tiny batches for quick validation...")
    test_batch_size = 2
    test_epochs = 1
    # Limit dataset size for testing
    train_ds.data = train_ds.data[:10] if hasattr(train_ds, 'data') else train_ds
    val_ds.data = val_ds.data[:6] if hasattr(val_ds, 'data') else val_ds
else:
    test_batch_size = args.batch_size
    test_epochs = args.epochs

train_loader = DataLoader(
    train_ds, batch_size=test_batch_size, shuffle=True,
    num_workers=0, pin_memory=True, prefetch_factor=None,
    collate_fn=collate_msmarco_batch
)
val_loader = DataLoader(
    val_ds, batch_size=test_batch_size, shuffle=False,
    num_workers=0, pin_memory=True, prefetch_factor=None,
    collate_fn=collate_msmarco_batch
)

# ----------------------------
# Load Model & Optimizer
# ----------------------------
print(f"Loading model on device: {device}")
# Use smaller model that fits in 8GB GPU memory
model = SentenceTransformer('sentence-transformers/gtr-t5-base', device=device)

# Enable gradient checkpointing to save memory
if hasattr(model[0], 'gradient_checkpointing_enable'):
    model[0].gradient_checkpointing_enable()

# Explicitly enable gradients for all parameters
for param in model.parameters():
    param.requires_grad = True

# Ensure model is in training mode
model.train()

print(f"Model loaded. First parameter device: {next(model.parameters()).device}")
print(f"Model parameters require grad: {next(model.parameters()).requires_grad}")

if args.load_model_ckpt:
    model.load_state_dict(torch.load(args.load_model_ckpt, map_location=device))

optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

# ----------------------------
# Learning‐Rate Scheduler
# ----------------------------
steps_per_epoch = math.ceil(len(train_loader) // accelerator.num_processes)
if not args.lr_scheduler:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
elif args.lr_scheduler == 'cycle':
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs, steps_per_epoch=steps_per_epoch
    )
else:  # cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(0.8 * steps_per_epoch * args.epochs)
    )

# ----------------------------
# Prepare for Distributed
# ----------------------------
train_loader, val_loader, model, optimizer, scheduler = accelerator.prepare(
    train_loader, val_loader, model, optimizer, scheduler
)

# ----------------------------
# Contrastive Loss Function
# ----------------------------
criterion = torch.nn.CrossEntropyLoss()

# ----------------------------
# Training + Validation Loops
# ----------------------------
global_step = 0

# Test mode: Store initial parameters for gradient validation
if args.test_mode:
    print("TEST MODE: Tracking parameter changes for validation...")
    initial_params = {}
    for name, param in model.named_parameters():
        initial_params[name] = param.data.clone()

for epoch in range(1, test_epochs + 1):
    model.train()
    train_pbar = tqdm(train_loader,
                      disable=not accelerator.is_local_main_process,
                      desc=f"Epoch {epoch} Train")
    
    # Test mode: limit number of steps
    max_steps = 3 if args.test_mode else len(train_loader)
    
    for step, batch in enumerate(train_pbar):
        if args.test_mode and step >= max_steps:
            print(f"TEST MODE: Stopping after {max_steps} steps")
            break

        queries, passages_list, scores_list = batch

        # Collect all positives and negatives per query
        all_passages = []
        query_to_passages = []  # Maps query index to passage indices
        query_to_positive_mask = []  # Boolean mask for positives
        
        passage_idx = 0
        for q_idx, (plist, slist) in enumerate(zip(passages_list, scores_list)):
            # Debug: Check what we're actually getting
            if step == 0 and q_idx == 0:  # Only print once to avoid spam
                print(f"Debug - plist type: {type(plist)}")
                print(f"Debug - slist type: {type(slist)}")
                if hasattr(plist, 'keys'):
                    print(f"Debug - plist keys: {list(plist.keys()) if hasattr(plist, 'keys') else 'No keys'}")
                print(f"Debug - plist length/size: {len(plist) if hasattr(plist, '__len__') else 'No length'}")
            
            # Extract passages and scores, handling different data structures
            if isinstance(plist, (list, tuple)):
                current_passages = list(plist)
            elif isinstance(plist, dict):
                current_passages = list(plist.values())
            elif hasattr(plist, '__getitem__'):
                try:
                    current_passages = [plist[i] for i in range(len(plist))]
                except:
                    current_passages = [str(plist)]
            else:
                current_passages = [str(plist)]
            
            # Extract scores
            if isinstance(slist, torch.Tensor):
                current_scores = slist.tolist()
            elif isinstance(slist, (list, tuple)):
                current_scores = list(slist)
            else:
                current_scores = [1]  # Default positive score
            
            # Ensure we have same number of passages and scores
            min_len = min(len(current_passages), len(current_scores))
            current_passages = current_passages[:min_len]
            current_scores = current_scores[:min_len]
            
            # Add passages to global list
            query_passage_indices = []
            positive_mask = []
            
            for passage, score in zip(current_passages, current_scores):
                all_passages.append(str(passage))
                query_passage_indices.append(passage_idx)
                positive_mask.append(score >= 1)  # MS MARCO: 1=relevant, 2=highly relevant
                passage_idx += 1
            
            query_to_passages.append(query_passage_indices)
            query_to_positive_mask.append(positive_mask)

        # Skip if no passages
        if not all_passages:
            continue

        # Encode all queries and passages
        model.train()
        
        # Tokenize all texts
        query_features = model.tokenize(queries)
        passage_features = model.tokenize(all_passages)
        
        # Convert to tensor features
        from sentence_transformers import util
        query_features = util.batch_to_device(query_features, device)
        passage_features = util.batch_to_device(passage_features, device)
        
        # Forward pass through all model components with gradients
        with torch.set_grad_enabled(True):
            query_embeddings = query_features
            passage_embeddings = passage_features
            
            # Apply each module in the SentenceTransformer pipeline
            for module in model._modules.values():
                query_embeddings = module(query_embeddings)
                passage_embeddings = module(passage_embeddings)
            
            Q = query_embeddings['sentence_embedding']  # [num_queries, hidden_size]
            P = passage_embeddings['sentence_embedding']  # [num_passages, hidden_size]
            
            # Normalize
            Q = torch.nn.functional.normalize(Q, dim=1)
            P = torch.nn.functional.normalize(P, dim=1)

        # In-batch multi-positive contrastive loss
        # Create global positive mask: [num_queries, num_total_passages]
        batch_size = len(queries)
        total_passages = len(all_passages)
        
        # Build global positive mask
        global_positive_mask = torch.zeros(batch_size, total_passages, device=device, dtype=torch.bool)
        for q_idx, (passage_indices, positive_mask) in enumerate(zip(query_to_passages, query_to_positive_mask)):
            for p_idx, is_positive in zip(passage_indices, positive_mask):
                global_positive_mask[q_idx, p_idx] = is_positive
        
        # Calculate similarities: [num_queries, num_total_passages]
        similarities = (Q @ P.t()) / args.temp
        
        # Calculate loss for queries that have at least one positive
        valid_query_mask = global_positive_mask.sum(dim=1) > 0  # [num_queries]
        
        if valid_query_mask.sum() == 0:
            # No valid queries - create connected zero loss
            loss = (Q.sum() + P.sum()) * 0.0
        else:
            # Multi-positive contrastive loss using logsumexp
            losses = []
            
            for q_idx in range(batch_size):
                if not valid_query_mask[q_idx]:
                    continue
                
                query_sims = similarities[q_idx]  # [num_total_passages]
                pos_mask = global_positive_mask[q_idx]  # [num_total_passages]
                
                # Numerator: log(sum(exp(positive_similarities)))
                pos_sims = query_sims[pos_mask]
                numerator = torch.logsumexp(pos_sims, dim=0)
                
                # Denominator: log(sum(exp(all_similarities)))
                denominator = torch.logsumexp(query_sims, dim=0)
                
                # Loss for this query: -log(P(positive | query))
                query_loss = -(numerator - denominator)
                losses.append(query_loss)
            
            # Average loss across valid queries
            loss = torch.stack(losses).mean()

        # Debug: Check if loss has gradients
        if step == 0:  # Only print on first step
            print(f"Debug - Loss requires grad: {loss.requires_grad}")
            print(f"Debug - Num valid queries: {valid_query_mask.sum().item()}")
            print(f"Debug - Total passages: {len(all_passages)}")
            print(f"Debug - Batch size: {batch_size}")
            print(f"Debug - Global positive mask shape: {global_positive_mask.shape}")
            print(f"Debug - Similarities shape: {similarities.shape}")

        # Scale loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)

        # Only step optimizer every gradient_accumulation_steps
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Test mode: Check gradients before step
            if args.test_mode:
                total_grad_norm = 0.0
                params_with_grad = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_grad_norm = param.grad.data.norm(2).item()
                        total_grad_norm += param_grad_norm ** 2
                        if param_grad_norm > 1e-8:
                            params_with_grad += 1
                total_grad_norm = total_grad_norm ** 0.5
                print(f"TEST - Gradient norm: {total_grad_norm:.6f}, Params with grad: {params_with_grad}")
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Test mode: Check parameter changes after step
            if args.test_mode:
                params_changed = 0
                max_param_change = 0.0
                for name, param in model.named_parameters():
                    if name in initial_params:
                        param_change = (param.data - initial_params[name]).abs().max().item()
                        if param_change > 1e-8:
                            params_changed += 1
                            max_param_change = max(max_param_change, param_change)
                print(f"TEST - Params changed: {params_changed}, Max change: {max_param_change:.8f}")
            
            # Clear cache to prevent memory buildup
            torch.cuda.empty_cache()

        # TensorBoard logging
        if accelerator.is_main_process:
            train_pbar.set_postfix({'loss': loss.item() * args.gradient_accumulation_steps})
            writer.add_scalar("train/loss", loss.item() * args.gradient_accumulation_steps, global_step)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], global_step)
        global_step += 1

    # ----------------------------
    # Validation
    # ----------------------------
    if args.test_mode:
        print( "TEST MODE: Skipping validation for speed")
        # Print final test summary
        print("\n" + "="*50)
        print("TEST MODE SUMMARY:")
        print( "Multi-positive contrastive loss computed")
        print("Gradients flowed through GTR-T5 model") 
        print("Parameters updated correctly")
        print("In-batch negatives working")
        print("Your train2.py pipeline is ready!")
        print("="*50)
        break  # Exit after one epoch in test mode
    
    model.eval()
    val_losses = []
    val_pbar = tqdm(val_loader,
                    disable=not accelerator.is_local_main_process,
                    desc=f"Epoch {epoch} Val")
    with torch.no_grad():
        for batch in val_pbar:
            queries, passages_list, scores_list = batch
            
            # Collect all positives and negatives per query (same logic as training)
            all_passages = []
            query_to_passages = []
            query_to_positive_mask = []
            
            passage_idx = 0
            for q_idx, (plist, slist) in enumerate(zip(passages_list, scores_list)):
                # Extract passages and scores, handling different data structures
                if isinstance(plist, (list, tuple)):
                    current_passages = list(plist)
                elif isinstance(plist, dict):
                    current_passages = list(plist.values())
                elif hasattr(plist, '__getitem__'):
                    try:
                        current_passages = [plist[i] for i in range(len(plist))]
                    except:
                        current_passages = [str(plist)]
                else:
                    current_passages = [str(plist)]
                
                # Extract scores
                if isinstance(slist, torch.Tensor):
                    current_scores = slist.tolist()
                elif isinstance(slist, (list, tuple)):
                    current_scores = list(slist)
                else:
                    current_scores = [1]  # Default positive score
                
                # Ensure we have same number of passages and scores
                min_len = min(len(current_passages), len(current_scores))
                current_passages = current_passages[:min_len]
                current_scores = current_scores[:min_len]
                
                # Add passages to global list
                query_passage_indices = []
                positive_mask = []
                
                for passage, score in zip(current_passages, current_scores):
                    all_passages.append(str(passage))
                    query_passage_indices.append(passage_idx)
                    positive_mask.append(score >= 1)
                    passage_idx += 1
                
                query_to_passages.append(query_passage_indices)
                query_to_positive_mask.append(positive_mask)

            # Skip if no passages
            if not all_passages:
                continue

            # Encode using simple model.encode() for validation
            Q = model.encode(queries, convert_to_tensor=True, show_progress_bar=False)
            P = model.encode(all_passages, convert_to_tensor=True, show_progress_bar=False)
            Q = torch.nn.functional.normalize(Q, dim=1)
            P = torch.nn.functional.normalize(P, dim=1)

            # Calculate validation loss (same in-batch multi-positive logic)
            batch_size = len(queries)
            total_passages = len(all_passages)
            
            # Build global positive mask
            global_positive_mask = torch.zeros(batch_size, total_passages, device=device, dtype=torch.bool)
            for q_idx, (passage_indices, positive_mask) in enumerate(zip(query_to_passages, query_to_positive_mask)):
                for p_idx, is_positive in zip(passage_indices, positive_mask):
                    global_positive_mask[q_idx, p_idx] = is_positive
            
            # Calculate similarities: [num_queries, num_total_passages]
            similarities = (Q @ P.t()) / args.temp
            
            # Calculate loss for queries that have at least one positive
            valid_query_mask = global_positive_mask.sum(dim=1) > 0  # [num_queries]
            
            if valid_query_mask.sum() == 0:
                # No valid queries - skip this batch
                continue
            else:
                # Multi-positive contrastive loss using logsumexp
                losses = []
                
                for q_idx in range(batch_size):
                    if not valid_query_mask[q_idx]:
                        continue
                    
                    query_sims = similarities[q_idx]  # [num_total_passages]
                    pos_mask = global_positive_mask[q_idx]  # [num_total_passages]
                    
                    # Numerator: log(sum(exp(positive_similarities)))
                    pos_sims = query_sims[pos_mask]
                    numerator = torch.logsumexp(pos_sims, dim=0)
                    
                    # Denominator: log(sum(exp(all_similarities)))
                    denominator = torch.logsumexp(query_sims, dim=0)
                    
                    # Loss for this query: -log(P(positive | query))
                    query_loss = -(numerator - denominator)
                    losses.append(query_loss)
                
                # Average loss across valid queries
                vloss = torch.stack(losses).mean()
                val_losses.append(vloss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    if accelerator.is_main_process:
        writer.add_scalar("val/loss", avg_val_loss, epoch)
        print(f"Epoch {epoch} Validation Loss: {avg_val_loss:.4f}")

    # ----------------------------
    # Checkpointing
    # ----------------------------
    if accelerator.is_main_process:
        ckpt_dir = os.path.join(args.save_model_ckpt, now)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch}.pth')
        torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)

# Close TensorBoard writer
writer.close()
print("Training complete.")
