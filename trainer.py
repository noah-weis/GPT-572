# Training Module

import torch
import math
from tqdm import tqdm

from config import DEVICE, TRAINING_CONFIG, SEED, TUNING_CONFIG
from data_loader import create_dataloaders
from gpt import GPTLanguageModel

import optuna
from optuna.samplers import TPESampler



def cosine_lr_schedule(step, max_steps, lr_max, lr_min=None, warmup_steps=None):
    """Cosine learning rate schedule with warmup."""
    if lr_min is None:
        lr_min = lr_max * 0.1
    if warmup_steps is None:
        warmup_steps = max_steps // 10
    
    if step < warmup_steps: # Linear warmup
        return lr_max * step / warmup_steps
    elif step > max_steps:
        return lr_min
    else: # Cosine decay
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return lr_min + coeff * (lr_max - lr_min)


def evaluate_model(model, data_loader, max_batches=None, show_progress=False):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # For progress bar :)
    total_batches = max_batches if max_batches else len(data_loader)
    
    with torch.no_grad():
        # Show progress bar only when explicitly requested (final evaluations)
        batch_iter = enumerate(data_loader)
        if show_progress:
            batch_iter = tqdm(batch_iter, total=total_batches, desc="Evaluating", leave=False)
        
        for batch_idx, batch in batch_iter:
            if max_batches and batch_idx >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            _, loss = model(input_ids, labels)
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar if it exists
            if show_progress and hasattr(batch_iter, 'set_postfix'):
                batch_iter.set_postfix({'avg_loss': f"{total_loss / num_batches:.4f}"})
        
        # Close progress bar if it was created
        if show_progress and hasattr(batch_iter, 'close'):
            batch_iter.close()
    
    model.train()
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity


def objective(trial, model_class):
    """Objective function for Optuna hyperparameter tuning."""
    # Define hyperparameter search space
    search_space = {
        'learning_rate': trial.suggest_categorical('learning_rate', [1e-4, 3e-4, 5e-4]),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 24]),
        'embed_size': trial.suggest_categorical('embed_size', [128, 256, 384]),
        'num_layers': trial.suggest_categorical('num_layers', [4, 6, 8]),
        'dropout': trial.suggest_categorical('dropout', [0.05, 0.1, 0.15])
    }
    
    # Create model with suggested hyperparameters
    model = model_class(custom_config=search_space)
    
    # Train and evaluate the model
    val_loss = train_model(model, custom_config=search_space, trial=trial)
    
    return val_loss


def train_model(model, custom_config=None, trial=None):
    """Train the GPT model with optional hyperparameter tuning support."""
    
    if not trial:
        print("\nPreprocessing...\n" + "-" * 60)
    
    torch.manual_seed(SEED)

    
    # Setup trainis_tuninging configuration
    config = {**TRAINING_CONFIG}
    if custom_config:
        config.update(custom_config)
        
    # Load data
    train_loader, val_loader, _ = create_dataloaders(config['batch_size'])
    
    model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    max_iters = config['max_iters']
    eval_interval = config['eval_interval']
    eval_iters = config['eval_iters']

    if not trial:
        print(f"\nTraining...\n" + "-" * 60)
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    patience = config['early_stopping_patience']
    
    # Training loop
    step = 0
    train_iter = iter(train_loader)
    
    # Progress bar description
    if trial:
        desc = f"Tuning (Trial {trial.number})"
        leave = False
    else:
        desc = "Training"
        leave = True
    
    progress_bar = tqdm(total=max_iters, desc=desc, leave=leave)
    
    while step < max_iters:
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Update learning rate
        lr = cosine_lr_schedule(step, max_iters, config['learning_rate'])
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Forward pass
        input_ids = batch['input_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        
        logits, loss = model(input_ids, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{lr:.6f}"
        })
        progress_bar.update(1)
        
        # Intermediate Evaluation
        if step > 0 and step % eval_interval == 0:
            val_loss, val_perplexity = evaluate_model(model, val_loader, eval_iters)


            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f"Early stopping at step {step} (patience {patience} exceeded)")
                    break
            
            # Report intermediate value for pruning (tuning mode only)
            if trial:
                trial.report(val_loss, step)
                if trial.should_prune():
                    progress_bar.close()
                    raise optuna.TrialPruned()
            
            # Clear GPU memory
            torch.cuda.empty_cache()
        
        step += 1
    
    progress_bar.close()
    
    # Final evaluation
    if trial:
        # More thorough final evaluation for tuning
        val_loss, val_perplexity = evaluate_model(model, val_loader, 100)
        return val_loss  # Return only validation loss for tuning
    else:
        # Full evaluation with progress bar for regular training
        val_loss, val_perplexity = evaluate_model(model, val_loader, show_progress=True)
        return model, val_loss, val_perplexity


def generate_text(model, tokenizer, prompt="Once upon a time", max_length=100, temperature=0.8):
    """Generate text using the trained model."""
    model.eval()
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=DEVICE)
    
    # Generate
    generated = model.generate(input_ids, max_length, temperature)
    
    # Decode
    generated_text = tokenizer.decode(generated[0].cpu().tolist())
    
    model.train()
    return generated_text


def run_hyperparameter_tuning():
    """Run hyperparameter optimization using Optuna."""
    print("Starting hyperparameter tuning...")
    print(f"Will run {TUNING_CONFIG['n_trials']} trials")
    
    # Create or load study
    storage = optuna.storages.RDBStorage(
        url="sqlite:///study.db",
        engine_kwargs={"connect_args": {"timeout": 100}}
    )
    
    study = optuna.create_study(
        study_name='gpt_optimization',
        direction='minimize',
        sampler=TPESampler(seed=SEED),
        storage=storage,
        load_if_exists=True
    )
    
    # Define objective with model class
    obj_fn = lambda trial: objective(trial, GPTLanguageModel)
    
    try:
        # Run optimization
        study.optimize(obj_fn, n_trials=TUNING_CONFIG['n_trials'])
    except KeyboardInterrupt:
        print("\nStudy interrupted! Progress has been saved.")
        print(f"Completed trials: {len(study.trials)}")
    
    return study.best_params, study.best_value

