#!/usr/bin/env python3
"""
Main entry point for running GPT language model.

Usage:
python main.py              # Train the model
python main.py --tune       # Run hyperparameter tuning
python main.py --evaluate   # Evaluate model quality
python main.py --generate   # Generate text using a saved model
"""

import torch
import argparse
import os
import json

from config import DEVICE, SEED, MODEL_CONFIG, LOG_DIR, GENERATION_CONFIG
from gpt import GPTLanguageModel
from trainer import train_model, generate_text, run_hyperparameter_tuning as run_tuning
from evaluator import quick_evaluation
from data_loader import get_tokenizer


def print_header():
    """Print a header for the program."""
    print("="*60)
    print("           GPT Language Model Training")
    print("="*60)
    
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f"Device: {device_name}")
    print(f"Seed: {SEED}")
    print(f"Model: {MODEL_CONFIG['num_layers']} layers, {MODEL_CONFIG['embed_size']} dims")
    print("")


def save_model(model, path, params=None):
    """Save the trained model."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

    if params:
        param_path = path.replace('.pt', '_params.json')
        with open(param_path, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"Model parameters saved to {param_path}")


def load_model(path, custom_config=None):
    """Load a saved model."""
    model = GPTLanguageModel(custom_config)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    return model


def run_training():
    """Run model training."""    
    # Initialize model
    model = GPTLanguageModel()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train the model
    trained_model, val_loss, val_perplexity = train_model(model)
    
    # Save the trained model
    model_path = os.path.join(LOG_DIR, 'gpt_model.pt')
    save_model(trained_model, model_path)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Tuning Data\t\tLoss: {val_loss:.4f},\tPerplexity: {val_perplexity:.2f}")
    print("="*60)
    
    return trained_model


def load_latest_model():
    """Load the latest saved model."""
    model_path = os.path.join(LOG_DIR, 'gpt_model.pt')
    best_model_path = os.path.join(LOG_DIR, 'best_gpt_model.pt')
    
    # Try to load best model first, then regular model
    if os.path.exists(best_model_path):
        print("Loading best tuned model...")
        best_params_path = best_model_path.replace('.pt', '_params.json')
        if not os.path.exists(best_params_path):
            print(f"Error: Parameter file not found at {best_params_path}")
            return None

        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        return load_model(best_model_path, custom_config=best_params)

    elif os.path.exists(model_path):
        print("Loading trained model...")
        return load_model(model_path)
    else:
        return None


def run_hyperparameter_tuning():
    """Run hyperparameter optimization."""     
    best_params, best_loss = run_tuning()
    
    print("\n" + "="*60)
    print("Hyperparameter Tuning Complete!")
    print(f"Best Validation Loss: {best_loss:.4f}")
    print("Best Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print("="*60)
    
    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    final_model = GPTLanguageModel(best_params)
    trained_model, _, _ = train_model(final_model, custom_config=best_params)
    
    # Save the best model
    best_model_path = os.path.join(LOG_DIR, 'best_gpt_model.pt')
    save_model(trained_model, best_model_path, best_params)
    
    return trained_model


def run_evaluation():
    """Evaluate model quality."""
    model = load_latest_model()
    if model is None:
        print("No saved model found. Training a model first...")
        model = run_training()

    # Run evaluation
    results = quick_evaluation(model)
    return results


def run_generation():
    """Generate text using a saved model."""
    model = load_latest_model()
    if model is None:
        print("No saved model found. Please train a model first.")
        return
    
    tokenizer = get_tokenizer()
    
    print("\nGenerating text samples...")
    print("="*60)
    
    # Generate multiple samples with different prompts
    prompts = GENERATION_CONFIG['prompts']
    max_length = GENERATION_CONFIG['max_length']
    temperature = GENERATION_CONFIG['temperature']

    for i, prompt in enumerate(prompts, 1):
        print(f"\nSample {i}:")
        print(f"Prompt: {prompt}")
        generated_text = generate_text(model, tokenizer, prompt, max_length=max_length, temperature=temperature)
        
        # Extract only the generated part
        generated_part = generated_text[len(prompt):].strip()
        print(f"Generated: {generated_part}")
        print("-" * 40)


def clear_all():
    """Clear all saved models, parameters, and tuning data."""
    files_to_delete = [
        os.path.join(LOG_DIR, 'gpt_model.pt'),
        os.path.join(LOG_DIR, 'gpt_model_params.json'),
        os.path.join(LOG_DIR, 'best_gpt_model.pt'),
        os.path.join(LOG_DIR, 'best_gpt_model_params.json'),
        'study.db'  # Optuna database
    ]
    
    deleted = False
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            deleted = True
    
    if not deleted:
        print("No files found to clear.")
    else:
        print("\nAll model files and tuning data have been cleared.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='GPT Language Model')
    parser.add_argument('--tune', action='store_true', 
                       help='Run hyperparameter tuning')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model quality')
    parser.add_argument('--generate', action='store_true', 
                       help='Generate text using a saved model')
    parser.add_argument('--clear', action='store_true',
                       help='Clear all saved models and tuning data')
    args = parser.parse_args()
    
    print_header()
    
    try:
        if args.clear:
            clear_all()
        elif args.tune:
            run_hyperparameter_tuning()
        elif args.evaluate:
            run_evaluation()
        elif args.generate:
            run_generation()
        else:
            run_training()
            
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
