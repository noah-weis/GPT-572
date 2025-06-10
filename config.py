# Configuration for GPT Project
import os
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVELOPMENT_MODE = False

SEED = 1337

DATA_CONFIG = {
    'dataset_name': 'roneneldan/TinyStories',
    'max_length': 128,
    'tokenizer_name': 'gpt2',
    'vocab_size': 50257,
    'train_split': 0.7,
    'val_split': 0.1,
    'test_split': 0.2,
    'max_examples': 1000 if DEVELOPMENT_MODE else None,
}

MODEL_CONFIG = {
    'embed_size': 256,
    'num_heads': 8,
    'num_layers': 6,
    'block_size': 128,
    'dropout': 0.1,
}

TRAINING_CONFIG = {
    'batch_size': 16,
    'learning_rate': 3e-4,
    'max_iters': 100 if DEVELOPMENT_MODE else 50000,
    'eval_interval': 10 if DEVELOPMENT_MODE else 5000,
    'eval_iters': 20 if DEVELOPMENT_MODE else 300,
    'early_stopping_patience': 2 if DEVELOPMENT_MODE else 5,
}

TUNING_CONFIG = {
    'n_trials': 5 if DEVELOPMENT_MODE else 50,
    'search_space': {
        'learning_rate': [1e-4, 3e-4, 5e-4],
        'batch_size': [8, 16, 24],
        'embed_size': [128, 256, 384],
        'num_layers': [4, 6, 8],
        'dropout': [0.05, 0.1, 0.15]
    }
}

LOG_DIR = 'models'
os.makedirs(LOG_DIR, exist_ok=True)

GENERATION_CONFIG = {
    'prompts': [
        "Alice was so tired when she got back home so she went",
        "Jack wanted to read a book, so he went to",
        "”Can cows fly?”, Alice asked her mother",
        "What do birds like to eat?”, Tom asked his mother",
        "What language do they speak in France?”, Tom asked his mother",
    ],
    'max_length': 80,
    'temperature': 0.8,
}

