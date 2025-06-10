import tiktoken
import torch
import random
import hashlib
from datasets import load_dataset, Dataset as HFDataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from config import DATA_CONFIG, TRAINING_CONFIG, SEED
from typing import List, Dict, Tuple, Optional


class TextDataset(Dataset):
    """Simple dataset for causal language modeling."""
    
    def __init__(self, texts: List[str], tokenizer: tiktoken.Encoding, max_length: int, pbar_label: str = "Processing"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples: List[List[int]] = []
        
        # Tokenize all texts with progress bar
        for text in tqdm(texts, desc=f"{pbar_label}:", unit="text"):
            tokens = tokenizer.encode(text)
            
            # Create training examples with sliding window
            for i in range(0, len(tokens) - max_length, max_length // 2):
                chunk = tokens[i:i + max_length + 1]
                if len(chunk) == max_length + 1:
                    self.examples.append(chunk)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokens = self.examples[idx]
        return {
            'input_ids': torch.tensor(tokens[:-1], dtype=torch.long),
            'labels': torch.tensor(tokens[1:], dtype=torch.long)
        }


def create_splits(dataset: HFDataset, seed: int = SEED) -> Dict[str, List[str]]:
    """Create secure splits using content hashing to prevent data leakage."""
    random.seed(seed)
    
    # Use content hash for deterministic splitting
    splits: Dict[str, List[str]] = {'train': [], 'val': [], 'test': []}
    
    for example in tqdm(dataset, desc="Splitting data", unit="example"):
        # Create hash of text content
        content_hash = hashlib.sha256(example['text'].encode('utf-8')).hexdigest()
        hash_int = int(content_hash[:8], 16) % 100  # Use first 8 chars for percentage
        
        if hash_int < DATA_CONFIG['train_split'] * 100:
            splits['train'].append(example['text'])
        elif hash_int < (DATA_CONFIG['train_split'] + DATA_CONFIG['val_split']) * 100:
            splits['val'].append(example['text'])
        else:
            splits['test'].append(example['text'])
    
    return splits


def validate_splits(train_texts: List[str], val_texts: List[str], test_texts: List[str]) -> bool:
    """Basic validation to check for data leakage between splits."""
    
    train_set = set(train_texts)
    val_set = set(val_texts)
    test_set = set(test_texts)
    
    # Check for overlaps
    train_val_overlap = len(train_set & val_set)
    train_test_overlap = len(train_set & test_set)
    val_test_overlap = len(val_set & test_set)
    
    total_overlaps = train_val_overlap + train_test_overlap + val_test_overlap
    
    if total_overlaps > 0:
        print(f"WARNING: Found {total_overlaps} overlapping texts between splits")
        print(f"  Train-Val: {train_val_overlap}, Train-Test: {train_test_overlap}, Val-Test: {val_test_overlap}")
        return False
    
    return True


# Cache for tokenized datasets to avoid re-tokenization during tuning
_dataset_cache: Dict[Tuple[int, str], Tuple[TextDataset, TextDataset, TextDataset]] = {}

def get_tokenizer() -> tiktoken.Encoding:
    """Get the tokenizer."""
    return tiktoken.get_encoding(DATA_CONFIG['tokenizer_name'])


def get_cached_datasets() -> Tuple[TextDataset, TextDataset, TextDataset]:
    """Get or create cached tokenized datasets."""
    global _dataset_cache
    
    cache_key = (DATA_CONFIG['max_length'], DATA_CONFIG['tokenizer_name'])
    
    if cache_key not in _dataset_cache:
        # Load dataset with progress indication
        dataset = load_dataset(DATA_CONFIG['dataset_name'])
        full_data: HFDataset = dataset['train']
        
        # Limit dataset size if specified
        if DATA_CONFIG.get('max_examples') is not None:
            max_size = min(len(full_data), DATA_CONFIG['max_examples'])
            print(f"Development mode: {max_size} examples")
            full_data = full_data.select(range(max_size))
        
        splits = create_splits(full_data)
        
        train_texts = splits['train']
        val_texts = splits['val']
        test_texts = splits['test']
        
        print(f"Data splits - Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
        
        validate_splits(train_texts, val_texts, test_texts)
        tokenizer = get_tokenizer()
        
        # Create datasets
        print("\nTokenizing data...")
        train_dataset = TextDataset(train_texts, tokenizer, DATA_CONFIG['max_length'], "Training")
        val_dataset = TextDataset(val_texts, tokenizer, DATA_CONFIG['max_length'], "Validation")
        test_dataset = TextDataset(test_texts, tokenizer, DATA_CONFIG['max_length'], "Testing")
        
        _dataset_cache[cache_key] = (train_dataset, val_dataset, test_dataset)
    else:
        print("Using cached tokenized datasets...")
        train_dataset, val_dataset, test_dataset = _dataset_cache[cache_key]
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for training, validation, and testing."""
    # Get cached datasets
    train_dataset, val_dataset, test_dataset = get_cached_datasets()
    
    # Use provided batch size or default from config
    effective_batch_size = batch_size if batch_size is not None else TRAINING_CONFIG['batch_size']
    


    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=0, # There are issues with multiprocessing on Windows and our data is small enough to not need it
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
