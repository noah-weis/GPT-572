# Simple Text Quality Evaluator

import torch
import math
import re
from collections import Counter
from tqdm import tqdm

from config import DEVICE, GENERATION_CONFIG
from data_loader import get_tokenizer


def generate_sample(model, tokenizer, prompt, max_length=100, temperature=0.8):
    """Generate a single text sample."""
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], device=DEVICE)
    
    with torch.no_grad():
        generated = model.generate(input_ids, max_length, temperature)
        full_text = tokenizer.decode(generated[0].cpu().tolist())
        
    return full_text[len(prompt):].strip()


def calculate_perplexity(model, tokenizer, text):
    """Calculate perplexity for a given text."""
    tokens = tokenizer.encode(text)
    if len(tokens) < 2:
        return float('inf')
    
    input_ids = torch.tensor([tokens[:-1]], device=DEVICE)
    targets = torch.tensor([tokens[1:]], device=DEVICE)
    
    with torch.no_grad():
        _, loss = model(input_ids, targets)
        perplexity = torch.exp(loss).item()
    
    return perplexity


def calculate_diversity(texts):
    """Calculate lexical diversity metrics."""
    all_tokens = []
    for text in texts:
        tokens = re.findall(r'\w+', text.lower())
        all_tokens.extend(tokens)
    
    if not all_tokens:
        return {'diversity': 0.0, 'repetition_rate': 1.0}
    
    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)
    
    # Type-Token Ratio (diversity)
    diversity = unique_tokens / total_tokens
    
    # Repetition rate
    token_counts = Counter(all_tokens)
    repeated_tokens = sum(count - 1 for count in token_counts.values() if count > 1)
    repetition_rate = repeated_tokens / total_tokens
    
    return {
        'diversity': diversity,
        'repetition_rate': repetition_rate,
        'unique_tokens': unique_tokens,
        'total_tokens': total_tokens
    }


def calculate_coherence(text):
    """Simple coherence measure based on sentence structure."""
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 0.5  # Neutral score for single sentence
    
    # Check sentence length consistency
    lengths = [len(s.split()) for s in sentences]
    if not lengths:
        return 0.0
        
    avg_length = sum(lengths) / len(lengths)
    length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
    length_consistency = 1.0 / (1.0 + math.sqrt(length_variance) / max(avg_length, 1.0))
    
    # Simple grammar check (basic capitalization and punctuation)
    grammar_score = 0
    for sentence in sentences:
        if sentence and sentence[0].isupper():
            grammar_score += 1
    grammar_score = grammar_score / len(sentences) if sentences else 0
    
    # Combine metrics
    coherence = (length_consistency * 0.6 + grammar_score * 0.4)
    return min(1.0, max(0.0, coherence))


class TextEvaluator:
    """Simple evaluator for text generation quality."""
    
    def __init__(self, model):
        self.model = model.to(DEVICE).eval()
        self.tokenizer = get_tokenizer()
    
    def evaluate_generation_quality(self, prompts=None, num_samples=5):
        """Evaluate the model's text generation quality."""
        if prompts is None:
            prompts = GENERATION_CONFIG['prompts']
        results = {
            'samples': [],
            'avg_perplexity': 0,
            'avg_coherence': 0,
            'diversity_metrics': {}
        }
        
        all_generated = []
        total_perplexity = 0
        total_coherence = 0
                
        for prompt in prompts[:num_samples]:
            # Generate text
            generated = generate_sample(self.model, self.tokenizer, prompt, max_length=80, temperature=0.8)
            
            # Calculate metrics
            perplexity = calculate_perplexity(self.model, self.tokenizer, prompt + " " + generated)
            coherence = calculate_coherence(generated)
            
            sample_result = {
                'prompt': prompt,
                'generated': generated,
                'perplexity': perplexity,
                'coherence': coherence
            }
            
            results['samples'].append(sample_result)
            all_generated.append(generated)
            
            total_perplexity += perplexity
            total_coherence += coherence
            
        
        # Calculate averages
        results['avg_perplexity'] = total_perplexity / len(prompts)
        results['avg_coherence'] = total_coherence / len(prompts)
        
        # Calculate diversity metrics
        results['diversity_metrics'] = calculate_diversity(all_generated)
        
        return results
    
    def print_evaluation_results(self, results):
        """Print evaluation results in a readable format."""
        print("\n" + "="*60)
        print("TEXT GENERATION QUALITY EVALUATION")
        print("="*60)
        
        print(f"Average Perplexity: {results['avg_perplexity']:.2f}")
        print(f"Average Coherence: {results['avg_coherence']:.3f}")
        
        diversity = results['diversity_metrics']
        print(f"Lexical Diversity: {diversity['diversity']:.3f}")
        print(f"Repetition Rate: {diversity['repetition_rate']:.3f}")
        print(f"Vocabulary: {diversity['unique_tokens']} unique / {diversity['total_tokens']} total")
        
        print("\nSample Outputs:")
        print("-" * 40)
        
        for i, sample in enumerate(results['samples'][:3], 1):
            print(f"\nSample {i}:")
            print(f"Prompt: {sample['prompt']}")
            print(f"Generated: {sample['generated'][:100]}{'...' if len(sample['generated']) > 100 else ''}")
            print(f"Quality: Perplexity={sample['perplexity']:.2f}, Coherence={sample['coherence']:.3f}")


def quick_evaluation(model):
    """Run a quick quality evaluation."""
    evaluator = TextEvaluator(model)
    results = evaluator.evaluate_generation_quality()
    evaluator.print_evaluation_results(results)
    
    return results 