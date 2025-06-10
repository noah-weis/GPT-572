# GPT Language Model

import torch
import torch.nn as nn
import math
from config import DATA_CONFIG, MODEL_CONFIG


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_size, num_heads, block_size, dropout=None):
        super().__init__()
        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.dropout_rate = dropout if dropout is not None else MODEL_CONFIG['dropout']
        
        # Combined linear layer for Q, K, V
        self.qkv_proj = nn.Linear(embed_size, 3 * embed_size, bias=False)
        self.output_proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Causal mask
        self.register_buffer("causal_mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape  # Batch size, sequence length, embedding dimensionality
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Final output projection
        out = self.output_proj(out)
        out = self.dropout(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, embed_size, dropout=None):
        super().__init__()
        dropout_rate = dropout if dropout is not None else MODEL_CONFIG['dropout']
        
        self.net = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward."""
    
    def __init__(self, embed_size, num_heads, block_size, dropout=None):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads, block_size, dropout)
        self.feed_forward = FeedForward(embed_size, dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Attention with residual connection
        x = x + self.attention(self.ln1(x))
        # Feed-forward with residual connection
        x = x + self.feed_forward(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, custom_config=None):
        super().__init__()
        
        # Use custom config if provided, otherwise use default
        if custom_config is not None:
            config = {**MODEL_CONFIG, **custom_config}
        else:
            config = MODEL_CONFIG
        
        # Model configuration
        self.vocab_size = DATA_CONFIG['vocab_size']
        self.embed_size = config['embed_size']
        self.block_size = config.get('block_size', MODEL_CONFIG['block_size'])
        self.num_heads = config.get('num_heads', MODEL_CONFIG['num_heads'])
        self.num_layers = config['num_layers']
        
        assert self.embed_size % self.num_heads == 0, f"embed_size ({self.embed_size}) must be divisible by num_heads ({self.num_heads})"
        
        # Model layers
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.position_embedding = nn.Embedding(self.block_size, self.embed_size)
        
        self.transformer_blocks = nn.Sequential(*[
            TransformerBlock(self.embed_size, self.num_heads, self.block_size, config.get('dropout', MODEL_CONFIG['dropout'])) 
            for _ in range(self.num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(self.embed_size)
        self.output_head = nn.Linear(self.embed_size, self.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Ensure sequence length doesn't exceed block size
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            if targets is not None:
                targets = targets[:, -self.block_size:]
            T = self.block_size
        
        # Get embeddings
        token_emb = self.token_embedding(idx)  # (B, T, embed_size)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))  # (T, embed_size)
        x = token_emb + pos_emb
        
        # Apply transformer blocks
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        logits = self.output_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate a sequence of tokens"""
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop sequence to block size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Get predictions
            logits, _ = self(idx_cond)
            
            # Apply temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            idx = torch.cat([idx, next_token], dim=1)
        
        self.train()
        return idx
    
