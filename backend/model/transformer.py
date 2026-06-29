import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = self.d_model // self.n_heads
        
        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.o_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B, h, T, d)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # QK^T / sqrt(d_head)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
            
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v # (B, h, T, d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.o_proj(y)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffwd = FeedForward(config)
        
    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffwd(self.ln2(x))
        return x

class LLM8086(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.max_seq_len, f"Cannot forward sequence of length {T}, max_seq_len is {self.config.max_seq_len}"
        
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        tok_emb = self.token_embedding(idx) # (B, T, d_model)
        pos_emb = self.position_embedding(pos) # (T, d_model)
        
        x = tok_emb + pos_emb
        
        # Causal mask (lower triangular)
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_v = logits.view(B*T, C)
            targets_v = targets.view(B*T)
            loss = F.cross_entropy(logits_v, targets_v)
            
        return logits, loss
