from dataclasses import dataclass

@dataclass
class LLMConfig:
    vocab_size: int = 500  # Will be overridden by tokenizer length
    max_seq_len: int = 256
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    dropout: float = 0.1
    device: str = "cpu"
