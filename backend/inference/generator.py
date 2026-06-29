import torch
import torch.nn.functional as F
import os
import sys

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import LLM8086
from model.config import LLMConfig
from tokenizer.assembly_tokenizer import AssemblyTokenizer

class Generator:
    def __init__(self, model_path="backend/checkpoints/model_latest.pt", vocab_path="backend/checkpoints/vocab.json"):
        self.tokenizer = AssemblyTokenizer()
        # Fallback paths if run from different dir
        if not os.path.exists(vocab_path):
            vocab_path = "../checkpoints/vocab.json"
        if not os.path.exists(model_path):
            model_path = "../checkpoints/model_latest.pt"

        if os.path.exists(vocab_path):
            self.tokenizer.load(vocab_path)
            
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = LLMConfig(
            vocab_size=self.tokenizer.get_vocab_size(),
            max_seq_len=256,
            d_model=384,
            n_heads=6,
            n_layers=6,
            device=self.device
        )
        
        self.model = LLM8086(self.config).to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            print(f"Loaded model from {model_path}")
        else:
            print("Warning: No checkpoint found, using untrained model")
        self.model.eval()
        
    def generate(self, prompt, max_new_tokens=50, temperature=1.0, top_k=10):
        tokens = self.tokenizer.encode(prompt)
        idx = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        
        for _ in range(max_new_tokens):
            # crop to max_seq_len
            idx_cond = idx[:, -self.config.max_seq_len:]
            
            with torch.no_grad():
                logits, _ = self.model(idx_cond)
            
            # get logits for the last step
            logits = logits[:, -1, :] / temperature
            
            # top-k sampling
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, next_token), dim=1)
            
            if next_token.item() == self.tokenizer.vocab.get('<|endoftext|>', -1):
                break
                
        return self.tokenizer.decode(idx[0].tolist())
