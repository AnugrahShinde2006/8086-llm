import torch
import torch.optim as optim
import os
import sys
import time

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.transformer import LLM8086
from model.config import LLMConfig
from tokenizer.assembly_tokenizer import AssemblyTokenizer
from data_pipeline.dataset import get_dataloader

def train():
    # Setup tokenizer
    tokenizer = AssemblyTokenizer()
    # Using the new master dataset which includes scraped knowledge!
    data_path = "backend/data/raw/master_dataset.txt"
        
    # Read text to ensure full vocab is built
    with open(data_path, 'r', encoding='utf-8') as f:
        tokenizer.encode(f.read(), update_vocab=True)
    
    # Model config optimized for speed on RTX 3050 6GB
    config = LLMConfig(
        vocab_size=tokenizer.get_vocab_size(),
        max_seq_len=256,
        d_model=384,
        n_heads=6,
        n_layers=6,
        # device="cuda" if torch.cuda.is_available() else "cpu"
        device="cuda"
    )
    
    # Dataloader
    batch_size = 16
    dataloader = get_dataloader(data_path, tokenizer, config.max_seq_len, batch_size=batch_size)
    
    model = LLM8086(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda')
    
    epochs = 2
    
    print(f"Starting training on {config.device} with vocab size {config.vocab_size}")
    
    # Checkpoint dir
    ckpt_dir = "backend/checkpoints"
    if not os.path.exists(ckpt_dir):
        ckpt_dir = "../checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(config.device), y.to(config.device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Automatic Mixed Precision
            with torch.amp.autocast('cuda'):
                logits, loss = model(x, y)
            
            scaler.scale(loss).backward()
            
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            if i % 50 == 0:
                print(f"Epoch {epoch} | Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} complete | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.2f}s")
        
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_latest.pt"))
    tokenizer.save(os.path.join(ckpt_dir, "vocab.json"))
    print("Model and tokenizer saved!")

if __name__ == "__main__":
    train()
