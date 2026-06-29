import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Add parent directory to path to allow relative imports if run as script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer.assembly_tokenizer import AssemblyTokenizer

class AssemblyDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Tokenize the entire dataset
        self.tokens = self.tokenizer.encode(text, update_vocab=True)
        
    def __len__(self):
        # We need to be able to extract a slice of size max_seq_len + 1
        return len(self.tokens) - self.max_seq_len - 1
        
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.max_seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def get_dataloader(data_path, tokenizer, max_seq_len, batch_size, shuffle=True):
    dataset = AssemblyDataset(data_path, tokenizer, max_seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    tokenizer = AssemblyTokenizer()
    # Assume data generated earlier
    data_path = "../data/raw/synthetic_8086_data.txt"
    if os.path.exists(data_path):
        loader = get_dataloader(data_path, tokenizer, max_seq_len=16, batch_size=2)
        x, y = next(iter(loader))
        print("X shape:", x.shape)
        print("Y shape:", y.shape)
        print("X[0]:", x[0])
        print("Y[0]:", y[0])
