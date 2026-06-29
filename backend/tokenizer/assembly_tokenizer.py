import re
import json

class AssemblyTokenizer:
    def __init__(self):
        # 8086 specific keywords
        self.mnemonics = [
            'MOV', 'ADD', 'SUB', 'MUL', 'DIV', 'INC', 'DEC', 'CMP',
            'AND', 'OR', 'XOR', 'NOT', 'TEST', 'SHL', 'SHR', 'SAL', 'SAR', 'ROL', 'ROR', 'RCL', 'RCR',
            'JMP', 'CALL', 'RET', 'JE', 'JNE', 'JG', 'JGE', 'JL', 'JLE', 'JZ', 'JNZ', 'JC', 'JNC', 'JO', 'JNO', 'JS', 'JNS',
            'LOOP', 'LOOPE', 'LOOPNE', 'LOOPZ', 'LOOPNZ',
            'PUSH', 'POP', 'PUSHF', 'POPF',
            'XCHG', 'LEA', 'LDS', 'LES', 'LAHF', 'SAHF', 'XLAT',
            'IN', 'OUT',
            'MOVSB', 'MOVSW', 'CMPSB', 'CMPSW', 'SCASB', 'SCASW', 'LODSB', 'LODSW', 'STOSB', 'STOSW',
            'REP', 'REPE', 'REPZ', 'REPNE', 'REPNZ',
            'STC', 'CLC', 'CMC', 'STD', 'CLD', 'STI', 'CLI',
            'HLT', 'WAIT', 'ESC', 'LOCK', 'NOP', 'INT', 'INTO', 'IRET'
        ]
        self.registers = [
            'AX', 'BX', 'CX', 'DX',
            'AH', 'AL', 'BH', 'BL', 'CH', 'CL', 'DH', 'DL',
            'CS', 'DS', 'SS', 'ES',
            'SP', 'BP', 'SI', 'DI', 'IP', 'FLAGS'
        ]
        self.special_tokens = [
            '<|pad|>', '<|user|>', '<|assistant|>', '<|assembly|>', '<|endoftext|>'
        ]
        
        # Build initial vocabulary
        self.vocab = {}
        self.inverse_vocab = {}
        
        # Add special tokens
        for token in self.special_tokens:
            self._add_token(token)
            
        # Add basic ASCII characters for fallback
        for i in range(32, 127):
            self._add_token(chr(i))
        self._add_token('\n')
        self._add_token('\t')
            
        # Add mnemonics and registers
        for m in self.mnemonics:
            self._add_token(m)
        for r in self.registers:
            self._add_token(r)
            
        # Create regex pattern for tokenization
        special_pattern = "|".join([re.escape(t) for t in self.special_tokens])
        hex_pattern = r'[0-9A-Fa-f]+H'
        word_pattern = r'\b[A-Za-z_][A-Za-z0-9_]*\b'
        num_pattern = r'\d+'
        
        self.token_pattern = re.compile(f'({special_pattern}|{hex_pattern}|{word_pattern}|{num_pattern}|\\S|\\s)')
        self.pad_token_id = self.vocab['<|pad|>']
        
    def _add_token(self, token):
        if token not in self.vocab:
            idx = len(self.vocab)
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token
            
    def encode(self, text, update_vocab=False):
        tokens = []
        matches = self.token_pattern.findall(text)
        for match in matches:
            if not match:
                continue
            if match in self.vocab:
                tokens.append(self.vocab[match])
            else:
                if update_vocab:
                    self._add_token(match)
                    tokens.append(self.vocab[match])
                else:
                    # Fallback to character level encoding for OOV words
                    for char in match:
                        if char in self.vocab:
                            tokens.append(self.vocab[char])
        return tokens
        
    def decode(self, token_ids):
        return "".join([self.inverse_vocab.get(tid, "") for tid in token_ids])

    def get_vocab_size(self):
        return len(self.vocab)
        
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f)
            
    def load(self, filepath):
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
            self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
            self.pad_token_id = self.vocab.get('<|pad|>', 0)

if __name__ == "__main__":
    tokenizer = AssemblyTokenizer()
    text = "<|assembly|>\nMOV AX, 1234H\nADD AX, BX\nHLT\n<|endoftext|>"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    print("Original:", repr(text))
    print("Encoded:", encoded)
    print("Decoded:", repr(decoded))
    print("Vocab size:", tokenizer.get_vocab_size())
