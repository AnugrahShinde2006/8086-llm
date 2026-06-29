import json
import os
import random

def generate_qa_pairs():
    pairs = [
        {"prompt": "What is the 8086 microprocessor?", "completion": "The 8086 is a 16-bit microprocessor introduced by Intel in 1978. It has a 16-bit data bus and a 20-bit address bus, allowing it to address up to 1 MB of memory."},
        {"prompt": "What are the general-purpose registers in 8086?", "completion": "The 8086 has four 16-bit general-purpose registers: AX (Accumulator), BX (Base), CX (Count), and DX (Data). These can also be accessed as eight 8-bit registers: AH, AL, BH, BL, CH, CL, DH, and DL."},
        {"prompt": "Explain the role of the segment registers.", "completion": "The 8086 has four 16-bit segment registers: CS (Code Segment), DS (Data Segment), SS (Stack Segment), and ES (Extra Segment). They are used in combination with an offset to calculate the 20-bit physical address."},
        {"prompt": "How is the 20-bit physical address calculated?", "completion": "The physical address is calculated by shifting the 16-bit segment register value left by 4 bits (or multiplying by 16) and adding the 16-bit offset. Physical Address = (Segment Register * 10H) + Offset."},
        {"prompt": "What does the MOV instruction do?", "completion": "The MOV instruction transfers data from a source operand to a destination operand. The source can be a register, memory location, or immediate data, and the destination can be a register or memory location. Both operands must be the same size."},
        {"prompt": "Write an 8086 assembly program to add two 16-bit numbers.", "completion": "MOV AX, 1234H\nMOV BX, 5678H\nADD AX, BX\nHLT"},
        {"prompt": "Explain the flags register in 8086.", "completion": "The 8086 flags register is a 16-bit register containing 9 active flags: 6 status flags (Carry, Parity, Auxiliary Carry, Zero, Sign, Overflow) and 3 control flags (Trap, Interrupt Enable, Direction)."},
        {"prompt": "What is the difference between JMP and CALL instructions?", "completion": "JMP transfers execution to a specified address unconditionally, without saving the return address. CALL transfers execution to a procedure and pushes the return address (IP, and CS if far call) onto the stack so execution can return later using the RET instruction."},
    ]
    return pairs

def generate_assembly_examples():
    examples = [
        "; Program to move data\nMOV AX, 0005H\nMOV BX, AX\nHLT",
        "; Program to subtract numbers\nMOV AX, 0010H\nMOV BX, 0005H\nSUB AX, BX\nHLT",
        "; Program to multiply two 8-bit numbers\nMOV AL, 05H\nMOV BL, 04H\nMUL BL\nHLT",
        "; Program using logical AND\nMOV AX, FFFFH\nAND AX, 000FH\nHLT",
        "; Program to swap two registers\nMOV AX, 1111H\nMOV BX, 2222H\nXCHG AX, BX\nHLT",
        "; Program to increment a register\nMOV CX, 0000H\nINC CX\nINC CX\nHLT",
        "; Loop example\nMOV CX, 0005H\nMOV AX, 0000H\nSTART: ADD AX, 0001H\nLOOP START\nHLT"
    ]
    return examples

def generate_large_dataset(output_path, num_samples=10000):
    qa_pairs = generate_qa_pairs()
    assembly_examples = generate_assembly_examples()
    
    data = []
    
    # Add a lot of permutations
    for i in range(num_samples):
        if random.random() < 0.6:
            # QA pair
            pair = random.choice(qa_pairs)
            data.append(f"<|user|>\n{pair['prompt']}\n<|assistant|>\n{pair['completion']}\n<|endoftext|>")
        else:
            # Assembly example
            example = random.choice(assembly_examples)
            # randomly change values to create variance
            hex_val1 = f"{random.randint(0, 65535):04X}H"
            hex_val2 = f"{random.randint(0, 65535):04X}H"
            example = example.replace("1234H", hex_val1).replace("5678H", hex_val2).replace("0005H", hex_val1)
            data.append(f"<|assembly|>\n{example}\n<|endoftext|>")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(data))
    
    print(f"Generated {num_samples} samples at {output_path}")

if __name__ == "__main__":
    generate_large_dataset("../data/raw/synthetic_8086_data.txt")
