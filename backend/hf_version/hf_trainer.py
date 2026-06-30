import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def prepare_dataset(data_path):
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by our custom EOS token to create individual examples
    examples = [chunk.strip() for chunk in content.split("<|endoftext|>") if len(chunk.strip()) > 10]
    
    # Hugging Face Datasets format
    dataset_dict = {"text": examples}
    return Dataset.from_dict(dataset_dict)

def train_hf_model():
    model_id = "Qwen/Qwen2.5-3B"
    output_dir = "backend/hf_version/adapters"
    data_path = "backend/data/raw/master_dataset.txt"
    
    dataset = prepare_dataset(data_path)
    print(f"Found {len(dataset)} examples.")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. Configure 4-bit Quantization (QLoRA) for 6GB VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    
    print(f"Loading {model_id} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare model for PEFT
    model = prepare_model_for_kbit_training(model)
    
    # 3. Apply LoRA (Low-Rank Adaptation)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # 4. Training Arguments setup for RTX 3050 (6GB) using SFTConfig
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Reduced from 2 to prevent OOM on 7B model
        gradient_accumulation_steps=8, # Increased to maintain effective batch size
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=200, # Quick run for educational purposes
        save_steps=100,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="none", # Disable wandb logging
        dataset_text_field="text"
    )
    
    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )
    
    # 6. Train!
    print("Starting QLoRA Fine-tuning...")
    trainer.train()
    
    # Save the adapter
    print(f"Saving LoRA adapters to {output_dir}...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    train_hf_model()
