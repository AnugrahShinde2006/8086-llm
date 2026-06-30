import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

class HFGenerator:
    def __init__(self):
        self.base_model_id = "Qwen/Qwen2.5-3B"
        self.adapter_dir = "backend/hf_version/adapters"
        
        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir)
        
        # Load base model in 4-bit
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        
        print(f"Loading Base Model ({self.base_model_id})...")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load the LoRA adapter
        print(f"Applying LoRA Adapters from {self.adapter_dir}...")
        self.model = PeftModel.from_pretrained(base_model, self.adapter_dir)
        self.model.eval()
        
    def generate(self, prompt, max_new_tokens=200, temperature=0.7, top_k=50):
        # Format the prompt exactly how it was seen during training
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<|assistant|>" in generated_text:
            response = generated_text.split("<|assistant|>")[-1].strip()
        else:
            response = generated_text
            
        return response

if __name__ == "__main__":
    generator = HFGenerator()
    while True:
        try:
            q = input("\nUser (8086 LLM): ")
            if q.lower() in ['quit', 'exit']: break
            print("\nGenerating...")
            print("Assistant:", generator.generate(q))
        except KeyboardInterrupt:
            break
