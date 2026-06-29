from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.generator import Generator

app = FastAPI(title="8086 LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize generator globally
generator = Generator(model_path="backend/checkpoints/model_latest.pt", vocab_path="backend/checkpoints/vocab.json")

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 10

@app.post("/generate")
def generate_text(req: GenerateRequest):
    # Formulate the prompt with special tokens
    full_prompt = f"<|user|>\n{req.prompt}\n<|assistant|>\n"
    response = generator.generate(full_prompt, req.max_tokens, req.temperature, req.top_k)
    
    # Extract only the assistant part
    try:
        assistant_reply = response.split("<|assistant|>\n")[1].split("<|endoftext|>")[0].strip()
    except Exception as e:
        assistant_reply = response
        
    return {"reply": assistant_reply}

@app.get("/stats")
def get_stats():
    return {
        "vocab_size": generator.tokenizer.get_vocab_size(),
        "d_model": generator.config.d_model,
        "n_layers": generator.config.n_layers,
        "n_heads": generator.config.n_heads,
        "device": generator.config.device
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
