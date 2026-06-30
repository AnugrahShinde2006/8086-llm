from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_version.hf_generator import HFGenerator

app = FastAPI(title="8086 LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize generator globally
generator = HFGenerator()

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    temperature: float = 0.8
    top_k: int = 10

@app.post("/generate")
def generate_text(req: GenerateRequest):
    # The HFGenerator internally formats the prompt and strips the <|assistant|> tags
    response = generator.generate(req.prompt, max_new_tokens=req.max_tokens, temperature=req.temperature, top_k=req.top_k)
    return {"reply": response}

@app.get("/stats")
def get_stats():
    return {
        "model": generator.base_model_id,
        "type": "Hugging Face (QLoRA 4-bit)",
        "device": str(generator.model.device)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
