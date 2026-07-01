from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import urllib.request
import json

app = FastAPI(title="8086 LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# No need to initialize a global generator! Ollama handles it.

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 1024  # Increased from 50 to allow full, detailed answers
    temperature: float = 0.7
    top_k: int = 50

@app.post("/generate")
def generate_text(req: GenerateRequest):
    url = "http://localhost:11434/api/generate"
    data = json.dumps({
        "model": "8086-llm",
        "prompt": req.prompt,
        "stream": False,
        "options": {
            "temperature": req.temperature,
            "top_k": req.top_k,
            "num_predict": req.max_tokens
        }
    }).encode("utf-8")
    
    req_obj = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    
    try:
        with urllib.request.urlopen(req_obj) as response:
            result = json.loads(response.read().decode())
            return {"reply": result.get("response", "")}
    except Exception as e:
        return {"reply": f"Error communicating with Ollama: {str(e)}\nMake sure Ollama is running and you ran 'ollama create 8086-llm -f backend/ollama/Modelfile'!"}

@app.get("/stats")
def get_stats():
    return {
        "model": "qwen2.5:7b + 8086 LoRA",
        "type": "Ollama (GGUF 4-bit)",
        "device": "Auto-Offloaded (CPU/GPU)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
