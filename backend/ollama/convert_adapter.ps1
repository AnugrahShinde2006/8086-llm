Write-Host "Cloning llama.cpp to backend/ollama/llama.cpp..." -ForegroundColor Cyan
if (!(Test-Path "backend/ollama/llama.cpp")) {
    git clone https://github.com/ggerganov/llama.cpp.git backend/ollama/llama.cpp
} else {
    Write-Host "llama.cpp already cloned." -ForegroundColor Yellow
}

Write-Host "Installing requirements in virtual environment..." -ForegroundColor Cyan
& .\venv\Scripts\pip install -r backend/ollama/llama.cpp/requirements.txt

Write-Host "Converting LoRA adapters to GGUF format (This takes zero RAM!)..." -ForegroundColor Cyan
& .\venv\Scripts\python backend/ollama/llama.cpp/convert_lora_to_gguf.py backend/hf_version/adapters/ --outfile backend/ollama/8086_adapter.gguf

Write-Host "Conversion Complete!" -ForegroundColor Green
Write-Host "You can now run: ollama create 8086-llm -f backend/ollama/Modelfile" -ForegroundColor Yellow
