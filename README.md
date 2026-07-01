# 8086 Domain-Specific Large Language Model

An end-to-end, full-stack AI project that trains and serves a highly specialized Large Language Model (LLM) acting as a domain expert on the Intel 8086 microprocessor architecture and assembly programming.

This project was built to demonstrate both deep theoretical understanding (training a custom transformer from scratch) and production-grade MLOps engineering (QLoRA fine-tuning and GGUF deployment).

## 🚀 Features

- **Custom From-Scratch Architecture:** Includes a ~30M parameter decoder-only transformer built entirely from scratch in PyTorch with a custom character-fallback Assembly Tokenizer.
- **State-of-the-Art Fine-Tuning:** Features a robust QLoRA pipeline to fine-tune massive 7-Billion parameter models (like `Qwen2.5-7B`) on consumer hardware.
- **Dynamic LoRA GGUF Deployment:** Bypasses PyTorch memory constraints by converting adapters to GGUF format and dynamically hot-swapping them into a native Ollama instance at runtime.
- **FastAPI Backend:** A lightweight, lightning-fast bridge connecting the AI inference engines to the web.
- **Modern React Frontend:** A sleek, interactive chat interface with real-time 8086 Assembly syntax highlighting and dynamic model statistics.

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:
- [Python 3.10+](https://www.python.org/)
- [Node.js & npm](https://nodejs.org/)
- [Ollama](https://ollama.com/) (Must be running in the background)
- [Git](https://git-scm.com/)

## 🏃‍♂️ Getting Started

### 1. Setup the React Frontend
```powershell
cd frontend
npm install
```

### 2. Setup the Python Backend
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install fastapi uvicorn pydantic # Add any other standard requirements
```

### 3. Build the Ollama AI
To run the massive 7 Billion parameter model, we first need to convert the fine-tuned LoRA adapters to `.gguf` format and register them with Ollama.

Run the automated conversion script:
```powershell
powershell -ExecutionPolicy Bypass -File backend\ollama\convert_adapter.ps1
```

Once the conversion is complete, create the model in Ollama:
```powershell
ollama create 8086-llm -f backend\ollama\Modelfile
```

### 4. Start the Application
Start the FastAPI server:
```powershell
.\venv\Scripts\activate
python backend\api\main.py
```

In a new terminal, start the React frontend:
```powershell
cd frontend
npm run dev
```

Navigate to `http://localhost:5173` in your browser and start writing 8086 assembly!
