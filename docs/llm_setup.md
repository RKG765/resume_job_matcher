# LM Studio Setup Guide for Resume-Job Matcher

This guide explains how to set up **LM Studio** to generate AI-powered match explanations in the Resume-Job Matcher application.

## What is LM Studio?

LM Studio is a free desktop application that allows you to run Large Language Models (LLMs) locally on your computer. It provides an OpenAI-compatible API, making it easy to integrate with applications.

## Prerequisites

- **Windows/Mac/Linux** computer
- **8GB+ RAM** (16GB recommended for larger models)
- **GPU with 6GB+ VRAM** (optional but recommended for faster inference)

---

## Step 1: Download and Install LM Studio

1. Visit [https://lmstudio.ai](https://lmstudio.ai)
2. Download the installer for your operating system
3. Install and launch LM Studio

---

## Step 2: Download a Model

1. In LM Studio, click on **"Search"** in the left sidebar
2. Search for a model (recommended options):
   - **Llama 3.2 3B** (fast, good quality)
   - **Mistral 7B Instruct** (excellent for analysis)
   - **Phi-3 Mini** (lightweight, fast)
   - **Gemma 2B** (very lightweight)

3. Click **Download** on your chosen model
4. Wait for the download to complete

### Recommended Models by Hardware:

| RAM   | VRAM  | Recommended Model         |
|-------|-------|---------------------------|
| 8GB   | 4GB   | Phi-3 Mini, Gemma 2B      |
| 16GB  | 6GB   | Llama 3.2 3B, Qwen 2.5    |
| 32GB  | 8GB+  | Mistral 7B, Llama 3.1 8B  |

---

## Step 3: Start the Local Server

1. Click on **"Local Server"** in the left sidebar (or press `Ctrl+L`)
2. Select the model you downloaded from the dropdown
3. Click **"Start Server"**
4. The server will start on `http://localhost:1234`

You should see:
```
Server started successfully
Listening on http://localhost:1234
```

---

## Step 4: Configure the Resume-Job Matcher

The application is pre-configured to connect to LM Studio at `http://localhost:1234`. No additional configuration is needed if you're using the default settings.

### Default Configuration:
```json
{
  "api_url": "http://localhost:1234/v1/chat/completions",
  "api_key": "lm-studio",
  "model": "local-model"
}
```

### Custom Configuration (Optional):

You can configure the LLM service via the API:

```bash
curl -X POST http://localhost:5000/api/llm/configure \
  -H "Content-Type: application/json" \
  -d '{
    "api_url": "http://localhost:1234/v1/chat/completions",
    "api_key": "lm-studio",
    "model": "your-model-name",
    "enabled": true
  }'
```

### Check LLM Status:

```bash
curl http://localhost:5000/api/llm/status
```

Response:
```json
{
  "enabled": true,
  "is_available": true,
  "api_url": "http://localhost:1234/v1/chat/completions",
  "model": "local-model"
}
```

---

## Step 5: Test the Integration

1. Ensure LM Studio server is running
2. Start the Resume-Job Matcher backend: `python main.py`
3. Open the frontend at `http://localhost:5173`
4. Run a manual match - you should see AI-generated explanations!

---

## Troubleshooting

### LLM Not Available

If `is_available` returns `false`:

1. **Check LM Studio is running** - Ensure the local server is started
2. **Check the port** - Default is 1234, but LM Studio may use a different port
3. **Firewall** - Ensure localhost:1234 is not blocked

### Slow Responses

- Use a smaller model (Phi-3 Mini, Gemma 2B)
- Enable GPU acceleration in LM Studio settings
- Reduce `max_tokens` in the API call

### Fallback Mode

If the LLM is unavailable, the application automatically uses a **rule-based fallback** that generates explanations based on the match scores and skills. No LLM is required for basic functionality.

---

## Alternative LLM Options

### Option 1: Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.2

# Run with OpenAI-compatible API
ollama serve
```

Configure the app:
```json
{
  "api_url": "http://localhost:11434/v1/chat/completions",
  "model": "llama3.2"
}
```

### Option 2: OpenAI API

```json
{
  "api_url": "https://api.openai.com/v1/chat/completions",
  "api_key": "sk-your-openai-key",
  "model": "gpt-4o-mini"
}
```

### Option 3: Local vLLM

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.2-3B
```

---

## API Reference

### POST /api/llm/configure

Configure LLM service settings.

**Request Body:**
```json
{
  "api_url": "http://localhost:1234/v1/chat/completions",
  "api_key": "lm-studio",
  "model": "local-model",
  "enabled": true
}
```

### GET /api/llm/status

Get current LLM service status.

**Response:**
```json
{
  "enabled": true,
  "is_available": true,
  "api_url": "http://localhost:1234/v1/chat/completions",
  "model": "local-model"
}
```

---

## Summary

| Step | Action |
|------|--------|
| 1    | Download LM Studio from lmstudio.ai |
| 2    | Download a model (e.g., Llama 3.2 3B) |
| 3    | Start the local server (Ctrl+L → Start Server) |
| 4    | Run the Resume-Job Matcher backend |
| 5    | Match resumes and see AI explanations! |

The application works without an LLM (uses rule-based explanations), but LM Studio adds intelligent, context-aware analysis of why each candidate matches or doesn't match the job requirements.
