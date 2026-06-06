# 🤖 Generative AI with Deploy

> A structured, hands-on exploration of **Large Language Models (LLMs)** using **LangChain** — integrating multiple AI providers with a deployment-ready architecture.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Supported AI Providers](#-supported-ai-providers)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage Guide](#-usage-guide)
- [Script Variants Explained](#-script-variants-explained)
- [Environment Variables](#-environment-variables)
- [Requirements](#-requirements)
- [Roadmap](#-roadmap)
- [Author](#-author)
- [License](#-license)

---

## 🌟 Overview

This project demonstrates how to interact with **multiple leading AI providers** through a unified interface using [LangChain](https://www.langchain.com/). It covers:

- ✅ Connecting to **5 AI providers** (OpenAI, Groq, Mistral, Google Gemini, Hugging Face)
- ✅ Using both the **unified API** (`init_chat_model`) and **provider-specific classes**
- ✅ Configuring model **parameters** such as `temperature` and `max_tokens`
- ✅ Running models **remotely** (via API) and **locally** (on your machine)
- ✅ Clean, reproducible setup using **virtual environments** and **dotenv**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Your Application                     │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  LangChain Framework                    │
│                                                         │
│   ┌─────────────┐         ┌───────────────────────┐    │
│   │init_chat_   │         │ Provider-Specific      │    │
│   │model() API  │   OR    │ Chat Classes           │    │
│   │(Unified)    │         │ (ChatOpenAI, ChatGroq  │    │
│   └─────────────┘         │  ChatMistralAI, etc.)  │    │
│                           └───────────────────────┘    │
└────────────────┬──────────────────────────────┬────────┘
                 │                              │
     ┌───────────▼──────────────────────────────▼──────────┐
     │               AI Provider Layer                      │
     │                                                      │
     │  ┌─────────┐ ┌──────┐ ┌─────────┐ ┌──────────────┐ │
     │  │ OpenAI  │ │ Groq │ │ Mistral │ │Google Gemini │ │
     │  │ GPT-5   │ │Llama │ │Mistral- │ │Gemini-2.5-   │ │
     │  │ GPT-4.1 │ │  -4  │ │Small   │ │Flash-Lite    │ │
     │  └─────────┘ └──────┘ └─────────┘ └──────────────┘ │
     │                                                      │
     │  ┌──────────────────────────────────────────────┐   │
     │  │              Hugging Face                     │   │
     │  │  ┌─────────────────┐  ┌────────────────────┐ │   │
     │  │  │ Remote Endpoint │  │  Local Pipeline    │ │   │
     │  │  │ (DeepSeek-R1)   │  │  (TinyLlama)       │ │   │
     │  │  └─────────────────┘  └────────────────────┘ │   │
     │  └──────────────────────────────────────────────┘   │
     └──────────────────────────────────────────────────────┘
```

---

## 🤝 Supported AI Providers

| Provider | Model(s) Used | Access Type | LangChain Package |
|---|---|---|---|
| **OpenAI** | GPT-5, GPT-4.1 | Remote API | `langchain-openai` |
| **Groq** | Llama 4 Scout 17B | Remote API | `langchain-groq` |
| **Mistral AI** | Mistral Small 2506 | Remote API | `langchain-mistralai` |
| **Google Gemini** | Gemini 2.5 Flash Lite | Remote API | `langchain-google-genai` |
| **Hugging Face** | DeepSeek-R1, Phi-3-Mini, TinyLlama | Remote & Local | `langchain-huggingface` |

---

## 📁 Project Structure

```
generative-ai-with-deploy/
│
├── 📄 .env                          # API keys (never commit this)
├── 📄 .env.example                  # Template for environment setup
├── 📄 .gitignore                    # Git exclusions
├── 📄 requirements.txt              # All dependencies
├── 📄 pyproject.toml                # Package configuration
├── 📄 README.md                     # This file
│
└── 📂 chat-models/                  # LLM provider demos
    │
    ├── 📂 open-ai/
    │   ├── chat.py                              # Unified init_chat_model API
    │   ├── chat-with-class-usage.py             # ChatOpenAI class
    │   └── chat-with-class-usage-with-options.py  # With temperature config
    │
    ├── 📂 groq/
    │   ├── chat.py                              # Unified init_chat_model API
    │   ├── chat-with-class-usage.py             # ChatGroq class
    │   └── chat-with-class-usage-with-options.py  # With temperature config
    │
    ├── 📂 mistral-ai/
    │   ├── chat.py                              # Unified init_chat_model API
    │   ├── chat-with-class-usage.py             # ChatMistralAI class
    │   └── chat-with-class-usage-with-options.py  # With temperature & max_tokens
    │
    ├── 📂 google-gemini/
    │   ├── chat.py                              # Unified init_chat_model API
    │   ├── chat-with-class-usage.py             # ChatGoogleGenerativeAI class
    │   └── chat-with-class-usage-with-options.py  # With temperature config
    │
    └── 📂 hugging-face/
        ├── chat-with-class-usage.py             # Remote endpoint (DeepSeek-R1)
        ├── chat-with-class-usage-with-options.py  # Remote with temperature
        ├── chat-download-local.py               # Unified API (Phi-3-Mini)
        └── local-model.py                       # Fully local (TinyLlama pipeline)
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.8+**
- A virtual environment tool (`venv` or `uv`)
- API keys for the providers you wish to use

### Step 1 — Clone the Repository

```bash
git clone https://github.com/menkar/generative-ai-with-deploy.git
cd generative-ai-with-deploy
```

### Step 2 — Create a Virtual Environment

```bash
# Using Python venv
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (macOS / Linux)
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

```bash
# Copy the example file
cp .env.example .env
```

Open `.env` and fill in your API keys (see [Environment Variables](#-environment-variables) below).

### Step 5 — Run a Script

```bash
# Example: Run OpenAI chat
python chat-models/open-ai/chat.py

# Example: Run Groq with options
python chat-models/groq/chat-with-class-usage-with-options.py

# Example: Run Hugging Face locally (no API key needed)
python chat-models/hugging-face/local-model.py
```

---

## 📖 Usage Guide

### Approach 1 — Unified API (`init_chat_model`)

The simplest way to switch between providers without changing your code:

```python
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# Just change the model string to switch providers
model = init_chat_model("gpt-4.1", model_provider="openai")
# model = init_chat_model("groq:meta-llama/llama-4-scout-17b-16e-instruct")
# model = init_chat_model("google_genai:gemini-2.5-flash-lite")

response = model.invoke("What is generative AI?")
print(response.content)
```

### Approach 2 — Provider-Specific Classes

More control and access to provider-specific features:

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model="gpt-5",
    temperature=0,       # 0 = deterministic, 1 = creative
    max_tokens=1024
)

response = model.invoke("Explain machine learning in simple terms.")
print(response.content)
```

### Approach 3 — Local Model (No API Key Required)

Run a model entirely on your machine:

```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

pipeline = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 512,
        "do_sample": False,
    }
)

model = ChatHuggingFace(llm=pipeline)
response = model.invoke("What is generative AI?")
print(response.content)
```

---

## 🔧 Script Variants Explained

Each provider folder contains three types of scripts:

| Script | Purpose | Best For |
|---|---|---|
| `chat.py` | Uses `init_chat_model()` — provider-agnostic | Quick switching between providers |
| `chat-with-class-usage.py` | Uses provider-specific LangChain class | Understanding provider integration |
| `chat-with-class-usage-with-options.py` | Adds `temperature`, `max_tokens`, etc. | Fine-tuning model behaviour |

**Hugging Face** additionally provides:

| Script | Purpose |
|---|---|
| `chat-download-local.py` | Downloads and runs a HF model via unified API |
| `local-model.py` | Fully offline inference using `HuggingFacePipeline` |

---

## 🔑 Environment Variables

Create a `.env` file in the project root. Use the table below as a reference:

```
# .env.example — copy this to .env and fill in your keys

OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

| Variable | Provider | Where to Get It |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com](https://platform.openai.com/api-keys) |
| `GROQ_API_KEY` | Groq | [console.groq.com](https://console.groq.com/keys) |
| `GOOGLE_API_KEY` | Google Gemini | [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `MISTRAL_API_KEY` | Mistral AI | [console.mistral.ai](https://console.mistral.ai/) |
| `HUGGINGFACEHUB_API_TOKEN` | Hugging Face | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

> ⚠️ **Important:** Never commit your `.env` file to version control. It is already listed in `.gitignore`.

---

## 📦 Requirements

```
langchain                # Core LangChain framework
langchain-core           # LangChain base abstractions
langchain-community      # Community integrations
langgraph                # Graph-based LLM workflows
langchain-openai         # OpenAI integration
langchain-google-genai   # Google Gemini integration
langchain-groq           # Groq integration
langchain-mistralai      # Mistral AI integration
langchain-huggingface    # Hugging Face integration
huggingface-hub          # HF model hub access
python-dotenv            # .env file loader
faiss-cpu                # Vector similarity search (for RAG)
tiktoken                 # OpenAI token counter
fastapi                  # REST API framework (deployment)
uvicorn                  # ASGI server (deployment)
requests                 # HTTP client
```

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🗺️ Roadmap

- [x] OpenAI GPT integration
- [x] Groq / Llama integration
- [x] Mistral AI integration
- [x] Google Gemini integration
- [x] Hugging Face remote endpoint integration
- [x] Hugging Face local model (offline inference)
- [x] Model parameter configuration (temperature, max_tokens)
- [ ] Prompt templates and chat history
- [ ] Retrieval-Augmented Generation (RAG) with FAISS
- [ ] FastAPI REST API deployment
- [ ] Dockerized deployment
- [ ] Streaming responses
- [ ] LangGraph multi-agent workflows

---

## 👤 Author

**Swapnil Menkar**

- 📱 Mobile: [+91 8149005578](tel:+918149005578)
- 💼 LinkedIn: [linkedin.com/in/swapnil-menkar-7051852b](https://www.linkedin.com/in/swapnil-menkar-7051852b/)

---

## 📄 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute this project with proper attribution.

---

<p align="center">
  Built with ❤️ using <a href="https://www.langchain.com/">LangChain</a> &nbsp;|&nbsp;
  Powered by OpenAI · Groq · Mistral · Google · Hugging Face
</p>
