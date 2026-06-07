# 🤖 Generative AI with Deploy

> A comprehensive, hands-on exploration of **Large Language Models (LLMs)** using **LangChain** — covering multi-provider AI integration, embeddings, prompt engineering, schema-driven extraction, and production-ready Streamlit UIs.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Module Breakdown](#-module-breakdown)
  - [Chat Models](#1️⃣-chat-models)
  - [Chatbot Applications](#2️⃣-chatbot-applications)
  - [Embedding Models](#3️⃣-embedding-models)
  - [Prompt Templates](#4️⃣-prompt-templates)
  - [SwapAnalytics Project](#5️⃣-swapanalytics-project)
- [Supported AI Providers](#-supported-ai-providers)
- [Script Variants Explained](#-script-variants-explained)
- [Streamlit UIs](#-streamlit-uis)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Requirements](#-requirements)
- [Roadmap](#-roadmap)
- [Author](#-author)
- [License](#-license)

---

## 🌟 Overview

This project is a structured, progressive learning repository that demonstrates:

- ✅ Connecting to **5 AI providers** via unified and provider-specific LangChain APIs
- ✅ Building **CLI chatbots** with personality modes and conversation history
- ✅ Creating **Streamlit web UIs** for chatbots and information extractors
- ✅ Generating **text embeddings** locally and via API
- ✅ Engineering **prompt templates** with structured output rules
- ✅ Using **Pydantic schemas** for validated, type-safe AI output
- ✅ **Schema-driven JSON extraction** with copyable output
- ✅ Running models both **remotely** (API) and **locally** (on-device)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        User Interfaces                              │
│                                                                     │
│   ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│   │  Streamlit Chat  │  │ Mode-Select Chat │  │ Movie Extractor │  │
│   │   (chatbot-ui)   │  │ (manual-input-ui)│  │  (SwapAnalytics)│  │
│   └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘  │
└────────────┼────────────────────┼────────────────────┼────────────┘
             │                    │                    │
             ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      LangChain Framework                            │
│                                                                     │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────────────────┐  │
│  │init_chat_    │  │ Provider-Specific│  │   Prompt Templates    │  │
│  │model() API   │  │ Chat Classes     │  │   + Output Parsers    │  │
│  │  (Unified)   │  │ (ChatOpenAI etc) │  │   (Pydantic Schema)   │  │
│  └──────────────┘  └─────────────────┘  └───────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┼─────────────────────┐
        ▼                    ▼                      ▼
┌──────────────┐   ┌──────────────────┐   ┌───────────────────────┐
│  Remote LLMs  │   │   Embeddings     │   │    Local Models       │
│               │   │                  │   │                       │
│ ┌──────────┐  │   │ ┌─────────────┐  │   │ ┌──────────────────┐  │
│ │  OpenAI  │  │   │ │OpenAI Text  │  │   │ │ TinyLlama 1.1B   │  │
│ │  GPT-5   │  │   │ │Embedding    │  │   │ │ (HF Pipeline)    │  │
│ │  GPT-4.1 │  │   │ │3-large      │  │   └──────────────────┘  │
│ └──────────┘  │   │ └─────────────┘  │   │ ┌──────────────────┐  │
│ ┌──────────┐  │   │ ┌─────────────┐  │   │ │ Phi-3-Mini 4K    │  │
│ │  Groq    │  │   │ │HuggingFace  │  │   │ │ (init_chat_model)│  │
│ │  Llama 4 │  │   │ │all-MiniLM   │  │   │ └──────────────────┘  │
│ └──────────┘  │   │ └─────────────┘  │   └───────────────────────┘
│ ┌──────────┐  │   └──────────────────┘
│ │  Mistral │  │
│ │  Small   │  │
│ └──────────┘  │
│ ┌──────────┐  │
│ │  Gemini  │  │
│ │  2.5 Lite│  │
│ └──────────┘  │
│ ┌──────────┐  │
│ │HuggingFace│ │
│ │DeepSeek-R1│ │
│ └──────────┘  │
└───────────────┘
```

---

## 📁 Project Structure

```
generative-ai-with-deploy/
│
├── 📄 .env.example                         # Safe API key template
├── 📄 .gitignore                           # Git exclusions
├── 📄 requirements.txt                     # All Python dependencies
├── 📄 pyproject.toml                       # Package configuration
├── 📄 README.md                            # This file
│
├── 📂 chat-models/                         # LLM provider integration demos
│   ├── 📂 open-ai/
│   │   ├── chat.py                         # Unified API (GPT-4.1)
│   │   ├── chat-with-class-usage.py        # ChatOpenAI class (GPT-5)
│   │   └── chat-with-class-usage-with-options.py  # With temperature=0
│   │
│   ├── 📂 groq/
│   │   ├── chat.py                         # Unified API (Llama 4 Scout)
│   │   ├── chat-with-class-usage.py        # ChatGroq class
│   │   └── chat-with-class-usage-with-options.py  # With temperature=0
│   │
│   ├── 📂 mistral-ai/
│   │   ├── chat.py                         # Unified API (Mistral Small)
│   │   ├── chat-with-class-usage.py        # ChatMistralAI class
│   │   └── chat-with-class-usage-with-options.py  # temp=0.7, max_tokens=20
│   │
│   ├── 📂 google-gemini/
│   │   ├── chat.py                         # Unified API (Gemini 2.5 Flash Lite)
│   │   ├── chat-with-class-usage.py        # ChatGoogleGenerativeAI class
│   │   └── chat-with-class-usage-with-options.py  # With temperature=0
│   │
│   └── 📂 hugging-face/
│       ├── chat-with-class-usage.py        # Remote endpoint (DeepSeek-R1)
│       ├── chat-with-class-usage-with-options.py  # Remote + temperature=0.7
│       ├── chat-download-local.py          # Unified API (Phi-3-Mini, downloads)
│       └── local-model.py                  # Fully local (TinyLlama pipeline)
│
├── 📂 chatbot/                             # Chatbot applications
│   ├── chatbot-simple-prompt.py            # Minimal CLI chatbot
│   ├── chatbot-system-user-prompt.py       # CLI with system prompt (funny)
│   ├── chatbot-system-user-prompt-manual-input.py  # CLI with mode selection
│   ├── chatbot-ui.py                       # ★ Streamlit UI (funny mode)
│   └── chatbot-manual-input-ui.py          # ★ Streamlit UI (3 personality modes)
│
├── 📂 embedding-models/                    # Text embedding demos
│   ├── 📂 open-ai/
│   │   ├── embeddings-documents.py         # Embed multiple documents
│   │   └── embeddings-query.py             # Embed a single query
│   └── 📂 hugging-face/
│       └── embeddings-download-local.py    # Local sentence-transformer embeddings
│
├── 📂 prompt-templates/                    # Prompt engineering
│   └── movie-info-extractor.py             # Movie info extraction (GPT-4.1)
│
└── 📂 project/
    └── 📂 SwapAnalytics/                   # ★ Full-stack AI application
        ├── 📂 prompt-templates/
        │   ├── core.py                     # CLI movie extractor (Mistral)
        │   └── app.py                      # ★ Streamlit movie extractor UI
        └── 📂 prompt-templates-with-schema/
            ├── core.py                     # CLI schema extractor (Pydantic)
            └── app.py                      # ★ Streamlit schema extractor UI
```

---

## 📖 Module Breakdown

### 1️⃣ Chat Models

Each of the 5 provider folders demonstrates **how to connect to an LLM** using three progressively detailed approaches.

#### How Each Script Works

```
load_dotenv()  →  Create Model  →  model.invoke("question")  →  print(response.content)
```

#### OpenAI  (`chat-models/open-ai/`)

| File | Model | Parameters | Purpose |
|---|---|---|---|
| `chat.py` | GPT-4.1 | defaults | Unified `init_chat_model()` API |
| `chat-with-class-usage.py` | GPT-5 | defaults | Direct `ChatOpenAI` class |
| `chat-with-class-usage-with-options.py` | GPT-5 | `temperature=0` | Deterministic output |

#### Groq  (`chat-models/groq/`)

| File | Model | Parameters | Purpose |
|---|---|---|---|
| `chat.py` | Llama 4 Scout 17B | defaults | Unified API with Groq prefix |
| `chat-with-class-usage.py` | Llama 4 Scout 17B | defaults | Direct `ChatGroq` class |
| `chat-with-class-usage-with-options.py` | Llama 4 Scout 17B | `temperature=0` | Deterministic output |

#### Mistral AI  (`chat-models/mistral-ai/`)

| File | Model | Parameters | Purpose |
|---|---|---|---|
| `chat.py` | mistral-small-2506 | defaults | Unified API |
| `chat-with-class-usage.py` | mistral-small-2506 | defaults | Direct `ChatMistralAI` class |
| `chat-with-class-usage-with-options.py` | mistral-small-2506 | `temperature=0.7`, `max_tokens=20` | Short creative answer about ML |

#### Google Gemini  (`chat-models/google-gemini/`)

| File | Model | Parameters | Purpose |
|---|---|---|---|
| `chat.py` | gemini-2.5-flash-lite | defaults | Unified API |
| `chat-with-class-usage.py` | gemini-2.5-flash-lite | defaults | Direct `ChatGoogleGenerativeAI` class |
| `chat-with-class-usage-with-options.py` | gemini-2.5-flash-lite | `temperature=0` | Deterministic output |

#### Hugging Face  (`chat-models/hugging-face/`)

| File | Model | Access | Parameters | Purpose |
|---|---|---|---|---|
| `chat-with-class-usage.py` | DeepSeek-R1 | Remote API | defaults | `HuggingFaceEndpoint` + `ChatHuggingFace` |
| `chat-with-class-usage-with-options.py` | DeepSeek-R1 | Remote API | `temperature=0.7` | Remote endpoint with options |
| `chat-download-local.py` | Phi-3-Mini-4K | Downloads | `temperature=0.7`, `max_tokens=1024` | Unified API, downloads on first run |
| `local-model.py` | TinyLlama 1.1B | **Fully Local** | `max_new_tokens=512`, greedy | `HuggingFacePipeline` — no API key needed |

---

### 2️⃣ Chatbot Applications

Progressive chatbot implementations — from a minimal CLI to a full Streamlit UI.

#### Flow Diagram

```
Simple CLI          →   System Prompt CLI   →   Mode-Select CLI
      ↓                                               ↓
Streamlit UI (funny)  ←——————————————  Streamlit UI (3 modes)
```

#### `chatbot-simple-prompt.py` — Minimal CLI Chatbot

```
Start → Input loop → model.invoke(raw_string) → print reply → exit on "0"
```
- No system prompt, no message history tracking
- Simplest possible chatbot pattern
- Model: `Mistral Small 2506`, `temperature=0.9`

#### `chatbot-system-user-prompt.py` — CLI with Personality

```
Start → SystemMessage("funny AI") → Input loop → model.invoke(prompt) → print reply
```
- Adds a `SystemMessage` to set AI personality ("You are a funny AI agent")
- Maintains a `messages` list of `HumanMessage` / `AIMessage` objects
- Model: `Mistral Small 2506`, `temperature=0.9`

#### `chatbot-system-user-prompt-manual-input.py` — CLI Mode Selector

```
Show menu (1=Angry, 2=Funny, 3=Sad) → User picks mode
→ Set SystemMessage based on choice → Chat loop until "0"
```

| Choice | Mode | System Prompt |
|---|---|---|
| 1 | Angry | "You are an angry AI agent. You respond aggressively and impatiently." |
| 2 | Funny | "You are a funny AI agent. You respond with humor and jokes." |
| 3 | Sad | "You are a sad agent. You respond in a depressed and emotional tone." |

#### `chatbot-ui.py` — ★ Streamlit Chat UI

```
Load .env → Cache Mistral model → Session state (messages, display, thinking)
→ Render header → Show chat history → Show typing indicator
→ st.chat_input → Append HumanMessage → Rerun → Invoke model
→ Append AIMessage → Render bot bubble → Rerun
```

**Features:**
- Professional dark theme with Inter font
- Animated chat bubbles (user: right/blue, bot: left/dark)
- 3-dot typing animation while waiting
- Empty state with starter chips
- "Clear" button resets conversation
- Fixed input bar at bottom with purple glow on focus
- Fully responsive for mobile

#### `chatbot-manual-input-ui.py` — ★ Streamlit Mode-Select Chat UI

```
Screen 1: Mode selection grid (3 personality cards)
         → User clicks "Select 😤 / 😄 / 😢"
         → Sets SystemMessage, transitions to Screen 2

Screen 2: Chat interface with active mode colors
         → "🎭 Change Mode" resets to Screen 1
         → "🗑 Clear" resets conversation within same mode
```

**Personality Themes:**

| Mode | Color | System Prompt |
|---|---|---|
| 😤 Angry | Red | Aggressive & impatient |
| 😄 Funny | Amber | Humorous & witty |
| 😢 Sad | Blue | Depressed & emotional |

---

### 3️⃣ Embedding Models

Demonstrates how to convert text into **numerical vectors** for semantic search, similarity comparison, and RAG pipelines.

#### What are Embeddings?

```
Text String  →  Embedding Model  →  [0.12, -0.45, 0.87, ...]  (vector of floats)

Similar texts produce vectors that are close together in vector space.
This enables: semantic search, document similarity, RAG, clustering.
```

#### OpenAI Embeddings  (`embedding-models/open-ai/`)

| File | Method | Model | Dimensions | Input |
|---|---|---|---|---|
| `embeddings-documents.py` | `embed_documents()` | text-embedding-3-large | 64 | List of 3 strings |
| `embeddings-query.py` | `embed_query()` | text-embedding-3-large | 64 | Single query string |

```python
# embeddings-documents.py — embeds a batch
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=64)
result = embeddings.embed_documents(["text1", "text2", "text3"])

# embeddings-query.py — embeds one query
result = embeddings.embed_query("You are going to learn Gen AI")
```

#### Hugging Face Embeddings  (`embedding-models/hugging-face/`)

| File | Method | Model | Access | Input |
|---|---|---|---|---|
| `embeddings-download-local.py` | `embed_documents()` | all-MiniLM-L6-v2 | Local (downloads) | 3 hardcoded strings |

```python
# Runs entirely offline after first download — no API key required
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
result = embeddings.embed_documents([...])
```

---

### 4️⃣ Prompt Templates

#### `prompt-templates/movie-info-extractor.py`

Demonstrates **LangChain prompt templates** chained with a model using the pipe (`|`) operator.

```
ChatPromptTemplate  →  |  →  ChatOpenAI (GPT-4.1, temp=0)
                              ↓
                    chain.invoke({"paragraph": text})
                              ↓
                    Structured movie fields printed
```

**System prompt rules enforced:**
- Do NOT add explanations or extra commentary
- Follow the exact output format
- Write `NULL` for missing fields
- Keep summary to 2–3 lines
- Do NOT guess unknown facts

**Output format:**
```
Movie Title      :
Release Year     :
Genre            :
Director         :
Main Cast        :
Setting/Location :
Plot             :
Themes           :
Ratings          :
Notable Features :
Short Summary    :
```

---

### 5️⃣ SwapAnalytics Project

A **full-stack AI application** demonstrating production-style movie information extraction with two approaches — free-text prompt and Pydantic schema.

```
project/SwapAnalytics/
├── prompt-templates/           → Free-text extraction (Mistral)
│   ├── core.py                 → CLI version
│   └── app.py                  → Streamlit UI
└── prompt-templates-with-schema/  → Schema-validated extraction (Pydantic)
    ├── core.py                 → CLI version
    └── app.py                  → Streamlit UI
```

#### `prompt-templates/` — Text-Format Extractor

**`core.py`** (CLI):
```
input("Enter paragraph") → prompt.invoke({paragraph}) → model.invoke() → print(response.content)
```

**`app.py`** (Streamlit UI):

```
Top Nav (SwapAnalytics brand + Live badge)
Hero (headline + description)
Input section (centered textarea + extract button)
         ↓
  [Extract Movie Information]
         ↓
Result Banner ("Extraction Complete · 10 Fields Analysed")
         ↓
Field Grid (2 rows × 5 cards):
  🎬 Title  📅 Released  🎭 Genre  🎥 Director  👥 Cast
  🌍 Setting  📖 Plot  💡 Themes  ⭐ Ratings  🏆 Notable Features
         ↓
📝 Short Summary (left-accent card)
         ↓
🗒 Raw Model Output (expander)
```

#### `prompt-templates-with-schema/` — Pydantic Schema Extractor

**Pydantic `Movie` model:**

```python
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str
```

**`core.py`** (CLI):
```
input("Enter paragraph")
→ prompt.invoke({paragraph, format_instructions})  # Pydantic format injected
→ model.invoke()
→ print(response.content)  # JSON string
```

**`app.py`** (Streamlit UI):

```
Top Nav (⬡ Pydantic Schema badge + Live)
Hero (Schema-Driven Movie Extraction)
Centered Input (textarea + "Extract with Schema" button)
         ↓
  [⬡ Extract with Schema]
         ↓
Movie Hero Card:
  ┌──────────────────────────────────────────┐
  │  🎬 Movie Title                 [★★★★★]  │
  │  📅 Year chip  🎥 Director chip  8.4/10  │
  └──────────────────────────────────────────┘
         ↓
Genre Tags (purple pills)  |  Cast Tags (teal pills)
         ↓
📝 Summary (left-accent summary card)
         ↓
⬡ Pydantic Schema Data Table (all 6 typed fields)
         ↓
{ } JSON Output (copyable st.code block with copy button)
         ↓
🗒 Raw Model Output (expander)
```

**JSON output is type-safe and copyable:**
```json
{
  "title": "3 Idiots",
  "release_year": 2009,
  "genre": ["Comedy", "Drama"],
  "director": "Rajkumar Hirani",
  "cast": ["Aamir Khan", "R. Madhavan", "Sharman Joshi"],
  "rating": 8.4,
  "summary": "..."
}
```

---

## 🤝 Supported AI Providers

| Provider | Models Used | Access | LangChain Package | Env Key |
|---|---|---|---|---|
| **OpenAI** | GPT-5, GPT-4.1 | Remote API | `langchain-openai` | `OPENAI_API_KEY` |
| **Groq** | Llama 4 Scout 17B | Remote API | `langchain-groq` | `GROQ_API_KEY` |
| **Mistral AI** | mistral-small-2506 | Remote API | `langchain-mistralai` | `MISTRAL_API_KEY` |
| **Google Gemini** | gemini-2.5-flash-lite | Remote API | `langchain-google-genai` | `GOOGLE_API_KEY` |
| **Hugging Face** | DeepSeek-R1, Phi-3-Mini, TinyLlama | Remote + Local | `langchain-huggingface` | `HUGGINGFACEHUB_API_TOKEN` |

---

## 🔧 Script Variants Explained

Every provider folder contains three script types that build on each other:

```
chat.py                              ← Start here
  └─ Uses init_chat_model()            (provider-agnostic unified API)
  
chat-with-class-usage.py             ← Go deeper
  └─ Uses ChatOpenAI / ChatGroq etc    (provider-specific class, more control)

chat-with-class-usage-with-options.py ← Full control
  └─ Adds temperature, max_tokens etc  (fine-tune model behaviour)
```

| Script Pattern | Best For |
|---|---|
| `chat.py` | Quickly switching between providers |
| `chat-with-class-usage.py` | Understanding provider-specific integration |
| `chat-with-class-usage-with-options.py` | Controlling creativity and output length |

---

## 🖥️ Streamlit UIs

Four production-style Streamlit applications — each runnable independently:

| App | File | Port | What it does |
|---|---|---|---|
| **AI Chat (Funny)** | `chatbot/chatbot-ui.py` | 8502 | Chat with a funny AI using Mistral |
| **AI Chat (3 Modes)** | `chatbot/chatbot-manual-input-ui.py` | 8503 | Pick Angry / Funny / Sad personality then chat |
| **Movie Extractor** | `project/SwapAnalytics/prompt-templates/app.py` | 8504 | Paste a paragraph, get 10 structured movie fields |
| **Schema Extractor** | `project/SwapAnalytics/prompt-templates-with-schema/app.py` | 8505 | Pydantic-validated extraction with copyable JSON |

**Run any app:**
```powershell
# Chat UI (funny mode)
.venv\Scripts\streamlit run chatbot/chatbot-ui.py --server.port 8502

# Chat UI (mode selection)
.venv\Scripts\streamlit run chatbot/chatbot-manual-input-ui.py --server.port 8503

# Movie extractor
.venv\Scripts\streamlit run project/SwapAnalytics/prompt-templates/app.py --server.port 8504

# Schema extractor
.venv\Scripts\streamlit run "project/SwapAnalytics/prompt-templates-with-schema/app.py" --server.port 8505
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.8+**
- API keys for the providers you wish to use (see [Environment Variables](#-environment-variables))

### Step 1 — Clone the Repository

```bash
git clone https://github.com/menkar/generative-ai-with-deploy.git
cd generative-ai-with-deploy
```

### Step 2 — Create a Virtual Environment

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

```bash
cp .env.example .env
# Open .env and fill in your API keys
```

### Step 5 — Run a Script or UI

```powershell
# Run any chat model demo
.venv\Scripts\python chat-models/open-ai/chat.py

# Run a CLI chatbot
.venv\Scripts\python chatbot/chatbot-system-user-prompt.py

# Run an embedding demo
.venv\Scripts\python embedding-models/open-ai/embeddings-query.py

# Run the Streamlit chat UI
.venv\Scripts\streamlit run chatbot/chatbot-ui.py --server.port 8502
```

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```env
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
MISTRAL_API_KEY=your_mistral_api_key_here
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here
```

| Variable | Provider | Where to Get It | Used In |
|---|---|---|---|
| `OPENAI_API_KEY` | OpenAI | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | `chat-models/open-ai/`, `embedding-models/open-ai/`, `prompt-templates/` |
| `GROQ_API_KEY` | Groq | [console.groq.com/keys](https://console.groq.com/keys) | `chat-models/groq/` |
| `GOOGLE_API_KEY` | Google | [aistudio.google.com](https://aistudio.google.com/app/apikey) | `chat-models/google-gemini/` |
| `MISTRAL_API_KEY` | Mistral | [console.mistral.ai](https://console.mistral.ai/) | `chat-models/mistral-ai/`, `chatbot/`, `project/SwapAnalytics/` |
| `HUGGINGFACEHUB_API_TOKEN` | Hugging Face | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | `chat-models/hugging-face/` (remote scripts) |

> ⚠️ **Important:** Never commit your `.env` file. It is already in `.gitignore`.
>
> 💡 **Local models** (`local-model.py`, `embeddings-download-local.py`) require **no API key** — they run entirely offline.

---

## 📦 Requirements

```
langchain              # Core LangChain framework
langchain-core         # Base abstractions and interfaces
langchain-community    # Community integrations
langgraph              # Graph-based LLM workflows (future use)
langchain-openai       # OpenAI GPT integration
langchain-google-genai # Google Gemini integration
langchain-groq         # Groq / Llama integration
langchain-mistralai    # Mistral AI integration
langchain-huggingface  # Hugging Face integration
huggingface-hub        # HF model hub access
python-dotenv          # .env file loader
faiss-cpu              # Vector similarity search (future RAG)
tiktoken               # OpenAI token counter
fastapi                # REST API framework (future deployment)
uvicorn                # ASGI server (future deployment)
requests               # HTTP client
streamlit              # Web UI framework
```

---

## 🗺️ Roadmap

- [x] OpenAI GPT integration (GPT-5, GPT-4.1)
- [x] Groq / Llama 4 integration
- [x] Mistral AI integration
- [x] Google Gemini integration
- [x] Hugging Face remote endpoint (DeepSeek-R1)
- [x] Hugging Face local model (TinyLlama, Phi-3-Mini)
- [x] Model parameter configuration (temperature, max_tokens)
- [x] OpenAI & HuggingFace text embeddings
- [x] Prompt templates with structured output rules
- [x] Pydantic schema-driven extraction with type safety
- [x] CLI chatbots (simple, system prompt, mode selection)
- [x] Streamlit chatbot UI (single personality)
- [x] Streamlit chatbot UI (3-mode personality selector)
- [x] Streamlit movie extractor UI (SwapAnalytics)
- [x] Streamlit schema extractor UI (copyable JSON output)
- [ ] Full conversation history in chatbots
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
