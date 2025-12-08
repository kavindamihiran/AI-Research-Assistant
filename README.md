# 🔬 AI Research Assistant

> **A powerful AI-driven research tool that combines web search, Wikipedia knowledge, and intelligent summarization to deliver comprehensive research results in seconds.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)](https://langchain.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-Multi--Model%20API-purple.svg)](https://openrouter.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red.svg)](https://streamlit.io)

---

## 📖 Overview

The AI Research Assistant is an intelligent research companion built with **LangChain** and **OpenRouter AI**. It combines multiple information sources to provide structured, comprehensive research summaries with automatic source tracking and file management.

### 🎯 What Makes It Special

- **Multi-Model Support**: Access to 11+ free AI models via OpenRouter (Gemini, Llama, Mistral, and more)
- **Multi-Source Intelligence**: Seamlessly integrates web search (DuckDuckGo) and Wikipedia
- **Dual Interface**: Modern web UI + command-line interface for different workflows
- **Structured Output**: AI-generated summaries with proper citations and source tracking
- **Smart File Management**: Automatic timestamped saves with organized output
- **100% Free**: Uses OpenRouter's free tier models - no API costs!

---

## ✨ Key Features

### 🔍 **Intelligent Research**

- **Web Search**: Real-time web search using DuckDuckGo for latest information
- **Wikipedia Integration**: Access to comprehensive encyclopedia knowledge
- **AI-Powered Synthesis**: Multiple AI models create structured, coherent summaries
- **Source Attribution**: Automatic tracking and citation of all information sources

### 🤖 **Supported Free Models**

The following models are available via OpenRouter's free tier:

#### OpenAI OSS Models
- `openai/gpt-oss-120b:free`
- `openai/gpt-oss-20b:free`

#### TNG Models
- `tngtech/tng-r1t-chimera:free`

#### Qwen Models
- `qwen/qwen3-coder:free`
- `qwen/qwen3-4b:free`

#### Nvidia Models
- `nvidia/nemotron-nano-12b-v2-vl:free`
- `nvidia/nemotron-nano-9b-v2:free`

#### Other Models
- `alibaba/tongyi-deepresearch-30b-a3b:free`

> ⚠️ **Note on Rate Limits**: The "Free" tier on OpenRouter is limited to **50 requests per day**. If you see 429 errors or "Tools not supported", you have likely hit this limit. Add $10 credits to unlock 1000/day.

### 🖥️ **Dual Interface Options**

- **Web Interface**: Beautiful Streamlit-powered UI with real-time progress
- **Command Line**: Quick terminal-based research for power users
- **Progress Tracking**: Visual feedback during research process
- **Export Options**: Save results as organized text files

---

## 🚀 Quick Start (Beginner Friendly)

### Prerequisites

Before you begin, make sure you have:

- ✅ **Python 3.10 or higher** installed ([Download Python](https://www.python.org/downloads/))
- ✅ **Git** installed ([Download Git](https://git-scm.com/downloads))
- ✅ **OpenRouter API Key** (Free - [Get your key here](https://openrouter.ai/keys))
- ✅ **Internet Connection**

### Step 1: Clone the Repository

Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and run:

```bash
git clone https://github.com/YOUR_USERNAME/AI-Research-Assistant.git
cd AI-Research-Assistant
```

### Step 2: Create Virtual Environment

**Why?** This keeps the project's dependencies isolated from other Python projects.

```bash
# Create virtual environment
python -m venv myenv
```

**Activate the virtual environment:**

| Operating System | Command |
|------------------|---------|
| **Linux/macOS** | `source myenv/bin/activate` |
| **Windows (CMD)** | `myenv\Scripts\activate.bat` |
| **Windows (PowerShell)** | `myenv\Scripts\Activate.ps1` |

> ✅ You'll know it's activated when you see `(myenv)` at the start of your terminal prompt.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages. Wait for it to complete (may take 2-3 minutes).

### Step 4: Get Your OpenRouter API Key

1. Go to [OpenRouter Keys](https://openrouter.ai/keys)
2. Sign up or log in (you can use Google/GitHub)
3. Click "Create Key"
4. Copy your API key (starts with `sk-or-v1-...`)

### Step 5: Configure API Key

**Option A: Using the App (Recommended for Beginners)**

Simply run the app and it will prompt you for your API key on first launch!

**Option B: Create .env file manually**

```bash
# Linux/macOS
echo "OPENROUTER_API_KEY=sk-or-v1-your-key-here" > .env

# Windows (PowerShell)
"OPENROUTER_API_KEY=sk-or-v1-your-key-here" | Out-File -FilePath .env -Encoding utf8
```

### Step 6: Run the Application

**🌐 Web Interface (Recommended for Beginners):**

```bash
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`

**⌨️ Command Line Interface:**

```bash
python main.py
```

---

## 💡 Usage Guide

### 🌐 Using the Web Interface

1. **Launch**: Run `streamlit run app.py`
2. **Enter API Key**: On first run, paste your OpenRouter API key
3. **Select Model**: Choose an AI model from the sidebar dropdown
4. **Enter Query**: Type your research question
5. **Click "Start Research"**: Wait for the AI to gather and synthesize information
6. **View Results**: See structured summary, sources, and tools used
7. **Download**: Save your research as a text file

### 📝 Example Research Queries

**Technology & Science**
- "What is quantum computing and its current applications?"
- "Latest breakthroughs in artificial intelligence 2024"
- "How does CRISPR gene editing work?"

**Business & Economics**
- "Cryptocurrency market trends and regulations"
- "Remote work impact on productivity"
- "Sustainable business practices"

**History & Culture**
- "History of the internet and key milestones"
- "Space exploration achievements and future missions"
- "Cultural impact of social media"

---

## 📁 Project Structure

```
AI-Research-Assistant/
├── 📜 app.py              # Streamlit web application (main UI)
├── 🖥️ main.py             # Command-line interface
├── 🛠️ tools.py            # Research tools (search, wiki, save)
├── 📋 requirements.txt    # Python dependencies with versions
├── 🔐 .env                # Your API key (not committed to git)
├── 📖 README.md           # This documentation
├── 🚫 .gitignore          # Files to ignore in git
└── 🗂️ myenv/              # Virtual environment (not committed)
```

---

## 🐛 Troubleshooting

### ❌ "No endpoints found that support tool use" Error

**Cause**: The selected model doesn't support function/tool calling.

**Solution**: Select a model from the supported list (default: `google/gemini-2.0-flash-exp:free`)

### ❌ ModuleNotFoundError

**Cause**: Dependencies not installed or wrong Python environment.

**Solution**:
```bash
# Make sure virtual environment is activated
source myenv/bin/activate  # Linux/macOS
myenv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### ❌ API Key Error

**Cause**: Missing or invalid API key.

**Solution**:
1. Check your `.env` file exists and contains: `OPENROUTER_API_KEY=sk-or-v1-...`
2. Make sure there are no extra spaces or quotes
3. Verify your key at [OpenRouter Dashboard](https://openrouter.ai/settings/keys)

### ❌ Rate Limit Exceeded (429 Error)

**Cause**: OpenRouter free tier has strict daily limits:
- **50 requests/day** on the completely free tier
- **1000 requests/day** if you add $10 credits to your account

**Solution**:
1. Wait until the next day (limits reset at midnight UTC)
2. Try a different model (some have separate limits)
3. Add credits to your OpenRouter account for higher limits
4. Check your usage at [OpenRouter Activity](https://openrouter.ai/activity)

> **Note**: The error "Tools are not supported in streaming mode" sometimes appears when you're actually rate-limited. Wait and try again.

### ❌ "streamlit: command not found"

**Cause**: Virtual environment not activated or Streamlit not installed.

**Solution**:
```bash
# Activate virtual environment first
source myenv/bin/activate  # Linux/macOS
myenv\Scripts\activate     # Windows

# If still not working, reinstall
pip install streamlit
```

---

## ⚙️ Configuration

### Changing Default Model

Edit `app.py` or `main.py` and modify the `AVAILABLE_MODELS` list or default selection.

### API Key Management

- **In App**: Go to Sidebar → API Key Settings → Update Key
- **Manual**: Edit `.env` file with your new key

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin amazing-feature`)
5. Open Pull Request

---

## 📜 License

This project is open source and available under the MIT License.

---

## 🙏 Acknowledgments

- **LangChain**: For the excellent LLM application framework
- **OpenRouter**: For providing access to multiple AI models with free tier
- **Streamlit**: For the beautiful and simple web UI framework
- **DuckDuckGo**: For privacy-focused search capabilities
- **Wikipedia**: For comprehensive knowledge access

---

_Made with ❤️ by Kavinda Mihiran_
