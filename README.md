# 🔬 AI Research Assistant

> **A powerful AI-driven research tool that combines web search, Wikipedia knowledge, and intelligent summarization to deliver comprehensive research results in seconds.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-green.svg)](https://langchain.com)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini%20AI-orange.svg)](https://ai.google.dev)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-red.svg)](https://streamlit.io)

## 📖 Overview

The AI Research Assistant is an intelligent research companion built with **LangChain** and **Google Gemini AI**. It combines multiple information sources to provide structured, comprehensive research summaries with automatic source tracking and file management.

### 🎯 What Makes It Special

- **Multi-Source Intelligence**: Seamlessly integrates web search (DuckDuckGo) and Wikipedia
- **Dual Interface**: Modern web UI + command-line interface for different workflows
- **Structured Output**: AI-generated summaries with proper citations and source tracking
- **Smart File Management**: Automatic timestamped saves with organized output
- **Production Ready**: Built-in error handling, retry mechanisms, and rate limiting

---

## ✨ Key Features

### 🔍 **Intelligent Research**

- **Web Search**: Real-time web search using DuckDuckGo for latest information
- **Wikipedia Integration**: Access to comprehensive encyclopedia knowledge
- **AI-Powered Synthesis**: Google Gemini AI creates structured, coherent summaries
- **Source Attribution**: Automatic tracking and citation of all information sources

### 🖥️ **Dual Interface Options**

- **Web Interface**: Beautiful Streamlit-powered UI with real-time progress
- **Command Line**: Quick terminal-based research for power users
- **Progress Tracking**: Visual feedback during research process
- **Export Options**: Save results as organized text files

### 🛠️ **Production Features**

- **Smart Error Handling**: Graceful handling of API limits and network issues
- **Rate Limiting**: Respectful API usage with built-in delays
- **Environment Management**: Secure API key handling with `.env` files
- **Extensible Architecture**: Easy to add new tools and capabilities

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Python 3.10+ recommended)
- **Google AI API Key** ([Get free key](https://makersuite.google.com/app/apikey))
- **Internet Connection** (for web search and API calls)

### ⚡ Installation

1. **Clone the Repository**

   ```bash
   git clone <repository-url>
   cd AI-Research-Assistant
   ```

2. **Create Virtual Environment** (Recommended)

   ```bash
   # Create virtual environment
   python -m venv myenv

   # Activate (Linux/Mac)
   source myenv/bin/activate

   # Activate (Windows)
   myenv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**

   ```bash
   # Create .env file
   echo "GOOGLE_API_KEY=your_api_key_here" > .env
   ```

   > 💡 **Get your free API key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey)

### 🎉 Ready to Use!

**Web Interface** (Recommended for beginners):

```bash
streamlit run app.py
```

**Command Line** (For quick queries):

```bash
python main.py
```

---

## 💡 Usage Guide

### 🌐 Web Interface (Streamlit)

The web interface provides the most user-friendly experience:

1. **Launch the Application**

   ```bash
   streamlit run app.py
   ```

2. **Access the Interface**

   - Browser opens automatically at `http://localhost:8501`
   - Clean, intuitive design with sidebar navigation
   - Built-in help and feature guides

3. **Key Features**

   - **Query Input**: Large text area for research questions
   - **Real-time Progress**: Visual indicators during research
   - **Structured Results**: Organized summaries with source links
   - **Download Options**: Save results as timestamped files
   - **Settings Panel**: Customize output format and preferences

4. **Sample Workflow**
   ```
   1. Enter query: "Latest developments in renewable energy"
   2. Click "Start Research"
   3. Watch real-time progress indicators
   4. Review structured summary with sources
   5. Download results as text file
   ```

### ⌨️ Command Line Interface

Perfect for quick research tasks and automation:

1. **Launch the CLI**

   ```bash
   python main.py
   ```

2. **Enter Your Query**

   ```
   What can i help you research? Latest AI breakthroughs in 2024
   ```

3. **View Results**
   - Structured output displayed in terminal
   - Automatic save to `research_output.txt`
   - Source attribution included

### 📝 Example Research Queries

**Technology & Science**

- "What is quantum computing and its current applications?"
- "Latest breakthroughs in artificial intelligence 2024"
- "Climate change solutions and renewable energy trends"

**Business & Economics**

- "Cryptocurrency market trends and regulations"
- "Remote work impact on productivity and culture"
- "Sustainable business practices for small companies"

**History & Culture**

- "History of the internet and key milestones"
- "Cultural impact of social media on Gen Z"
- "Space exploration achievements and future missions"

---

## 📁 Project Architecture

```
AI-Research-Assistant/
├── 📜 app.py                  # Streamlit web application
├── 🖥️ main.py                # Command-line interface
├── 🛠️ tools.py               # Research tools (search, wiki, save)
├── 📋 requirements.txt       # Python dependencies
├── 🔐 .env                   # Environment variables (API keys)
├── 📊 research_output.txt    # Generated research files
├── 🗂️ myenv/                 # Virtual environment
└── 📖 README.md              # Documentation

Core Components:
├── 🧠 LangChain Agents       # AI orchestration and tool calling
├── 🔍 Search Tools           # DuckDuckGo + Wikipedia integration
├── 💾 File Management        # Automated saving with timestamps
├── 🎨 Streamlit UI           # Modern web interface
└── 🔧 Error Handling         # Robust retry mechanisms
```

### 🔧 Core Components Explained

**`main.py`** - Command Line Interface

- Handles direct user queries via terminal
- Implements the research agent with tool calling
- Provides structured output parsing
- Simple, focused workflow for power users

**`app.py`** - Web Interface

- Full-featured Streamlit application
- Real-time progress indicators and status updates
- Interactive UI with customizable settings
- Enhanced error handling and user feedback

**`tools.py`** - Research Tools

- **Search Tool**: DuckDuckGo web search integration
- **Wikipedia Tool**: Encyclopedia knowledge access
- **Save Tool**: Automated file management with timestamps
- Easily extensible for additional data sources

---

## ⚙️ API Configuration

### Google Gemini API Setup

1. **Get API Key**

   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create new project or use existing
   - Generate API key (free tier available)

2. **Configure Environment**

   ```bash
   # Method 1: .env file (recommended)
   echo "GOOGLE_API_KEY=your_actual_key_here" > .env

   # Method 2: Environment variable
   export GOOGLE_API_KEY="your_actual_key_here"
   ```

3. **Verify Setup**
   ```bash
   # Test your configuration
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✓ API Key configured' if os.getenv('GOOGLE_API_KEY') else '✗ API Key missing')"
   ```

### 📊 Rate Limits & Best Practices

**Free Tier Limits:**

- 15 requests per minute
- 1,500 requests per day
- Rate limiting automatically handled

**Optimization Tips:**

- Use specific queries for better results
- Combine related questions in single research session
- Monitor usage in Google AI Studio dashboard

---

## 🧩 Dependencies & Technical Stack

### Core Framework

```
langchain              # LLM application framework
langchain-google-genai # Google Gemini integration
langchain-community    # Community tools and utilities
pydantic              # Data validation and parsing
```

### Research Tools

```
duckduckgo-search     # Web search functionality
wikipedia             # Wikipedia API wrapper
python-dotenv         # Environment variable management
```

### User Interface

```
streamlit            # Web application framework
tenacity            # Retry mechanisms with exponential backoff
```

### 🔄 Installation Commands

**Standard Installation:**

```bash
pip install -r requirements.txt
```

**Development Installation:**

```bash
pip install -r requirements.txt
pip install jupyter notebook  # For development/testing
```

---

## 🐛 Troubleshooting & FAQ

### Common Issues & Solutions

#### 🚫 **ModuleNotFoundError**

```bash
# Problem: Missing dependencies
# Solution: Reinstall requirements
pip install --upgrade -r requirements.txt

# Alternative: Use virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

#### 🔑 **API Authentication Errors**

```bash
# Problem: Invalid or missing API key
# Solution: Verify API key setup
cat .env  # Check file contents
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('GOOGLE_API_KEY'))"
```

#### ⏰ **Rate Limit Exceeded**

```bash
# Problem: Too many API calls
# Solution: Wait and implement delays
# The app handles this automatically, but you can:
# 1. Wait 1 minute between requests
# 2. Upgrade to paid Google AI plan
# 3. Use more specific queries
```

#### 🌐 **Network/Connection Issues**

```bash
# Problem: Search tools failing
# Solution: Check internet connection
curl -I https://duckduckgo.com  # Test web access
ping google.com                 # Test general connectivity
```

### 💡 **Performance Optimization**

**For Better Results:**

- Use specific, focused research questions
- Avoid overly broad queries like "tell me about everything"
- Combine related sub-questions in one session

**For Faster Performance:**

- Use command-line interface for simple queries
- Batch multiple related questions together
- Keep API key in `.env` file (faster loading)

### 🔧 **Advanced Configuration**

**Custom Model Selection:**

```python
# In main.py or app.py, modify:
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or "gemini-1.5-pro"
    temperature=0.1,           # Lower = more focused
    max_tokens=2000           # Adjust response length
)
```

**Custom File Output:**

```python
# In tools.py, modify save_to_txt function:
def save_to_txt(data: str, filename: str = "custom_research.txt"):
    # Your custom file handling logic
```

---

## 🤝 Contributing & Support

### 📈 **Future Enhancements**

- [ ] Support for additional AI models (OpenAI, Anthropic)
- [ ] PDF and document upload capabilities
- [ ] Multi-language research support
- [ ] Export to markdown/PDF formats
- [ ] Research project management features

### 🐛 **Found a Bug?**

- Check existing issues in repository
- Provide detailed error messages and steps to reproduce
- Include your Python version and OS information

### 💪 **Want to Contribute?**

- Fork the repository
- Create feature branch (`git checkout -b amazing-feature`)
- Commit changes (`git commit -m 'Add amazing feature'`)
- Push to branch (`git push origin amazing-feature`)
- Open Pull Request

---

### **Acknowledgments**

- **LangChain**: For the excellent LLM application framework
- **Google**: For Gemini AI API and generous free tier
- **Streamlit**: For the beautiful and simple web UI framework
- **DuckDuckGo**: For privacy-focused search capabilities
- **Wikipedia**: For comprehensive knowledge access

_Made by kavinda mihiran._
