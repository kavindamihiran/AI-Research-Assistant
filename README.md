# AI Research Assistant

An AI-powered research workspace built with Streamlit, LangChain, and LangGraph. It combines live web search, Wikipedia context, and structured LLM synthesis to produce detailed reports with direct links, sources, key findings, confidence, and suggested follow-up questions.

## What's New

- First-class NVIDIA NIM support through the OpenAI-compatible NVIDIA API endpoint.
- OpenRouter support retained for free/community model workflows.
- Shared research engine for the web app and CLI.
- Richer structured output: summary, key findings, sources, tools used, confidence, and follow-ups.
- Safer API key handling through environment variables or local `.env`.
- Markdown export from the web UI.
- Safer research file output in `research_outputs/`.

## Technology Stack

| Technology | Purpose |
| --- | --- |
| **Python 3.10+** | Core application language |
| **Streamlit** | Interactive web interface, provider settings, report views, and Markdown export |
| **LangChain** | Model integration, messages, tools, output parsing, and the shared research workflow |
| **LangGraph** | ReAct agent orchestration and multi-step tool-calling workflow |
| **LangSmith** | Optional tracing, debugging, and observability for LangChain and LangGraph runs |
| **NVIDIA NIM** | OpenAI-compatible hosted model provider |
| **OpenRouter** | OpenAI-compatible access to free/community models |
| **DuckDuckGo Search** | Live web research and source discovery |
| **Wikipedia** | Encyclopedia context for background research |
| **Pydantic** | Structured response schemas and output validation |
| **python-dotenv** | Local environment variable and API key loading |

## How It Works

1. The user selects NVIDIA NIM or OpenRouter and enters a research question.
2. LangGraph runs a LangChain ReAct agent with web search, Wikipedia, and file tools.
3. The selected model evaluates tool results and synthesizes a detailed report.
4. Pydantic validates and normalizes the structured response.
5. Streamlit presents the detailed report, summary, direct links, sources, and export options.
6. LangSmith can optionally trace the agent workflow for debugging and performance analysis.

## Project Structure

```text
AI-Research-Assistant/
├── app.py              # Streamlit web UI
├── main.py             # Command-line UI
├── core.py             # Shared provider, model, agent, and parsing logic
├── tools.py            # Search, Wikipedia, and save tools
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
└── README.md
```

## Requirements

- Python 3.10+
- Internet access
- One of these API keys:
  - NVIDIA API key from the NVIDIA API catalog
  - OpenRouter API key from OpenRouter

## Setup

Create and activate a virtual environment:

```bash
python -m venv myenv
```

Windows PowerShell:

```powershell
myenv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source myenv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from the example:

```bash
copy .env.example .env
```

Set at least one key:

```env
NVIDIA_API_KEY=your-nvidia-key
OPENROUTER_API_KEY=your-openrouter-key
```

Do not commit `.env`.

### Optional LangSmith Tracing

LangSmith tracing is optional. To inspect LangChain and LangGraph runs in LangSmith, add:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=ai-research-assistant
```

When these variables are not configured, the research assistant continues to work normally without LangSmith tracing.

## Run the Web App

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, usually `http://localhost:8501`.

In the sidebar, choose:

- Provider: `NVIDIA NIM` or `OpenRouter`
- Model: a default model or a custom model ID
- API key: read from `.env`, or entered and saved through the app

## Run the CLI

```bash
python main.py
```

The CLI prompts for provider, model, optional custom model ID, and the research question.

## NVIDIA NIM Notes

The app uses NVIDIA's OpenAI-compatible endpoint:

```text
https://integrate.api.nvidia.com/v1
```

Default NVIDIA model IDs are stored in `core.py`. NVIDIA's catalog changes over time, so the UI and CLI both include a custom model ID field. If a model is retired or unavailable for your account, copy a current model ID from the NVIDIA API catalog and paste it into the custom model field.

## Troubleshooting

`Missing NVIDIA_API_KEY` or `Missing OPENROUTER_API_KEY`

Add the key to `.env`, or enter it in the web app's API key panel.

`429` rate limit errors

The selected provider or model is rate-limited. Try another model or wait for the limit to reset.

Tool-calling errors

Some models do not support tool use. Choose a tool-capable model or use a provider catalog model that supports chat tools/function calling.

Import errors

Make sure the virtual environment is active, then reinstall:

```bash
pip install -r requirements.txt
```

## Security

Never hard-code API keys in source files. Keep secrets in `.env`, environment variables, or your deployment platform's secret manager. If an API key was pasted into chat, logs, or committed history, rotate it in the provider dashboard.
