# AI Research Assistant

An AI-powered research workspace built with Streamlit, LangChain, and LangGraph. It combines live web search, Wikipedia context, and structured LLM synthesis to produce detailed reports with direct links, sources, key findings, confidence, and suggested follow-up questions.

## What's New

- Multi-agent research team built on LangGraph: a planner, parallel researchers, a writer, and a critic collaborate on each report.
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

## Multi-Agent Mode

The default research mode is a LangGraph team defined in `multi_agent.py`:

```text
planner ──fan-out──▶ researcher × N (parallel, web search + Wikipedia)
                          │
                          ▼
                       writer ──▶ critic ──approved──▶ finalize
                          ▲          │
                          ├─revise───┤
        gap researcher ◀──missing evidence (one extra round max)
```

| Agent | Role |
| --- | --- |
| **Planner** | Splits the request into 2-4 focused sub-questions |
| **Researchers** | One ReAct agent per sub-question, run in parallel, return cited notes |
| **Writer** | Synthesizes all findings into the structured report |
| **Critic** | Scores the draft, then approves it, asks for a revision, or requests research on gaps |

Loops are bounded (at most 3 critic passes and 1 extra research round). Progress is streamed live to the web UI and CLI. Choose **Single agent** in the sidebar or CLI for the faster original ReAct workflow with fewer API calls.

## Project Structure

```text
AI-Research-Assistant/
├── app.py              # Streamlit web UI
├── main.py             # Command-line UI
├── core.py             # Shared provider, model, agent, and parsing logic
├── multi_agent.py      # LangGraph multi-agent research team
├── supervisor_agent.py # Centralized supervisor-led research team
├── tests/              # Offline tests for the multi-agent graph
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

Recommended model IDs are stored in `core.py`, but the model dropdown is also populated live from each provider's `/models` endpoint (cached for an hour; use the ↻ button to refresh). Retired models disappear from the list automatically and newly released ones show up without code changes. If the catalog can't be reached, the curated defaults are shown, and you can always paste any model ID into the custom model field.

## Supervisor Mode (Centralized)

`supervisor_agent.py` builds a hub-and-spoke LangGraph. A supervisor agent decides what happens after every step, and every worker reports back to it:

```text
              ┌──────────── supervisor (LLM) ────────────┐
              │   next: research | write | review | finish
   research ▼ (≤3 parallel)   write ▼     review ▼     finish ▼
         researcher            writer      critic      finalize → END
              └──────── every worker returns to the supervisor ┘
```

The supervisor chooses, and code-level guardrails enforce the rules:

- A draft must exist before review or finish, and findings must exist before writing.
- Questions that already have findings are not researched again.
- Budgets: 8 supervisor steps, 6 researchers in total (at most 3 at once), 3 drafts.
- If the supervisor fails or picks an invalid step, a fixed fallback policy runs: research, write, review, finish.

| | Multi-agent team | Supervisor team |
| --- | --- | --- |
| Who routes | Fixed edges and a Python router | The supervisor LLM, checked by guardrails |
| Flexibility | Same structure every run | Can skip, repeat, or reorder steps |
| Cost | No routing calls | One extra LLM call per step |
| Best for | Predictable, repeatable runs | Open-ended questions that need adaptive research |

Supervisor runs appear in LangSmith as `supervisor-research` with the `supervisor` tag. Each decision and its reason are in the `supervisor` node outputs.

## Run the Tests

```bash
python -m unittest discover -s tests -t .
```

The tests use scripted fake agents, so they need no API key or network.

## Troubleshooting

`Unknown scheme for proxy URL URL('socks://...')`

httpx does not accept the `socks://` scheme from `ALL_PROXY`. Use `socks5://` instead, or unset `ALL_PROXY` when `HTTPS_PROXY` is already set.

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
