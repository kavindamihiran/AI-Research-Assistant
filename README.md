# AI Research Assistant

An AI-powered research workspace with a Streamlit web app and a command-line interface. It combines web search, Wikipedia context, and structured LLM synthesis to produce concise research briefs with sources, key findings, confidence, and suggested follow-up questions.

## What's New

- First-class NVIDIA NIM support through the OpenAI-compatible NVIDIA API endpoint.
- OpenRouter support retained for free/community model workflows.
- Shared research engine for the web app and CLI.
- Richer structured output: summary, key findings, sources, tools used, confidence, and follow-ups.
- Safer API key handling through environment variables or local `.env`.
- Markdown export from the web UI.
- Safer research file output in `research_outputs/`.

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
