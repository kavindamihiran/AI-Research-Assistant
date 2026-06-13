from __future__ import annotations

import html
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

from dotenv import load_dotenv

from core import (
    PROVIDER_ENV_KEYS,
    PROVIDER_NVIDIA_NIM,
    PROVIDER_OPENROUTER,
    get_api_key,
    get_default_model,
    get_provider_options,
    mask_key,
    perform_research,
)


load_dotenv(override=True)

HOST = "127.0.0.1"
PORT = 8510
PROVIDERS = [PROVIDER_NVIDIA_NIM, PROVIDER_OPENROUTER]


def page(title: str, body: str) -> bytes:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #151b22;
      --muted: #5d6975;
      --line: #d9e0e7;
      --panel: #f7f9fb;
      --accent: #76b900;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    main {{
      width: min(1040px, calc(100vw - 32px));
      margin: 32px auto 56px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 34px; }}
    p {{ color: var(--muted); line-height: 1.55; }}
    label {{ display: block; font-weight: 650; margin: 18px 0 7px; }}
    input, select, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 11px 12px;
      font: inherit;
      background: white;
    }}
    textarea {{ min-height: 132px; resize: vertical; }}
    button {{
      margin-top: 18px;
      border: 0;
      border-radius: 7px;
      padding: 12px 16px;
      background: var(--accent);
      color: #101510;
      font-weight: 750;
      cursor: pointer;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    .panel {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-top: 18px;
      background: var(--panel);
    }}
    .result {{
      border-left: 5px solid var(--accent);
      background: #fbfdf8;
    }}
    .error {{
      border-left: 5px solid #c93333;
      background: #fff8f8;
      white-space: pre-wrap;
    }}
    ul {{ padding-left: 22px; }}
    li {{ margin: 7px 0; }}
    @media (max-width: 720px) {{
      .grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 28px; }}
    }}
  </style>
</head>
<body>
  <main>{body}</main>
</body>
</html>""".encode("utf-8")


def provider_model_options(provider: str, selected_model: str) -> str:
    options = []
    for option in get_provider_options(provider):
        selected = " selected" if option.model_id == selected_model else ""
        label = f"{option.label} ({option.model_id})"
        options.append(f'<option value="{html.escape(option.model_id)}"{selected}>{html.escape(label)}</option>')
    return "\n".join(options)


def render_form(
    provider: str = PROVIDER_NVIDIA_NIM,
    model: str | None = None,
    query: str = "",
    result_html: str = "",
) -> bytes:
    model = model or get_default_model(provider)
    provider_options = "\n".join(
        f'<option value="{html.escape(item)}"{" selected" if item == provider else ""}>{html.escape(item)}</option>'
        for item in PROVIDERS
    )
    key_label = PROVIDER_ENV_KEYS[provider]
    current_key = mask_key(get_api_key(provider)) or "not set"

    body = f"""
<h1>AI Research Assistant</h1>
<p>No-websocket review mode. This page uses plain HTTP, so it avoids the Streamlit browser connection issue.</p>

<form method="post">
  <div class="grid">
    <div>
      <label for="provider">Provider</label>
      <select id="provider" name="provider">{provider_options}</select>
    </div>
    <div>
      <label for="model">Model</label>
      <select id="model" name="model">{provider_model_options(provider, model)}</select>
    </div>
  </div>

  <label for="custom_model">Custom model ID</label>
  <input id="custom_model" name="custom_model" placeholder="Optional provider/model-name">

  <label for="api_key">{html.escape(key_label)} <span style="color:#6b7280;font-weight:500;">({html.escape(current_key)})</span></label>
  <input id="api_key" name="api_key" type="password" placeholder="Optional: paste key for this run">

  <label for="query">Research question</label>
  <textarea id="query" name="query" placeholder="What would you like to research?">{html.escape(query)}</textarea>

  <button type="submit">Start Research</button>
</form>

{result_html}
"""
    return page("AI Research Assistant", body)


def render_result(result) -> str:
    findings = "".join(f"<li>{html.escape(item)}</li>" for item in result.key_findings) or "<li>None returned</li>"
    sources = "".join(f"<li>{html.escape(item)}</li>" for item in result.sources) or "<li>None returned</li>"
    link_values = []
    link_text = "\n".join(
        [result.detailed_report or "", result.summary or "", "\n".join(result.sources), "\n".join(result.source_links)]
    )
    for link in [*result.source_links, *re.findall(r"https?://[^\s\]\)\"'>,]+", link_text)]:
        link = link.rstrip(".,;:")
        if link and link not in link_values:
            link_values.append(link)
    links = "".join(
        f'<li><a href="{html.escape(link)}" target="_blank" rel="noreferrer">{html.escape(link)}</a></li>'
        for link in link_values
    ) or "<li>None returned</li>"
    followups = "".join(f"<li>{html.escape(item)}</li>" for item in result.suggested_followups) or "<li>None returned</li>"
    tools = html.escape(", ".join(result.tools_used) or "None returned")

    return f"""
<section class="panel result">
  <h2>{html.escape(result.topic)}</h2>
  <p><strong>Confidence:</strong> {html.escape(result.confidence)}</p>
  <h3>Summary</h3>
  <p>{html.escape(result.summary)}</p>
  <h3>Detailed Report</h3>
  <p style="white-space:pre-wrap;">{html.escape(result.detailed_report or result.summary)}</p>
  <h3>Key Findings</h3>
  <ul>{findings}</ul>
  <h3>Links</h3>
  <ul>{links}</ul>
  <h3>Sources</h3>
  <ul>{sources}</ul>
  <h3>Tools Used</h3>
  <p>{tools}</p>
  <h3>Suggested Follow-ups</h3>
  <ul>{followups}</ul>
</section>
"""


def render_error(message: str) -> str:
    return f'<section class="panel error"><strong>Research failed</strong>\n{html.escape(message)}</section>'


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.respond(render_form())

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        values = parse_qs(self.rfile.read(length).decode("utf-8"))
        provider = values.get("provider", [PROVIDER_NVIDIA_NIM])[0]
        model = values.get("custom_model", [""])[0].strip() or values.get("model", [get_default_model(provider)])[0]
        api_key = values.get("api_key", [""])[0].strip() or get_api_key(provider)
        query = values.get("query", [""])[0].strip()

        result_html = ""
        if not query:
            result_html = render_error("Enter a research question first.")
        elif not api_key:
            result_html = render_error(f"Missing {PROVIDER_ENV_KEYS[provider]}. Paste a key or set it in .env.")
        else:
            os.environ[PROVIDER_ENV_KEYS[provider]] = api_key
            try:
                result_html = render_result(perform_research(provider, api_key, model, query))
            except Exception as exc:
                result_html = render_error(str(exc))

        self.respond(render_form(provider=provider, model=model, query=query, result_html=result_html))

    def log_message(self, format: str, *args) -> None:
        return

    def respond(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


if __name__ == "__main__":
    print(f"Review server running at http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
