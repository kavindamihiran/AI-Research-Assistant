import os
import re
from html import escape
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from core import (
    PROVIDER_ENV_KEYS,
    PROVIDER_NVIDIA_NIM,
    PROVIDER_OPENROUTER,
    ResearchResponse,
    clear_api_key,
    get_api_key,
    get_default_model,
    get_provider_options,
    mask_key,
    perform_research,
    provider_help_url,
    safe_filename,
    save_api_key,
)


load_dotenv(override=True)


PROVIDERS = [PROVIDER_NVIDIA_NIM, PROVIDER_OPENROUTER]
DEFAULT_MODEL_VERSION = "gpt-oss-120b-default"


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
    :root {
        --accent: #8bed62;
        --accent-dark: #56c93f;
        --bg: #0b0f14;
        --sidebar: #0f141b;
        --panel: #151b24;
        --panel-soft: #10161f;
        --field: #0b1118;
        --line: #2a3544;
        --text: #f4f7f2;
        --muted: #b2bdc9;
        --faint: #7f8b99;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    .main .block-container {
        padding-top: 1rem;
        max-width: 1180px;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background: var(--sidebar) !important;
        border-right: 1px solid var(--line) !important;
    }
    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] small {
        color: var(--muted) !important;
    }
    .hero {
        background: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        margin-bottom: .9rem;
        box-shadow: 0 14px 32px rgba(0, 0, 0, .24);
    }
    .hero-kicker {
        color: var(--accent) !important;
        font-size: .9rem;
        font-weight: 800;
        margin-bottom: .25rem;
        text-transform: uppercase;
    }
    .hero-title {
        color: var(--text) !important;
        font-size: 3rem;
        line-height: 1.03;
        font-weight: 850;
        margin: 0;
    }
    .hero-subtitle {
        color: var(--muted) !important;
        font-size: 1.12rem;
        line-height: 1.45;
        margin: .55rem 0 0;
        max-width: 860px;
    }
    .context-row {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: .95rem;
    }
    .context-pill {
        min-height: 34px;
        padding: .38rem .7rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: var(--panel-soft) !important;
        color: #e6eee2 !important;
        font-size: .95rem;
        font-weight: 700;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
        box-shadow: 0 12px 26px rgba(0, 0, 0, .18);
    }
    label, [data-testid="stWidgetLabel"] p {
        color: var(--text) !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
    }
    textarea, input, select, [data-baseweb="select"] > div {
        background: var(--field) !important;
        color: var(--text) !important;
        border-color: var(--line) !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
    }
    textarea {
        min-height: 150px !important;
    }
    textarea:focus, input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 1px var(--accent) !important;
    }
    .stButton button {
        min-height: 44px;
        border-radius: 8px;
        border: 1px solid var(--line) !important;
        background: var(--panel-soft) !important;
        color: var(--text) !important;
        font-weight: 800;
        font-size: 1rem;
    }
    .stButton button[kind="primary"] {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #091107 !important;
    }
    .metric-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .75rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 8px;
        padding: .9rem 1rem;
    }
    .metric-label {
        color: var(--faint) !important;
        font-size: .9rem;
        margin-bottom: .2rem;
    }
    .metric-value {
        color: var(--text) !important;
        font-size: 1.05rem;
        font-weight: 800;
    }
    div[data-testid="stAlert"] {
        background: #132337 !important;
        color: #dbeafe !important;
        border: 1px solid #2f507a !important;
    }
    [data-baseweb="tab"] {
        color: var(--muted) !important;
        font-size: 1rem;
    }
    [aria-selected="true"] {
        color: var(--accent) !important;
    }
    a {
        color: var(--accent) !important;
    }
    code, pre {
        background: var(--field) !important;
        border-color: var(--line) !important;
        color: var(--text) !important;
    }
    p, li, span {
        font-size: 1rem;
    }
    .research-status {
        margin: 1rem 0;
        padding: 1.25rem 1.35rem;
        border: 1px solid rgba(139, 237, 98, .45);
        border-radius: 10px;
        background: #101820;
        box-shadow: 0 14px 34px rgba(0, 0, 0, .28);
    }
    .research-status-title {
        color: var(--accent) !important;
        font-size: 1.55rem !important;
        font-weight: 850;
        line-height: 1.15;
        margin: 0 0 .35rem;
    }
    .research-status-copy {
        color: var(--muted) !important;
        font-size: 1.05rem !important;
        margin: 0;
    }
    .research-progress {
        position: relative;
        height: 8px;
        margin-top: 1rem;
        overflow: hidden;
        border-radius: 999px;
        background: #243040;
    }
    .research-progress::after {
        content: "";
        position: absolute;
        inset: 0;
        width: 42%;
        border-radius: inherit;
        background: linear-gradient(90deg, transparent, var(--accent), transparent);
        animation: research-slide 1.2s ease-in-out infinite;
    }
    @keyframes research-slide {
        0% { transform: translateX(-110%); }
        100% { transform: translateX(260%); }
    }
</style>
""",
    unsafe_allow_html=True,
)


def display_links(links: list[str]) -> None:
    if not links:
        st.caption("No direct links were returned.")
        return
    for index, link in enumerate(links, 1):
        safe_link = link.strip()
        st.markdown(f"{index}. [{safe_link}]({safe_link})")


def collect_display_links(result: ResearchResponse) -> list[str]:
    links = []
    text = "\n".join([result.detailed_report, result.summary, "\n".join(result.sources), "\n".join(result.source_links)])
    for link in [*result.source_links, *re.findall(r"https?://[^\s\]\)\"'>,]+", text)]:
        link = link.rstrip(".,;:")
        if link and link not in links:
            links.append(link)
    return links


def render_hero(provider: str, model_name: str | None = None) -> None:
    model_html = (
        f'<span class="context-pill">Model: {escape(model_name)}</span>'
        if model_name
        else ""
    )
    st.markdown(
        f"""
<section class="hero">
  <div class="hero-kicker">Research Workspace</div>
  <h1 class="hero-title">AI Research Assistant</h1>
  <p class="hero-subtitle">Build detailed, source-linked research reports with web search, Wikipedia context, and NVIDIA NIM models.</p>
  <div class="context-row">
    <span class="context-pill">Provider: {escape(provider)}</span>
    {model_html}
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def ensure_state() -> None:
    if "provider" not in st.session_state:
        st.session_state.provider = PROVIDER_NVIDIA_NIM
    if "model_name" not in st.session_state:
        st.session_state.model_name = get_default_model(st.session_state.provider)
    if st.session_state.get("default_model_version") != DEFAULT_MODEL_VERSION:
        st.session_state.model_name = get_default_model(st.session_state.provider)
        st.session_state.default_model_version = DEFAULT_MODEL_VERSION
    if "custom_model" not in st.session_state:
        st.session_state.custom_model = ""
    if "research_results" not in st.session_state:
        st.session_state.research_results = None
    if "research_error" not in st.session_state:
        st.session_state.research_error = None
    if "history" not in st.session_state:
        st.session_state.history = []


def active_model(provider: str) -> str:
    custom = st.session_state.get("custom_model", "").strip()
    return custom or st.session_state.get("model_name") or get_default_model(provider)


def render_key_setup(provider: str) -> None:
    render_hero(provider)

    env_key = PROVIDER_ENV_KEYS[provider]
    st.info(f"Missing `{env_key}`. Enter a key below or add it to your `.env` file.")

    with st.form("api_key_form"):
        api_key_input = st.text_input(
            f"{provider} API key",
            type="password",
            placeholder="Paste your API key",
            help=f"Create or manage keys at {provider_help_url(provider)}",
        )
        save_to_env = st.checkbox("Save key to .env for future sessions", value=True)
        submitted = st.form_submit_button("Start researching", type="primary", use_container_width=True)

    if submitted:
        if not api_key_input.strip():
            st.error("Please enter an API key.")
            return
        if save_to_env:
            save_api_key(provider, api_key_input.strip())
        else:
            os.environ[env_key] = api_key_input.strip()
        st.rerun()

    st.markdown(f"[Get a {provider} API key]({provider_help_url(provider)})")


def render_sidebar() -> tuple[str, str, str | None]:
    with st.sidebar:
        st.title("Research Lab")
        st.caption("Models, keys, and runtime settings")

        provider = st.selectbox(
            "Provider",
            PROVIDERS,
            index=PROVIDERS.index(st.session_state.provider),
            help="NVIDIA NIM uses the NVIDIA API catalog. OpenRouter keeps the original free-model workflow.",
        )

        if provider != st.session_state.provider:
            st.session_state.provider = provider
            st.session_state.model_name = get_default_model(provider)
            st.session_state.custom_model = ""
            st.session_state.research_results = None
            st.session_state.research_error = None
            st.rerun()

        options = get_provider_options(provider)
        labels = [option.label for option in options]
        ids_by_label = {option.label: option.model_id for option in options}
        current_model = st.session_state.model_name
        current_label = next((option.label for option in options if option.model_id == current_model), labels[0])
        selected_label = st.selectbox(
            "Model",
            labels,
            index=labels.index(current_label),
        )
        st.session_state.model_name = ids_by_label[selected_label]

        custom_model = st.text_input(
            "Custom model ID",
            value=st.session_state.custom_model,
            placeholder="Optional: provider/model-name",
            help="Use this when NVIDIA or OpenRouter exposes a newer model than the defaults.",
        )
        st.session_state.custom_model = custom_model.strip()

        api_key = get_api_key(provider)
        env_key = PROVIDER_ENV_KEYS[provider]
        with st.expander("API key", expanded=False):
            st.caption(f"Environment variable: `{env_key}`")
            st.text_input("Current key", value=mask_key(api_key), disabled=True)
            new_key = st.text_input("Replace key", type="password", placeholder="New API key")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Update", use_container_width=True):
                    if new_key.strip():
                        save_api_key(provider, new_key.strip())
                        st.success("Key updated.")
                        st.rerun()
                    st.warning("Enter a key first.")
            with col_b:
                if st.button("Clear", use_container_width=True):
                    clear_api_key(provider)
                    st.rerun()

        st.divider()
        st.caption("Tips")
        st.write("Ask for date ranges, comparisons, source links, or detailed sections when you need a deeper report.")

    return provider, active_model(provider), get_api_key(provider)


def render_result(result: ResearchResponse) -> None:
    st.success("Research complete")
    escaped_topic = escape(result.topic)
    escaped_confidence = escape(result.confidence.title())
    links = collect_display_links(result)

    st.markdown(
        f"""
<div class="metric-strip">
  <div class="metric-card"><div class="metric-label">Topic</div><div class="metric-value">{escaped_topic}</div></div>
  <div class="metric-card"><div class="metric-label">Confidence</div><div class="metric-value">{escaped_confidence}</div></div>
  <div class="metric-card"><div class="metric-label">Links</div><div class="metric-value">{len(links)}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

    tab_report, tab_summary, tab_links, tab_sources, tab_export = st.tabs(
        ["Detailed Report", "Summary", "Links", "Sources", "Export"]
    )

    with tab_report:
        st.markdown(result.detailed_report or result.summary)

    with tab_summary:
        st.markdown("#### Summary")
        st.write(result.summary)
        if result.key_findings:
            st.markdown("#### Key Findings")
            for finding in result.key_findings:
                st.markdown(f"- {finding}")
        if result.suggested_followups:
            st.markdown("#### Suggested Follow-ups")
            for followup in result.suggested_followups:
                st.markdown(f"- {followup}")

    with tab_links:
        display_links(links)

    with tab_sources:
        if result.sources:
            for index, source in enumerate(result.sources, 1):
                st.markdown(f"{index}. {source}")
        else:
            st.caption("No sources were returned by the model.")
        if result.tools_used:
            st.markdown("#### Tools Used")
            st.write(", ".join(result.tools_used))

    with tab_export:
        markdown = build_markdown_export(result)
        st.download_button(
            label="Download Markdown",
            data=markdown,
            file_name=f"{safe_filename(result.topic)}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        st.code(markdown, language="markdown")


def build_markdown_export(result: ResearchResponse) -> str:
    findings = "\n".join(f"- {item}" for item in result.key_findings) or "- None returned"
    sources = "\n".join(f"- {source}" for source in result.sources) or "- None returned"
    links = "\n".join(f"- {link}" for link in collect_display_links(result)) or "- None returned"
    followups = "\n".join(f"- {item}" for item in result.suggested_followups) or "- None returned"
    tools = ", ".join(result.tools_used) or "None returned"
    return f"""# {result.topic}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Confidence: {result.confidence}

## Summary
{result.summary}

## Detailed Report
{result.detailed_report or result.summary}

## Key Findings
{findings}

## Links
{links}

## Sources
{sources}

## Tools Used
{tools}

## Suggested Follow-ups
{followups}
"""


def main() -> None:
    ensure_state()
    provider, model_name, api_key = render_sidebar()

    if not api_key:
        render_key_setup(provider)
        return

    render_hero(provider, model_name)

    with st.container(border=True):
        with st.form("research_form"):
            query = st.text_area(
                "Research question",
                placeholder="Example: Research quantum computing applications in healthcare, finance, and logistics. Include links and detailed analysis.",
                height=140,
            )
            col_a, col_b = st.columns([1, 3])
            with col_a:
                submitted = st.form_submit_button("Start Research", type="primary", use_container_width=True)
            with col_b:
                st.caption("The assistant will produce a detailed report, direct links, sources, and exportable Markdown.")

    col_clear, _ = st.columns([1, 4])
    with col_clear:
        if st.button("Clear Results", use_container_width=True):
            st.session_state.research_results = None
            st.session_state.research_error = None
            st.rerun()

    if submitted:
        if not query.strip():
            st.warning("Enter a research question first.")
        else:
            status_slot = st.empty()
            status_slot.markdown(
                """
<div class="research-status">
  <div class="research-status-title">Researching and synthesizing...</div>
  <p class="research-status-copy">Gathering sources, checking links, and building a detailed report.</p>
  <div class="research-progress"></div>
</div>
""",
                unsafe_allow_html=True,
            )
            with st.spinner(""):
                try:
                    result = perform_research(provider, api_key, model_name, query.strip())
                    st.session_state.research_results = result
                    st.session_state.research_error = None
                    st.session_state.history.insert(
                        0,
                        {
                            "time": datetime.now().strftime("%H:%M"),
                            "provider": provider,
                            "model": model_name,
                            "topic": result.topic,
                        },
                    )
                    st.session_state.history = st.session_state.history[:8]
                except Exception as exc:
                    st.session_state.research_results = None
                    st.session_state.research_error = str(exc)
                finally:
                    status_slot.empty()

    if st.session_state.research_error:
        st.error(st.session_state.research_error)

    if st.session_state.research_results:
        render_result(st.session_state.research_results)

    if st.session_state.history:
        with st.expander("Recent research", expanded=False):
            for item in st.session_state.history:
                st.markdown(f"- {item['time']} | {item['provider']} | `{item['model']}` | {item['topic']}")


if __name__ == "__main__":
    main()
