import json
import os
import re
import uuid
import builtins
from dataclasses import dataclass
from typing import Iterable

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from tools import save_tool, search_tool, wiki_tool


builtins.uuid = uuid  # Compatibility shim for newer Python/LangGraph combinations.
load_dotenv(override=True)


class ResearchResponse(BaseModel):
    topic: str = Field(description="The researched topic or question.")
    summary: str = Field(description="A concise executive summary of the report.")
    detailed_report: str = Field(
        default="",
        description="A comprehensive, multi-section research report with evidence, analysis, limitations, and implications.",
    )
    key_findings: list[str] = Field(default_factory=list, description="Important findings.")
    sources: list[str] = Field(default_factory=list, description="Source names or URLs used.")
    source_links: list[str] = Field(default_factory=list, description="Direct source URLs used in the research.")
    tools_used: list[str] = Field(default_factory=list, description="Tools used during research.")
    confidence: str = Field(default="medium", description="low, medium, or high.")
    suggested_followups: list[str] = Field(default_factory=list, description="Useful next questions.")


@dataclass(frozen=True)
class ModelOption:
    provider: str
    model_id: str
    label: str
    supports_tools: bool = True


PROVIDER_OPENROUTER = "OpenRouter"
PROVIDER_NVIDIA_NIM = "NVIDIA NIM"


MODEL_OPTIONS: list[ModelOption] = [
    ModelOption(PROVIDER_OPENROUTER, "openai/gpt-oss-120b:free", "GPT OSS 120B Free"),
    ModelOption(PROVIDER_OPENROUTER, "openai/gpt-oss-20b:free", "GPT OSS 20B Free"),
    ModelOption(PROVIDER_OPENROUTER, "qwen/qwen3-next-80b-a3b-instruct:free", "Qwen3 Next 80B Free"),
    ModelOption(PROVIDER_OPENROUTER, "qwen/qwen3-coder:free", "Qwen3 Coder Free"),
    ModelOption(PROVIDER_OPENROUTER, "nvidia/nemotron-3-super-120b-a12b:free", "Nemotron 3 Super Free"),
    ModelOption(PROVIDER_OPENROUTER, "nvidia/nemotron-3-nano-30b-a3b:free", "Nemotron 3 Nano Free"),
    ModelOption(PROVIDER_NVIDIA_NIM, "openai/gpt-oss-120b", "GPT OSS 120B"),
    ModelOption(PROVIDER_NVIDIA_NIM, "nvidia/llama-3.3-nemotron-super-49b-v1", "Llama Nemotron Super 49B"),
    ModelOption(PROVIDER_NVIDIA_NIM, "moonshotai/kimi-k2.6", "Kimi K2.6"),
    ModelOption(PROVIDER_NVIDIA_NIM, "z-ai/glm-5.1", "GLM 5.1"),
    ModelOption(PROVIDER_NVIDIA_NIM, "deepseek-ai/deepseek-v4-flash", "DeepSeek V4 Flash"),
    ModelOption(PROVIDER_NVIDIA_NIM, "meta/llama-3.1-70b-instruct", "Llama 3.1 70B Instruct"),
]


PROVIDER_ENV_KEYS = {
    PROVIDER_OPENROUTER: "OPENROUTER_API_KEY",
    PROVIDER_NVIDIA_NIM: "NVIDIA_API_KEY",
}


def get_provider_options(provider: str) -> list[ModelOption]:
    return [option for option in MODEL_OPTIONS if option.provider == provider]


def get_default_model(provider: str) -> str:
    options = get_provider_options(provider)
    return options[0].model_id if options else ""


def get_api_key(provider: str) -> str | None:
    env_key = PROVIDER_ENV_KEYS[provider]
    return os.getenv(env_key)


def mask_key(value: str | None) -> str:
    if not value:
        return ""
    if len(value) <= 14:
        return "***"
    return f"{value[:8]}...{value[-4:]}"


def save_api_key(provider: str, api_key: str) -> None:
    from dotenv import set_key

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        open(env_path, "a", encoding="utf-8").close()

    env_key = PROVIDER_ENV_KEYS[provider]
    set_key(env_path, env_key, api_key)
    os.environ[env_key] = api_key


def clear_api_key(provider: str) -> None:
    from dotenv import set_key

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    env_key = PROVIDER_ENV_KEYS[provider]
    if os.path.exists(env_path):
        set_key(env_path, env_key, "")
    os.environ.pop(env_key, None)


def provider_help_url(provider: str) -> str:
    if provider == PROVIDER_NVIDIA_NIM:
        return "https://build.nvidia.com/explore/discover"
    return "https://openrouter.ai/keys"


def build_llm(provider: str, api_key: str, model_name: str, json_mode: bool = False) -> ChatOpenAI:
    extra_kwargs = {"model_kwargs": {"response_format": {"type": "json_object"}}} if json_mode else {}
    max_tokens = 4096 if provider == PROVIDER_OPENROUTER else 8192

    if provider == PROVIDER_NVIDIA_NIM:
        return ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base="https://integrate.api.nvidia.com/v1",
            streaming=False,
            disable_streaming=True,
            temperature=0.2,
            max_completion_tokens=max_tokens,
            **extra_kwargs,
        )

    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        streaming=False,
        disable_streaming=True,
        temperature=0.2,
        max_completion_tokens=max_tokens,
        default_headers={
            "HTTP-Referer": "https://github.com/AI-Research-Assistant",
            "X-Title": "AI Research Assistant",
        },
        **extra_kwargs,
    )


def build_agent(provider: str, api_key: str, model_name: str):
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    llm = build_llm(provider, api_key, model_name)
    tools = [search_tool, wiki_tool, save_tool]

    system_prompt = f"""
You are an AI research assistant. Use the available tools when the question benefits
from current web context, encyclopedia context, or saving intermediate notes.

Research rules:
- Use 2-6 focused tool calls for broad research questions, then stop.
- Prefer current web search for modern or fast-moving topics.
- Put direct URLs in source_links. Also include readable source names in sources.
- Separate established facts from uncertainty.
- Keep the summary clear, neutral, and useful.
- Always write detailed_report as a substantial report, not a short paragraph. Include sections for overview, background, key evidence, analysis, limitations, practical implications, and next steps when relevant.
- Return detailed_report as one string, not an array or object.
- Include inline source names or URLs in detailed_report when the tool results contain links.
- Do not invent citations.

Return only JSON matching this schema:
{parser.get_format_instructions()}
"""

    return create_react_agent(llm, tools, prompt=system_prompt), parser


def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("No JSON object found in the model response.")
    return json.loads(match.group())


def stringify_report(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                title = item.get("section") or item.get("title") or item.get("heading")
                content = item.get("content") or item.get("text") or item.get("body") or item
                if title:
                    parts.append(f"## {title}\n{stringify_report(content)}")
                else:
                    parts.append(stringify_report(content))
            else:
                parts.append(stringify_report(item))
        return "\n\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return "\n\n".join(f"## {key}\n{stringify_report(item)}" for key, item in value.items())
    return str(value).strip()


def string_list(value: object) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    cleaned = []
    for item in items:
        if isinstance(item, dict):
            text = item.get("url") or item.get("link") or item.get("source") or item.get("title") or json.dumps(item)
        else:
            text = str(item)
        text = text.strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def coerce_research_response(data: dict, query: str = "", transcript: str = "") -> ResearchResponse:
    data = dict(data)
    data["topic"] = stringify_report(data.get("topic")) or query
    data["summary"] = stringify_report(data.get("summary"))
    data["detailed_report"] = stringify_report(data.get("detailed_report"))
    data["key_findings"] = string_list(data.get("key_findings"))
    data["sources"] = string_list(data.get("sources"))
    data["source_links"] = string_list(data.get("source_links"))
    data["tools_used"] = string_list(data.get("tools_used"))
    data["confidence"] = stringify_report(data.get("confidence")) or "medium"
    data["suggested_followups"] = string_list(data.get("suggested_followups"))

    if not data["summary"] and data["detailed_report"]:
        data["summary"] = data["detailed_report"][:1000]
    if not data["detailed_report"] and data["summary"]:
        data["detailed_report"] = data["summary"]

    return enrich_response(ResearchResponse(**data), query, transcript)


def parse_response(parser: PydanticOutputParser, messages: Iterable[object]) -> ResearchResponse:
    for message in reversed(list(messages)):
        content = getattr(message, "content", None)
        if isinstance(content, str) and "{" in content and "}" in content:
            try:
                return coerce_research_response(extract_json(content))
            except Exception:
                return parser.parse(content)

    raise ValueError("No structured final response was returned by the model.")


def extract_links(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s\]\)\"'>,]+", text)
    cleaned = []
    for url in urls:
        url = url.rstrip(".,;:")
        if url and url not in cleaned:
            cleaned.append(url)
    return cleaned


def enrich_response(response: ResearchResponse, query: str = "", transcript: str = "") -> ResearchResponse:
    text_for_links = "\n".join(
        [
            response.summary or "",
            response.detailed_report or "",
            "\n".join(response.sources),
            "\n".join(response.source_links),
            transcript,
        ]
    )
    links = []
    for link in [*response.source_links, *extract_links(text_for_links)]:
        if link and link not in links:
            links.append(link)

    if not response.detailed_report:
        response.detailed_report = response.summary
    if not response.topic and query:
        response.topic = query
    response.source_links = links[:20]
    return response


def render_transcript(messages: Iterable[object], max_chars: int = 14000) -> str:
    rendered = []
    for message in messages:
        role = getattr(message, "type", message.__class__.__name__)
        name = getattr(message, "name", None)
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)

        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        elif content is None:
            content = ""

        prefix = f"{role}"
        if name:
            prefix += f":{name}"
        rendered.append(f"{prefix}: {content}")

        if tool_calls:
            rendered.append(f"{role}:tool_calls: {json.dumps(tool_calls, ensure_ascii=False)}")

    transcript = "\n\n".join(rendered)
    if len(transcript) <= max_chars:
        return transcript
    return transcript[-max_chars:]


def fallback_response(query: str, messages: Iterable[object], error: Exception | None = None) -> ResearchResponse:
    transcript = render_transcript(messages, max_chars=6000)
    candidates = []
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, str) and content.strip():
            candidates.append(content.strip())

    for candidate in reversed(candidates):
        if "{" in candidate and "}" in candidate:
            try:
                return coerce_research_response(extract_json(candidate), query, transcript)
            except Exception:
                pass

    summary = candidates[-1] if candidates else "Research completed, but the model did not return structured output."
    urls = sorted(set(re.findall(r"https?://[^\s\]\)\"'>]+", transcript)))
    tools_used = sorted(set(re.findall(r"(search|wikipedia|save_text_to_file)", transcript, flags=re.IGNORECASE)))
    if error:
        summary = f"{summary}\n\nStructured-output fallback was used because: {error}"

    return ResearchResponse(
        topic=query,
        summary=summary[:4000],
        detailed_report=summary[:8000],
        key_findings=[],
        sources=urls[:10],
        source_links=urls[:20],
        tools_used=tools_used,
        confidence="low",
        suggested_followups=[],
    )


def normalize_response(
    provider: str,
    api_key: str,
    model_name: str,
    query: str,
    messages: Iterable[object],
) -> ResearchResponse:
    transcript = render_transcript(messages)
    schema = ResearchResponse.model_json_schema()
    system_prompt = """
You convert research-agent transcripts into valid JSON.
Return only one JSON object. Do not use markdown, code fences, comments, or prose.
Use only information present in the transcript. Do not invent sources.
If a field is unknown, use an empty list, "medium", or a concise uncertainty note.
The detailed_report field must be a comprehensive multi-section report, not just a summary.
The detailed_report field must be one string. Do not return an array or object for detailed_report.
Extract every direct URL you can find into source_links.
"""
    user_prompt = f"""
Original query:
{query}

Required JSON schema:
{json.dumps(schema, ensure_ascii=False)}

Research transcript:
{transcript}
"""

    attempts = [
        build_llm(provider, api_key, model_name, json_mode=True),
        build_llm(provider, api_key, model_name, json_mode=False),
    ]

    last_error: Exception | None = None
    for llm in attempts:
        try:
            response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            return coerce_research_response(extract_json(str(content)), query, transcript)
        except Exception as exc:
            last_error = exc

    return fallback_response(query, messages, last_error)


def direct_structured_response(provider: str, api_key: str, model_name: str, query: str, error: Exception) -> ResearchResponse:
    system_prompt = """
You are an AI research assistant. Return only valid JSON with these keys:
topic, summary, detailed_report, key_findings, sources, source_links, tools_used, confidence, suggested_followups.
Use an empty sources list if no external tools were available. Do not use markdown.
The detailed_report field must be a comprehensive multi-section report, not just a summary.
The detailed_report field must be one string. Do not return an array or object for detailed_report.
"""
    user_prompt = f"""
The tool-using research agent failed with this error:
{error}

Answer the user's query directly in valid JSON:
{query}
"""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]

    for json_mode in (True, False):
        try:
            response = build_llm(provider, api_key, model_name, json_mode=json_mode).invoke(messages)
            content = getattr(response, "content", "")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            parsed = coerce_research_response(extract_json(str(content)), query)
            if not parsed.tools_used:
                parsed.tools_used = ["direct_model_fallback"]
            if not parsed.confidence:
                parsed.confidence = "low"
            return parsed
        except Exception:
            continue

    return fallback_response(query, [HumanMessage(content=query), HumanMessage(content=str(error))], error)


def perform_research(provider: str, api_key: str, model_name: str, query: str) -> ResearchResponse:
    agent, parser = build_agent(provider, api_key, model_name)
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            {"recursion_limit": 20},
        )
    except Exception as exc:
        return direct_structured_response(provider, api_key, model_name, query, exc)

    try:
        return enrich_response(parse_response(parser, result["messages"]), query, render_transcript(result["messages"]))
    except Exception:
        return normalize_response(provider, api_key, model_name, query, result["messages"])


def safe_filename(value: str, fallback: str = "research") -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower()).strip("._-")
    return cleaned[:80] or fallback
