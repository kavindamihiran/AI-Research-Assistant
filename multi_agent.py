"""Multi-agent research team built on LangGraph.

planner -> researcher (parallel fan-out, one per sub-question) -> writer -> critic
The critic either approves the draft, sends it back to the writer for revision,
or dispatches researchers to fill evidence gaps. Loops are bounded.
"""

import json
import operator
import os
from typing import Annotated, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Send

from core import (
    ResearchResponse,
    build_llm,
    coerce_research_response,
    extract_json,
    extract_links,
    render_transcript,
)
from tools import search_tool, wiki_tool


MAX_SUB_QUESTIONS = 4
MAX_GAP_QUESTIONS = 2
MAX_REVISIONS = 2
MAX_RESEARCH_ROUNDS = 2
MAX_NOTES_CHARS = 5000


class Finding(TypedDict):
    question: str
    notes: str
    sources: list[str]
    tools_used: list[str]


class ResearchState(TypedDict, total=False):
    query: str
    sub_questions: list[str]
    findings: Annotated[list[Finding], operator.add]
    draft: dict
    critique: dict
    revision: int
    research_rounds: int
    final: ResearchResponse


class ResearcherInput(TypedDict):
    query: str
    question: str


ResearchFn = Callable[[str, str], Finding]
JsonLLM = Callable[[str, str], dict]


def make_json_llm(provider: str, api_key: str, model_name: str) -> JsonLLM:
    """Return a function that prompts the model and parses a JSON object, trying JSON mode first."""
    attempts = [
        build_llm(provider, api_key, model_name, json_mode=True),
        build_llm(provider, api_key, model_name, json_mode=False),
    ]

    def invoke(system_prompt: str, user_prompt: str) -> dict:
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        last_error: Exception | None = None
        for llm in attempts:
            try:
                content = llm.invoke(messages).content
                if isinstance(content, list):
                    content = json.dumps(content, ensure_ascii=False)
                return extract_json(str(content))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Model did not return valid JSON: {last_error}")

    return invoke


def make_researcher(provider: str, api_key: str, model_name: str) -> ResearchFn:
    """Return a function that runs a tool-using ReAct agent on one sub-question."""
    agent = create_react_agent(
        build_llm(provider, api_key, model_name),
        [search_tool, wiki_tool],
        prompt="""
You are a research specialist on a team. Investigate ONE sub-question thoroughly.

Rules:
- Make 1-4 focused tool calls (web search for current facts, wikipedia for background), then stop.
- Write concise research notes: key facts, figures, dates, and disagreements between sources.
- Cite the direct URL next to every claim that came from a tool result.
- Say clearly when evidence is thin or conflicting. Never invent sources.
- Reply with plain-text notes only, no JSON.
""",
    )

    def research(query: str, question: str) -> Finding:
        prompt = f"Overall research goal: {query}\n\nYour sub-question: {question}"
        config = {"recursion_limit": 12, "run_name": f"researcher: {question[:60]}"}
        try:
            result = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config)
        except Exception as exc:
            return Finding(question=question, notes=f"Research failed: {exc}", sources=[], tools_used=[])

        messages = result["messages"]
        notes = str(getattr(messages[-1], "content", "") or "").strip()
        tools_used = sorted({m.name for m in messages if getattr(m, "type", "") == "tool" and m.name})
        sources = extract_links(render_transcript(messages, max_chars=20000))
        return Finding(question=question, notes=notes, sources=sources[:15], tools_used=tools_used)

    return research


def format_findings(findings: list[Finding]) -> str:
    blocks = []
    for index, finding in enumerate(findings, 1):
        sources = "\n".join(f"- {url}" for url in finding["sources"]) or "- none"
        blocks.append(
            f"### Finding {index}: {finding['question']}\n{finding['notes'][:MAX_NOTES_CHARS]}\n\nSources:\n{sources}"
        )
    return "\n\n".join(blocks)


def write_report(
    json_llm: JsonLLM,
    query: str,
    findings: list[Finding],
    draft: dict | None = None,
    critique: dict | None = None,
) -> dict:
    """Writer agent: synthesize findings into a ResearchResponse-shaped dict, revising when given a critique."""
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    revision_notes = ""
    if draft and critique:
        issues = "\n".join(f"- {issue}" for issue in critique.get("issues", []))
        revision_notes = f"""
Your previous draft:
{json.dumps(draft, ensure_ascii=False)[:8000]}

Reviewer issues to fix:
{issues or "- none listed"}
"""
    system_prompt = f"""
You are the lead writer of a research team. Synthesize the researchers' findings into one report.
- Use only information present in the findings. Do not invent citations.
- detailed_report must be one substantial markdown string with sections for overview, background,
  key evidence, analysis, limitations, practical implications, and next steps when relevant.
- Cite source URLs inline in detailed_report. Put every URL you cite in source_links.
- Separate established facts from uncertainty, and set confidence to low, medium, or high.

Return only JSON matching this schema:
{parser.get_format_instructions()}
"""
    user_prompt = f"Research request:\n{query}\n\nFindings:\n{format_findings(findings)}\n{revision_notes}"
    try:
        return json_llm(system_prompt, user_prompt)
    except Exception as exc:
        return draft or {"topic": query, "summary": f"Writer failed: {exc}"}


def review_report(json_llm: JsonLLM, query: str, findings: list[Finding], draft: dict) -> dict:
    """Critic agent: score the draft against the findings. Fails open (approves) if the model call fails."""
    system_prompt = f"""
You are a strict research reviewer. Check the draft report against the findings for:
unsupported claims, missing citations, gaps in coverage of the request, and contradictions.
Approve when the report is accurate, well-sourced, and answers the request (score >= 7).
Only list follow_up_questions when important evidence is missing and new research would fix it
(at most {MAX_GAP_QUESTIONS}). Return only JSON:
{{"approved": true, "score": 8, "issues": ["..."], "follow_up_questions": ["..."]}}
"""
    user_prompt = f"""Research request:
{query}

Findings:
{format_findings(findings)}

Draft report:
{json.dumps(draft, ensure_ascii=False)[:12000]}
"""
    try:
        return json_llm(system_prompt, user_prompt)
    except Exception:
        return {"approved": True, "issues": [], "follow_up_questions": []}


def finalize_response(query: str, draft: dict, findings: list[Finding]) -> ResearchResponse:
    """Validate the draft with Pydantic and merge in links and tools taken from the researchers' transcripts."""
    transcript = "\n".join(url for finding in findings for url in finding["sources"])
    response = coerce_research_response(draft, query, transcript)
    for finding in findings:
        for tool in finding["tools_used"]:
            if tool not in response.tools_used:
                response.tools_used.append(tool)
    return response


def build_research_graph(json_llm: JsonLLM, research: ResearchFn):
    def planner(state: ResearchState) -> dict:
        system_prompt = f"""
You are the lead planner of a research team. Break the user's research request into
2-{MAX_SUB_QUESTIONS} focused, non-overlapping sub-questions that together cover it well.
Return only JSON: {{"sub_questions": ["...", "..."]}}
"""
        try:
            data = json_llm(system_prompt, state["query"])
            questions = [str(q).strip() for q in data.get("sub_questions", []) if str(q).strip()]
        except Exception:
            questions = []
        return {
            "sub_questions": questions[:MAX_SUB_QUESTIONS] or [state["query"]],
            "revision": 0,
            "research_rounds": 1,
        }

    def dispatch_research(state: ResearchState) -> list[Send]:
        return [Send("researcher", {"query": state["query"], "question": q}) for q in state["sub_questions"]]

    def researcher(state: ResearcherInput) -> dict:
        return {"findings": [research(state["query"], state["question"])]}

    def writer(state: ResearchState) -> dict:
        draft = write_report(
            json_llm, state["query"], state.get("findings", []), state.get("draft"), state.get("critique")
        )
        return {"draft": draft}

    def critic(state: ResearchState) -> dict:
        critique = review_report(json_llm, state["query"], state.get("findings", []), state.get("draft", {}))
        return {"critique": critique, "revision": state.get("revision", 0) + 1}

    def route_after_critic(state: ResearchState):
        critique = state.get("critique", {})
        if critique.get("approved") or state.get("revision", 0) > MAX_REVISIONS:
            return "finalize"
        gaps = [str(q).strip() for q in critique.get("follow_up_questions", []) if str(q).strip()]
        if gaps and state.get("research_rounds", 1) < MAX_RESEARCH_ROUNDS:
            return [Send("gap_researcher", {"query": state["query"], "question": q}) for q in gaps[:MAX_GAP_QUESTIONS]]
        return "writer"

    def gap_researcher(state: ResearcherInput) -> dict:
        return {"findings": [research(state["query"], state["question"])]}

    def count_round(state: ResearchState) -> dict:
        return {"research_rounds": state.get("research_rounds", 1) + 1}

    def finalize(state: ResearchState) -> dict:
        return {"final": finalize_response(state["query"], state.get("draft", {}), state.get("findings", []))}

    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner)
    graph.add_node("researcher", researcher)
    graph.add_node("gap_researcher", gap_researcher)
    graph.add_node("count_round", count_round)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", dispatch_research, ["researcher"])
    graph.add_edge("researcher", "writer")
    graph.add_edge("gap_researcher", "count_round")
    graph.add_edge("count_round", "writer")
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges("critic", route_after_critic, ["finalize", "writer", "gap_researcher"])
    graph.add_edge("finalize", END)
    return graph.compile()


ProgressFn = Callable[[str, dict], None]


def langsmith_enabled() -> bool:
    tracing = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or ""
    has_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    return tracing.lower() == "true" and bool(has_key)


def run_research_graph(
    graph,
    query: str,
    on_progress: ProgressFn | None = None,
    metadata: dict | None = None,
    run_name: str = "multi-agent-research",
    tags: list[str] | None = None,
) -> ResearchResponse:
    final: ResearchResponse | None = None
    # run_name, tags and metadata make each run easy to find and filter in LangSmith.
    config = {
        "recursion_limit": 40,
        "max_concurrency": 3,
        "run_name": run_name,
        "tags": tags or ["multi-agent", "research"],
        "metadata": {"query": query, **(metadata or {})},
    }
    for chunk in graph.stream({"query": query, "findings": []}, config, stream_mode="updates"):
        for node, update in chunk.items():
            if on_progress and update is not None:
                on_progress(node, update)
            if node == "finalize":
                final = update["final"]
    if final is None:
        raise RuntimeError("Multi-agent research finished without a final report.")
    return final


def perform_multi_agent_research(
    provider: str,
    api_key: str,
    model_name: str,
    query: str,
    on_progress: ProgressFn | None = None,
) -> ResearchResponse:
    graph = build_research_graph(
        make_json_llm(provider, api_key, model_name),
        make_researcher(provider, api_key, model_name),
    )
    metadata = {"provider": provider, "model": model_name, "mode": "multi-agent"}
    return run_research_graph(graph, query, on_progress, metadata)


def describe_progress(node: str, update: dict) -> str:
    """Human-readable one-liner for a graph step, shared by the CLI and web UI."""
    if node == "supervisor":
        decision = update.get("decision", {})
        questions = decision.get("questions") or []
        detail = f" ({len(questions)} questions: {'; '.join(questions)})" if questions else ""
        override = " [guardrail override]" if decision.get("overridden") else ""
        return f"Supervisor -> {decision.get('next', '?')}{detail}: {decision.get('reason', '')}{override}"
    if node == "planner":
        questions = update.get("sub_questions", [])
        return f"Planner split the request into {len(questions)} sub-questions: " + "; ".join(questions)
    if node in ("researcher", "gap_researcher"):
        finding = update["findings"][0]
        label = "Gap researcher" if node == "gap_researcher" else "Researcher"
        return f"{label} finished '{finding['question']}' ({len(finding['sources'])} links)"
    if node == "count_round":
        return "Gap research round complete, revising the report"
    if node == "writer":
        return "Writer drafted the report"
    if node == "critic":
        critique = update.get("critique", {})
        verdict = "approved" if critique.get("approved") else "requested changes"
        return f"Critic {verdict} (score {critique.get('score', '?')})"
    if node == "finalize":
        return "Report finalized"
    return node
