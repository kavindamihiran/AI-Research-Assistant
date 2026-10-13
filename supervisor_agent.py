"""Centralized (supervisor) multi-agent research team built on LangGraph.

A supervisor LLM is the hub: after every step it decides which worker runs next
(research, write, review, or finish) and every worker reports back to it.
Code-level guardrails validate each decision and enforce hard budgets, so the
supervisor can steer the team but cannot loop forever or skip required steps.
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from core import ResearchResponse
from multi_agent import (
    Finding,
    JsonLLM,
    ProgressFn,
    ResearcherInput,
    ResearchFn,
    finalize_response,
    make_json_llm,
    make_researcher,
    review_report,
    run_research_graph,
    write_report,
)


MAX_SUPERVISOR_STEPS = 8
MAX_RESEARCHERS = 6
MAX_RESEARCHERS_PER_DISPATCH = 3
MAX_DRAFTS = 3

CHOICES = ("research", "write", "review", "finish")


class SupervisorState(TypedDict, total=False):
    query: str
    findings: Annotated[list[Finding], operator.add]
    draft: dict
    critique: dict
    decision: dict
    decisions: Annotated[list[dict], operator.add]
    steps: int
    researchers_used: int
    drafts: int
    reviewed_drafts: int
    final: ResearchResponse


def summarize_state(state: SupervisorState) -> str:
    """Compact status report the supervisor reads before each decision."""
    findings = state.get("findings", [])
    drafts = state.get("drafts", 0)
    lines = [
        f"Steps used: {state.get('steps', 0)}/{MAX_SUPERVISOR_STEPS}",
        f"Researchers used: {state.get('researchers_used', 0)}/{MAX_RESEARCHERS} "
        f"(max {MAX_RESEARCHERS_PER_DISPATCH} per dispatch)",
        f"Drafts written: {drafts}/{MAX_DRAFTS}",
        "Findings so far:",
    ]
    lines += [f"- {f['question']} ({len(f['sources'])} links)" for f in findings] or ["- none"]
    if drafts:
        reviewed = state.get("reviewed_drafts", 0) == drafts
        lines.append(f"Current draft: draft {drafts}, reviewed: {'yes' if reviewed else 'no'}")
    else:
        lines.append("Current draft: none")
    critique = state.get("critique")
    if critique:
        lines.append(
            f"Latest review (of draft {state.get('reviewed_drafts', 0)}): "
            f"approved={critique.get('approved')}, score={critique.get('score', '?')}"
        )
        lines += [f"  issue: {issue}" for issue in critique.get("issues", [])]
        lines += [f"  missing evidence: {q}" for q in critique.get("follow_up_questions", [])]
    return "\n".join(lines)


def fallback_choice(state: SupervisorState) -> str:
    """Deterministic policy used when the supervisor's choice is invalid or unavailable."""
    drafts = state.get("drafts", 0)
    if not state.get("findings") and state.get("researchers_used", 0) < MAX_RESEARCHERS:
        return "research"
    if not drafts:
        return "write"
    if state.get("reviewed_drafts", 0) < drafts:
        return "review"
    return "finish"


def is_allowed(state: SupervisorState, choice: str, questions: list[str]) -> bool:
    drafts = state.get("drafts", 0)
    if choice == "research":
        return bool(questions)
    if choice == "write":
        return bool(state.get("findings")) and drafts < MAX_DRAFTS
    if choice == "review":
        return drafts > 0 and state.get("reviewed_drafts", 0) < drafts
    if choice == "finish":
        return drafts > 0
    return False


def resolve_decision(state: SupervisorState, proposal: dict) -> dict:
    """Apply guardrails to the supervisor's proposal and return the decision that will actually run."""
    query = state["query"]
    reason = str(proposal.get("reason", "")).strip()[:300]

    if state.get("steps", 0) >= MAX_SUPERVISOR_STEPS:
        choice = "finish" if state.get("drafts", 0) else "write"
        return {"next": choice, "questions": [], "reason": "step budget exhausted", "overridden": True}

    choice = str(proposal.get("next", "")).strip().lower()
    budget = min(MAX_RESEARCHERS_PER_DISPATCH, MAX_RESEARCHERS - state.get("researchers_used", 0))
    seen = {f["question"].strip().lower() for f in state.get("findings", [])}
    questions = []
    for item in proposal.get("questions") or []:
        question = str(item).strip()
        if question and question.lower() not in seen and question not in questions:
            questions.append(question)
    questions = questions[: max(budget, 0)]

    if is_allowed(state, choice, questions):
        return {"next": choice, "questions": questions if choice == "research" else [], "reason": reason, "overridden": False}

    fallback = fallback_choice(state)
    fallback_questions = [query] if fallback == "research" else []
    note = f"'{choice or 'nothing'}' not allowed now" if proposal else "supervisor unavailable"
    return {
        "next": fallback,
        "questions": fallback_questions,
        "reason": f"{note}; fallback policy chose {fallback}",
        "overridden": True,
    }


SUPERVISOR_PROMPT = f"""
You are the supervisor of a research team. You decide which worker acts next. Workers:
- research: researchers investigate 1-{MAX_RESEARCHERS_PER_DISPATCH} focused questions in parallel with web search and Wikipedia
- write: the writer drafts or revises the report from all findings
- review: the critic checks the current draft for unsupported claims, missing citations and gaps
- finish: publish the current draft

How to decide:
- Start by researching 2-3 focused, non-overlapping questions that together cover the request.
- Write once the findings cover the request. Always review a draft before finishing.
- If the critic reports missing evidence, research those gaps. If it reports writing problems, send it back to write.
- Finish when the critic approves, or when budgets are nearly used up.
- Never repeat a question that already has findings.

Return only JSON: {{"next": "research|write|review|finish", "questions": ["..."], "reason": "one short sentence"}}
"""


def build_supervisor_graph(json_llm: JsonLLM, research: ResearchFn):
    def supervisor(state: SupervisorState) -> dict:
        proposal: dict = {}
        if state.get("steps", 0) < MAX_SUPERVISOR_STEPS:
            user_prompt = f"Research request:\n{state['query']}\n\nTeam status:\n{summarize_state(state)}"
            try:
                proposal = json_llm(SUPERVISOR_PROMPT, user_prompt)
            except Exception:
                proposal = {}
        decision = resolve_decision(state, proposal)
        update = {"decision": decision, "decisions": [decision], "steps": state.get("steps", 0) + 1}
        if decision["next"] == "research":
            update["researchers_used"] = state.get("researchers_used", 0) + len(decision["questions"])
        return update

    def route_from_supervisor(state: SupervisorState):
        decision = state["decision"]
        if decision["next"] == "research":
            return [Send("researcher", {"query": state["query"], "question": q}) for q in decision["questions"]]
        return {"write": "writer", "review": "critic", "finish": "finalize"}[decision["next"]]

    def researcher(state: ResearcherInput) -> dict:
        return {"findings": [research(state["query"], state["question"])]}

    def writer(state: SupervisorState) -> dict:
        draft = write_report(
            json_llm, state["query"], state.get("findings", []), state.get("draft"), state.get("critique")
        )
        return {"draft": draft, "drafts": state.get("drafts", 0) + 1}

    def critic(state: SupervisorState) -> dict:
        critique = review_report(json_llm, state["query"], state.get("findings", []), state.get("draft", {}))
        return {"critique": critique, "reviewed_drafts": state.get("drafts", 0)}

    def finalize(state: SupervisorState) -> dict:
        return {"final": finalize_response(state["query"], state.get("draft", {}), state.get("findings", []))}

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, ["researcher", "writer", "critic", "finalize"])
    for worker in ("researcher", "writer", "critic"):
        graph.add_edge(worker, "supervisor")
    graph.add_edge("finalize", END)
    return graph.compile()


def perform_supervisor_research(
    provider: str,
    api_key: str,
    model_name: str,
    query: str,
    on_progress: ProgressFn | None = None,
) -> ResearchResponse:
    graph = build_supervisor_graph(
        make_json_llm(provider, api_key, model_name),
        make_researcher(provider, api_key, model_name),
    )
    metadata = {"provider": provider, "model": model_name, "mode": "supervisor"}
    return run_research_graph(
        graph, query, on_progress, metadata, run_name="supervisor-research", tags=["supervisor", "research"]
    )

