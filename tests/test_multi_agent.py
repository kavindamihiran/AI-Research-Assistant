import unittest

from multi_agent import Finding, build_research_graph, run_research_graph


def fake_research(query: str, question: str) -> Finding:
    slug = question.lower().replace(" ", "-")
    return Finding(
        question=question,
        notes=f"Notes on {question} (https://example.com/{slug})",
        sources=[f"https://example.com/{slug}"],
        tools_used=["search"],
    )


class ScriptedLLM:
    """Routes each prompt to a scripted reply based on which agent is calling."""

    def __init__(self, critiques: list[dict]):
        self.critiques = list(critiques)
        self.calls: list[str] = []

    def __call__(self, system_prompt: str, user_prompt: str) -> dict:
        if "lead planner" in system_prompt:
            self.calls.append("planner")
            return {"sub_questions": ["history of X", "current state of X"]}
        if "lead writer" in system_prompt:
            self.calls.append("writer")
            return {
                "topic": "X",
                "summary": "Summary of X",
                "detailed_report": "## Overview\nX (https://example.com/history-of-x)",
                "key_findings": ["finding"],
                "sources": ["Example"],
                "source_links": [],
                "confidence": "medium",
            }
        if "research reviewer" in system_prompt:
            self.calls.append("critic")
            return self.critiques.pop(0)
        raise AssertionError(f"Unexpected prompt: {system_prompt[:80]}")


class MultiAgentGraphTest(unittest.TestCase):
    def run_graph(self, critiques: list[dict]):
        llm = ScriptedLLM(critiques)
        events: list[str] = []
        graph = build_research_graph(llm, fake_research)
        result = run_research_graph(graph, "Research X", lambda node, update: events.append(node))
        return result, llm, events

    def test_approved_on_first_pass(self):
        result, llm, events = self.run_graph([{"approved": True, "score": 9}])

        self.assertEqual(llm.calls, ["planner", "writer", "critic"])
        self.assertEqual(events.count("researcher"), 2)
        self.assertEqual(result.topic, "X")
        self.assertIn("search", result.tools_used)
        self.assertIn("https://example.com/history-of-x", result.source_links)
        self.assertIn("https://example.com/current-state-of-x", result.source_links)

    def test_critic_requests_revision(self):
        _, llm, _ = self.run_graph(
            [
                {"approved": False, "issues": ["missing citation"], "follow_up_questions": []},
                {"approved": True},
            ]
        )
        self.assertEqual(llm.calls, ["planner", "writer", "critic", "writer", "critic"])

    def test_critic_dispatches_gap_research_once(self):
        gap = {"approved": False, "issues": ["thin"], "follow_up_questions": ["future of X"]}
        result, llm, events = self.run_graph([gap, gap, {"approved": True}])

        self.assertEqual(events.count("gap_researcher"), 1)
        self.assertEqual(llm.calls, ["planner", "writer", "critic", "writer", "critic", "writer", "critic"])
        self.assertIn("https://example.com/future-of-x", result.source_links)

    def test_revision_loop_is_bounded(self):
        reject = {"approved": False, "issues": ["bad"], "follow_up_questions": []}
        result, llm, events = self.run_graph([reject] * 10)

        self.assertEqual(llm.calls.count("critic"), 3)
        self.assertEqual(events[-1], "finalize")
        self.assertEqual(result.summary, "Summary of X")


if __name__ == "__main__":
    unittest.main()
