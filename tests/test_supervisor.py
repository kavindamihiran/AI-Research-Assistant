import unittest

from multi_agent import run_research_graph
from supervisor_agent import MAX_DRAFTS, MAX_RESEARCHERS, MAX_SUPERVISOR_STEPS, build_supervisor_graph
from tests.test_multi_agent import fake_research


class ScriptedTeam:
    """Fake LLM: the supervisor follows a script, the writer drafts, and the critic always approves."""

    def __init__(self, decisions: list[dict] | None = None, supervisor_fails: bool = False):
        self.decisions = list(decisions or [])
        self.supervisor_fails = supervisor_fails
        self.calls: list[str] = []

    def __call__(self, system_prompt: str, user_prompt: str) -> dict:
        if "supervisor of a research team" in system_prompt:
            self.calls.append("supervisor")
            if self.supervisor_fails:
                raise RuntimeError("model returned junk")
            return self.decisions.pop(0) if self.decisions else {"next": "finish"}
        if "lead writer" in system_prompt:
            self.calls.append("writer")
            return {"topic": "X", "summary": "Summary of X", "detailed_report": "## Overview\nX"}
        if "research reviewer" in system_prompt:
            self.calls.append("critic")
            return {"approved": True, "score": 8}
        raise AssertionError(f"Unexpected prompt: {system_prompt[:80]}")


def research(*questions: str) -> dict:
    return {"next": "research", "questions": list(questions), "reason": "need evidence"}


WRITE = {"next": "write", "reason": "enough evidence"}
REVIEW = {"next": "review", "reason": "check the draft"}
FINISH = {"next": "finish", "reason": "approved"}


class SupervisorGraphTest(unittest.TestCase):
    def run_team(self, team: ScriptedTeam):
        decisions: list[dict] = []
        events: list[str] = []

        def on_progress(node: str, update: dict) -> None:
            events.append(node)
            if node == "supervisor":
                decisions.append(update["decision"])

        result = run_research_graph(build_supervisor_graph(team, fake_research), "Research X", on_progress)
        return result, decisions, events

    def test_supervisor_routes_every_step(self):
        team = ScriptedTeam([research("history of X", "current state of X"), WRITE, REVIEW, FINISH])
        result, decisions, events = self.run_team(team)

        self.assertEqual([d["next"] for d in decisions], ["research", "write", "review", "finish"])
        self.assertFalse(any(d["overridden"] for d in decisions))
        self.assertEqual(events.count("researcher"), 2)
        self.assertEqual(team.calls, ["supervisor", "supervisor", "writer", "supervisor", "critic", "supervisor"])
        self.assertIn("https://example.com/history-of-x", result.source_links)
        self.assertIn("search", result.tools_used)

    def test_finish_without_draft_is_redirected_to_write(self):
        team = ScriptedTeam([research("history of X"), FINISH, REVIEW, FINISH])
        _, decisions, _ = self.run_team(team)

        self.assertEqual(decisions[1]["next"], "write")
        self.assertTrue(decisions[1]["overridden"])
        self.assertEqual(decisions[-1]["next"], "finish")

    def test_fallback_policy_when_supervisor_fails(self):
        team = ScriptedTeam(supervisor_fails=True)
        result, decisions, _ = self.run_team(team)

        self.assertEqual([d["next"] for d in decisions], ["research", "write", "review", "finish"])
        self.assertTrue(all(d["overridden"] for d in decisions))
        self.assertEqual(decisions[0]["questions"], ["Research X"])
        self.assertEqual(result.summary, "Summary of X")

    def test_research_budget_is_enforced(self):
        script = [research(f"q{n}a", f"q{n}b", f"q{n}c", f"q{n}d") for n in range(5)]
        team = ScriptedTeam(script)
        _, decisions, events = self.run_team(team)

        self.assertEqual(events.count("researcher"), MAX_RESEARCHERS)
        self.assertTrue(all(len(d["questions"]) <= 3 for d in decisions))
        self.assertEqual(events[-1], "finalize")

    def test_duplicate_questions_are_not_researched_again(self):
        team = ScriptedTeam([research("history of X"), research("History of X"), REVIEW, FINISH])
        _, decisions, events = self.run_team(team)

        self.assertEqual(events.count("researcher"), 1)
        self.assertTrue(decisions[1]["overridden"])
        self.assertEqual(decisions[1]["next"], "write")

    def test_draft_and_step_budgets_are_enforced(self):
        script = [research("a", "b", "c"), WRITE, REVIEW, research("d", "e", "f"), WRITE, REVIEW, WRITE, REVIEW]
        script += [WRITE] * 5
        team = ScriptedTeam(script)
        _, decisions, events = self.run_team(team)

        self.assertEqual(team.calls.count("writer"), MAX_DRAFTS)
        self.assertEqual(team.calls.count("supervisor"), MAX_SUPERVISOR_STEPS)
        self.assertEqual(decisions[-1]["next"], "finish")
        self.assertEqual(decisions[-1]["reason"], "step budget exhausted")
        self.assertEqual(events[-1], "finalize")


if __name__ == "__main__":
    unittest.main()
