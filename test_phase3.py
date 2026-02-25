"""
test_phase3.py
────────────────────────────────────────────────────────────
Unit tests for Phase 3: Orchestrator routing and dispatch.

All API calls are mocked – no network access required.
Run with:  python test_phase3.py
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Groq import: use real SDK or offline mock ────────────────
try:
    import groq
except ImportError:
    import mock_groq as groq
    import sys; sys.modules["groq"] = groq


import groq

from core.key_manager import KeyManager
from core.orchestrator import Orchestrator, FALLBACK_AGENT
from agents.base_agent import BaseAgent


# ══════════════════════════════════════════════════════════════
#  Fixtures
# ══════════════════════════════════════════════════════════════

def make_mock_km() -> KeyManager:
    km = MagicMock(spec=KeyManager)
    km.get_current_key.return_value = "gsk_fake"
    km.rotate_key.return_value      = "gsk_fake2"
    km.total_keys.return_value      = 2
    km.make_client.return_value     = MagicMock()
    return km


SAMPLE_PLAN = {
    "agent": "Code",
    "action_type": "script",
    "parameters": {"script_type": "bash", "commands": ["echo hello"]},
    "requires_root": False,
    "confidence_score": 0.9,
    "risk_level": "low",
    "dry_run_safe": True,
    "payload_hash": "abc123",
}


# ══════════════════════════════════════════════════════════════
#  Orchestrator Tests
# ══════════════════════════════════════════════════════════════

class TestOrchestrator(unittest.TestCase):

    def _make_orch(self) -> Orchestrator:
        km = make_mock_km()
        orch = Orchestrator(key_manager=km, max_retries=3)
        return orch

    # ── Test 1: Routing to AutomationAgent ───────────────────

    def test_routes_to_automation_agent(self):
        """Router returns AutomationAgent → Orchestrator dispatches to it."""
        orch = self._make_orch()

        router_response = {
            "selected_agent":    "AutomationAgent",
            "synthesized_context": "Delete /sdcard/test.txt",
        }
        agent_response = dict(SAMPLE_PLAN)

        # Mock the router's generate() call
        orch._router.generate = MagicMock(return_value=router_response)
        # Mock AutomationAgent's run() call
        orch._agents["AutomationAgent"].run = MagicMock(return_value=agent_response)

        plan = orch.handle("Delete that file", history=[])

        orch._agents["AutomationAgent"].run.assert_called_once_with(
            "Delete /sdcard/test.txt"
        )
        self.assertEqual(plan["_selected_agent"], "AutomationAgent")

    # ── Test 2: Context synthesis ─────────────────────────────

    def test_synthesized_context_passed_not_raw_input(self):
        """Sub-agent receives the synthesised context, NOT the raw user input."""
        orch = self._make_orch()

        synthesized = "Write Python code to list all .jpg files recursively."
        router_response = {
            "selected_agent":    "CodeAgent",
            "synthesized_context": synthesized,
        }
        agent_response = dict(SAMPLE_PLAN)

        orch._router.generate  = MagicMock(return_value=router_response)
        orch._agents["CodeAgent"].run = MagicMock(return_value=agent_response)

        raw_input = "hey can you show me all my photos?"
        orch.handle(raw_input, history=[])

        # run() must be called with the synthesised context, not raw input
        orch._agents["CodeAgent"].run.assert_called_once_with(synthesized)

    # ── Test 3: Fallback on invalid agent name ────────────────

    def test_fallback_on_unknown_agent_name(self):
        """Router returns an unknown agent → fallback to KnowledgeAgent."""
        orch = self._make_orch()

        orch._router.generate = MagicMock(return_value={
            "selected_agent":    "NonExistentAgent",
            "synthesized_context": "Some task",
        })
        orch._agents[FALLBACK_AGENT].run = MagicMock(return_value=SAMPLE_PLAN)

        plan = orch.handle("Some input", history=[])

        self.assertEqual(plan["_selected_agent"], FALLBACK_AGENT)

    # ── Test 4: Fallback on router exception ──────────────────

    def test_fallback_on_router_runtime_error(self):
        """Router raises RuntimeError (all keys exhausted) → fallback."""
        orch = self._make_orch()

        orch._router.generate = MagicMock(
            side_effect=RuntimeError("All keys exhausted")
        )
        orch._agents[FALLBACK_AGENT].run = MagicMock(return_value=SAMPLE_PLAN)

        plan = orch.handle("Any input", history=[])
        self.assertEqual(plan["_selected_agent"], FALLBACK_AGENT)

    # ── Test 5: RateLimitError retry ─────────────────────────

    def test_router_retries_on_rate_limit(self):
        """Router raises RateLimitError once, then succeeds on retry."""
        orch = self._make_orch()

        success = {
            "selected_agent": "KnowledgeAgent",
            "synthesized_context": "Who is the president?",
        }
        orch._router.generate = MagicMock(side_effect=[
            groq.RateLimitError("429", response=MagicMock(), body={}),
            success,
        ])
        orch._agents["KnowledgeAgent"].run = MagicMock(return_value=SAMPLE_PLAN)

        plan = orch.handle("Who is the president?", history=[])
        self.assertEqual(plan["_selected_agent"], "KnowledgeAgent")
        self.assertEqual(orch._router.generate.call_count, 2)

    # ── Test 6: History formatting ────────────────────────────

    def test_history_formatted_compactly(self):
        """Only the last 6 turns are forwarded to the router."""
        orch = self._make_orch()

        # Build 10 history turns
        history = [
            {"role": "user" if i % 2 == 0 else "model", "content": f"msg {i}"}
            for i in range(10)
        ]

        captured_prompts = []

        def capture_generate(prompt, system_instruction, **kwargs):
            captured_prompts.append(prompt)
            return {
                "selected_agent": "KnowledgeAgent",
                "synthesized_context": "task",
            }

        orch._router.generate = capture_generate
        orch._agents["KnowledgeAgent"].run = MagicMock(return_value=SAMPLE_PLAN)

        orch.handle("new question", history=history)

        # The router prompt should NOT contain all 10 history entries
        prompt_text = captured_prompts[0]
        # Last 6 messages (msg 4–9) should be present, first 4 absent
        self.assertIn("msg 4", prompt_text)
        self.assertNotIn("msg 0", prompt_text)

    # ── Test 7: handle() annotates plan with selected agent ───

    def test_handle_annotates_selected_agent(self):
        """handle() adds _selected_agent key to the returned plan."""
        orch = self._make_orch()

        orch._router.generate = MagicMock(return_value={
            "selected_agent":    "ReasoningAgent",
            "synthesized_context": "Reason about X",
        })
        orch._agents["ReasoningAgent"].run = MagicMock(return_value=SAMPLE_PLAN)

        plan = orch.handle("Think about X", history=[])
        self.assertIn("_selected_agent", plan)
        self.assertEqual(plan["_selected_agent"], "ReasoningAgent")


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
