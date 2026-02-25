"""
test_phase2.py
────────────────────────────────────────────────────────────
Unit tests for Phase 2: BaseAgent and Sub-Agents.

All Groq API calls are mocked – no network access required.
Run with:  python test_phase2.py
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Groq import: use real SDK or offline mock ────────────────
try:
    import groq
except ImportError:
    import mock_groq as groq
    import sys; sys.modules["groq"] = groq


import groq

from core.key_manager import KeyManager
from agents.base_agent import BaseAgent
from agents.sub_agents import (
    ReasoningAgent, CodeAgent,
    KnowledgeAgent, AutomationAgent,
    HEAVY_MODEL, FAST_MODEL,
)


# ══════════════════════════════════════════════════════════════
#  Shared fixtures
# ══════════════════════════════════════════════════════════════

VALID_PLAN = {
    "agent": "Code",
    "action_type": "script",
    "parameters": {"script_type": "bash", "commands": ["echo hello"]},
    "requires_root": False,
    "confidence_score": 0.9,
    "risk_level": "low",
    "dry_run_safe": True,
    "payload_hash": "abc123",
}

VALID_PLAN_JSON = json.dumps(VALID_PLAN)


def make_mock_km() -> KeyManager:
    """Return a KeyManager with two fake keys (no disk I/O)."""
    km = MagicMock(spec=KeyManager)
    km.get_current_key.return_value = "gsk_fake_key_1"
    km.rotate_key.return_value      = "gsk_fake_key_2"
    km.total_keys.return_value      = 2
    km.make_client.return_value     = MagicMock()   # fake groq.Groq instance
    return km


def make_mock_response(content: str) -> MagicMock:
    """Build a fake Groq API response object."""
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = content
    return mock_resp


# ══════════════════════════════════════════════════════════════
#  BaseAgent Tests
# ══════════════════════════════════════════════════════════════

class TestBaseAgent(unittest.TestCase):

    def setUp(self):
        self.km    = make_mock_km()
        self.agent = BaseAgent(
            model_name=FAST_MODEL,
            key_manager=self.km,
        )
        # Replace the internal client with a mock
        self.mock_client = MagicMock()
        self.agent._client = self.mock_client

    # ── Happy path ────────────────────────────────────────────

    def test_successful_json_response(self):
        """Model returns clean JSON → parsed dict returned."""
        self.mock_client.chat.completions.create.return_value = \
            make_mock_response(VALID_PLAN_JSON)

        result = self.agent.generate(
            prompt="Write a test script",
            system_instruction="You are a code agent.",
        )
        self.assertIsInstance(result, dict)
        self.assertEqual(result["agent"], "Code")
        self.assertEqual(result["action_type"], "script")

    def test_strips_markdown_fences(self):
        """Model wraps JSON in ```json fences → still parsed correctly."""
        fenced = f"```json\n{VALID_PLAN_JSON}\n```"
        self.mock_client.chat.completions.create.return_value = \
            make_mock_response(fenced)

        result = self.agent.generate(
            prompt="task", system_instruction="sys"
        )
        self.assertEqual(result["agent"], "Code")

    # ── Retry on RateLimitError ───────────────────────────────

    def test_rotate_key_on_rate_limit(self):
        """
        First call → RateLimitError.
        Second call (after key rotation) → success.
        """
        success_response = make_mock_response(VALID_PLAN_JSON)

        self.mock_client.chat.completions.create.side_effect = [
            groq.RateLimitError("429", response=MagicMock(), body={}),
            success_response,
        ]

        # After rotate_key(), make_client() returns same mock_client
        self.km.make_client.return_value = self.mock_client

        result = self.agent.generate(
            prompt="task", system_instruction="sys"
        )
        self.km.rotate_key.assert_called_once()
        self.assertEqual(result["agent"], "Code")

    def test_exhausted_keys_raises(self):
        """All retries fail with RateLimitError → RuntimeError raised."""
        self.km.rotate_key.return_value = None   # no more keys
        self.mock_client.chat.completions.create.side_effect = \
            groq.RateLimitError("429", response=MagicMock(), body={})

        with self.assertRaises(RuntimeError):
            self.agent.generate(
                prompt="task", system_instruction="sys"
            )

    # ── JSON parse error ──────────────────────────────────────

    def test_invalid_json_raises(self):
        """Model returns non-JSON → RuntimeError raised after retries."""
        self.mock_client.chat.completions.create.return_value = \
            make_mock_response("This is not JSON at all.")

        with self.assertRaises(RuntimeError):
            self.agent.generate(prompt="task", system_instruction="sys")


# ══════════════════════════════════════════════════════════════
#  Sub-Agent Model Routing Tests
# ══════════════════════════════════════════════════════════════

class TestSubAgentModels(unittest.TestCase):

    def setUp(self):
        self.km = make_mock_km()

    def test_reasoning_agent_uses_heavy_model(self):
        agent = ReasoningAgent(self.km)
        self.assertEqual(agent.model_name, HEAVY_MODEL)

    def test_code_agent_uses_heavy_model(self):
        agent = CodeAgent(self.km)
        self.assertEqual(agent.model_name, HEAVY_MODEL)

    def test_knowledge_agent_uses_fast_model(self):
        agent = KnowledgeAgent(self.km)
        self.assertEqual(agent.model_name, FAST_MODEL)

    def test_automation_agent_uses_fast_model(self):
        agent = AutomationAgent(self.km)
        self.assertEqual(agent.model_name, FAST_MODEL)


# ══════════════════════════════════════════════════════════════
#  Sub-Agent run() Integration Tests (mocked API)
# ══════════════════════════════════════════════════════════════

class TestSubAgentRun(unittest.TestCase):

    def _make_agent_with_mock(self, agent_class):
        km = make_mock_km()
        agent = agent_class(km)
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = \
            make_mock_response(VALID_PLAN_JSON)
        agent._client = mock_client
        return agent, mock_client

    def test_reasoning_agent_run_returns_dict(self):
        agent, _ = self._make_agent_with_mock(ReasoningAgent)
        result = agent.run("Analyse this problem step by step.")
        self.assertIsInstance(result, dict)

    def test_code_agent_run_returns_dict(self):
        agent, _ = self._make_agent_with_mock(CodeAgent)
        result = agent.run("Write a script to list all files.")
        self.assertIsInstance(result, dict)

    def test_knowledge_agent_run_returns_dict(self):
        agent, _ = self._make_agent_with_mock(KnowledgeAgent)
        result = agent.run("What is the capital of France?")
        self.assertIsInstance(result, dict)

    def test_automation_agent_run_returns_dict(self):
        agent, _ = self._make_agent_with_mock(AutomationAgent)
        result = agent.run("Move all .txt files to /sdcard/docs/")
        self.assertIsInstance(result, dict)

    def test_heavy_model_uses_reasoning_effort(self):
        """Verify reasoning_effort is passed for heavy models."""
        agent, mock_client = self._make_agent_with_mock(ReasoningAgent)
        agent.run("Complex reasoning task.")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertIn("reasoning_effort", call_kwargs)
        self.assertEqual(call_kwargs["reasoning_effort"], "high")

    def test_fast_model_no_reasoning_effort(self):
        """Fast models must NOT include reasoning_effort."""
        agent, mock_client = self._make_agent_with_mock(KnowledgeAgent)
        agent.run("Simple question.")
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        self.assertNotIn("reasoning_effort", call_kwargs)


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
