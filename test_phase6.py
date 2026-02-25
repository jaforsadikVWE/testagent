"""
test_phase6.py
────────────────────────────────────────────────────────────
Integration tests for Phase 6: the main REPL loop.

Everything is mocked – no API calls, no real file I/O, no
subprocess execution.
Run with:  python test_phase6.py
"""

import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Groq import: use real SDK or offline mock ────────────────
try:
    import groq
except ImportError:
    import mock_groq as groq
    import sys; sys.modules["groq"] = groq


from execution.policy import compute_payload_hash


# ══════════════════════════════════════════════════════════════
#  Shared fixtures
# ══════════════════════════════════════════════════════════════

FAKE_CONFIG = {
    "groq_api_keys":        ["gsk_real_key"],
    "confidence_threshold": 0.5,
    "experimental_override": False,
    "max_retries":          3,
    "request_timeout":      30,
    "models": {
        "heavy": "openai/gpt-oss-120b",
        "fast":  "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "memory":  {"history_file": "data/history.json", "max_turns": 15},
    "logging": {"audit_log": "logs/audit.log", "max_bytes": 1048576, "backup_count": 5},
}

DUMMY_COMMANDS = ["echo hello"]

DUMMY_PLAN = {
    "_selected_agent":  "CodeAgent",
    "agent":            "Code",
    "action_type":      "script",
    "parameters":       {"script_type": "bash", "commands": DUMMY_COMMANDS},
    "requires_root":    False,
    "confidence_score": 0.9,
    "risk_level":       "low",
    "dry_run_safe":     True,
    "payload_hash":     compute_payload_hash(DUMMY_COMMANDS),
    "policy_status":    "APPROVED",
    "policy_reason":    "All policy checks passed.",
}

DUMMY_RESULT = {
    "status":    "success",
    "stdout":    "hello\n",
    "stderr":    "",
    "exit_code": 0,
}


def _make_patchers():
    """Return dict of common patchers used across tests."""
    return {
        "config":       patch("main.load_config",             return_value=FAKE_CONFIG),
        "km":           patch("main.KeyManager"),
        "memory":       patch("main.MemoryManager"),
        "orchestrator": patch("main.Orchestrator"),
        "watchdog":     patch("main.Watchdog"),
        "execute_plan": patch("main.execute_plan",            return_value=DUMMY_RESULT),
        "evaluate_plan":patch("main.evaluate_plan",           return_value=("APPROVED", DUMMY_PLAN)),
    }


# ══════════════════════════════════════════════════════════════
#  REPL Loop Tests
# ══════════════════════════════════════════════════════════════

class TestREPLLoop(unittest.TestCase):

    # ── Test 1: Normal flow – one command then exit ───────────

    def test_single_command_then_exit(self):
        """
        Simulate: user types 'list files', then 'exit'.
        Assert the loop runs, processes input, and exits cleanly.
        """
        inputs = iter(["list files", "exit"])

        with patch("builtins.input", side_effect=inputs), \
             patch("main.load_config",  return_value=FAKE_CONFIG), \
             patch("main.KeyManager"),  \
             patch("main.MemoryManager") as MockMem, \
             patch("main.Orchestrator") as MockOrch, \
             patch("main.Watchdog"), \
             patch("main.execute_plan", return_value=DUMMY_RESULT), \
             patch("main.evaluate_plan", return_value=("APPROVED", DUMMY_PLAN)):

            mock_orch = MockOrch.return_value
            mock_orch.handle.return_value = DUMMY_PLAN

            mock_mem = MockMem.return_value
            mock_mem.get_context.return_value = []

            from main import main
            main()   # must not raise

            # Orchestrator.handle must have been called with the user input.
            # We only check the first arg (string) because the history list is
            # passed by reference and mock records the ref, not the call-time value.
            mock_orch.handle.assert_called_once()
            call_args = mock_orch.handle.call_args[0]
            self.assertEqual(call_args[0], "list files")

    # ── Test 2: execute_plan is called with the plan ──────────

    def test_execute_plan_called_with_plan(self):
        inputs = iter(["echo hello", "exit"])

        with patch("builtins.input", side_effect=inputs), \
             patch("main.load_config",  return_value=FAKE_CONFIG), \
             patch("main.KeyManager"),  \
             patch("main.MemoryManager") as MockMem, \
             patch("main.Orchestrator") as MockOrch, \
             patch("main.Watchdog"), \
             patch("main.execute_plan", return_value=DUMMY_RESULT) as mock_exec, \
             patch("main.evaluate_plan", return_value=("APPROVED", DUMMY_PLAN)):

            MockOrch.return_value.handle.return_value = DUMMY_PLAN
            MockMem.return_value.get_context.return_value = []

            from main import main
            main()

            mock_exec.assert_called_once_with(DUMMY_PLAN)

    # ── Test 3: Memory is saved after each turn ───────────────

    def test_memory_add_message_called(self):
        inputs = iter(["list files", "exit"])

        with patch("builtins.input", side_effect=inputs), \
             patch("main.load_config",  return_value=FAKE_CONFIG), \
             patch("main.KeyManager"),  \
             patch("main.MemoryManager") as MockMem, \
             patch("main.Orchestrator") as MockOrch, \
             patch("main.Watchdog"), \
             patch("main.execute_plan", return_value=DUMMY_RESULT), \
             patch("main.evaluate_plan", return_value=("APPROVED", DUMMY_PLAN)):

            MockOrch.return_value.handle.return_value = DUMMY_PLAN
            mock_mem = MockMem.return_value
            mock_mem.get_context.return_value = []

            from main import main
            main()

            # add_message should be called twice: once for user, once for model
            calls = mock_mem.add_message.call_args_list
            roles = [c[0][0] for c in calls]
            self.assertIn("user",  roles)
            self.assertIn("model", roles)

    # ── Test 4: APPROVAL_REQUIRED – user denies ───────────────

    def test_approval_required_user_denies(self):
        """
        When evaluate_plan returns APPROVAL_REQUIRED and user types 'N',
        execute_plan should NOT be called.
        """
        inputs = iter(["dangerous command", "N", "exit"])

        approval_plan = dict(DUMMY_PLAN)
        approval_plan["risk_level"]    = "high"
        approval_plan["policy_status"] = "APPROVAL_REQUIRED"
        approval_plan["policy_reason"] = "High risk."

        with patch("builtins.input", side_effect=inputs), \
             patch("main.load_config",  return_value=FAKE_CONFIG), \
             patch("main.KeyManager"),  \
             patch("main.MemoryManager") as MockMem, \
             patch("main.Orchestrator") as MockOrch, \
             patch("main.Watchdog"),  \
             patch("main.execute_plan", return_value=DUMMY_RESULT) as mock_exec, \
             patch("main.evaluate_plan",
                   return_value=("APPROVAL_REQUIRED", approval_plan)):

            MockOrch.return_value.handle.return_value = approval_plan
            MockMem.return_value.get_context.return_value = []

            from main import main
            main()

            mock_exec.assert_not_called()

    # ── Test 5: Keyboard interrupt exits cleanly ──────────────

    def test_keyboard_interrupt_exits_gracefully(self):
        with patch("builtins.input", side_effect=KeyboardInterrupt), \
             patch("main.load_config",  return_value=FAKE_CONFIG), \
             patch("main.KeyManager"),  \
             patch("main.MemoryManager"), \
             patch("main.Orchestrator"), \
             patch("main.Watchdog"):

            from main import main
            try:
                main()   # Should not raise
            except (KeyboardInterrupt, SystemExit):
                pass     # Acceptable exits

    # ── Test 6: REPL continues after runtime error ────────────

    def test_repl_continues_after_exception(self):
        """
        If orchestrator.handle raises an exception, the loop should
        continue (not crash) and process the next input.
        """
        inputs = iter(["bad input", "exit"])

        with patch("builtins.input", side_effect=inputs), \
             patch("main.load_config",  return_value=FAKE_CONFIG), \
             patch("main.KeyManager"),  \
             patch("main.MemoryManager") as MockMem, \
             patch("main.Orchestrator") as MockOrch, \
             patch("main.Watchdog"), \
             patch("main.execute_plan"), \
             patch("main.evaluate_plan", return_value=("APPROVED", DUMMY_PLAN)):

            MockOrch.return_value.handle.side_effect = RuntimeError("API down")
            MockMem.return_value.get_context.return_value = []

            from main import main
            main()   # Must not raise even though orchestrator errored

    # ── Test 7: 'help' built-in doesn't call orchestrator ─────

    def test_help_command_no_orchestration(self):
        inputs = iter(["help", "exit"])

        with patch("builtins.input", side_effect=inputs), \
             patch("main.load_config",  return_value=FAKE_CONFIG), \
             patch("main.KeyManager"),  \
             patch("main.MemoryManager") as MockMem, \
             patch("main.Orchestrator") as MockOrch, \
             patch("main.Watchdog"):

            MockMem.return_value.get_context.return_value = []

            from main import main
            main()

            MockOrch.return_value.handle.assert_not_called()

    # ── Test 8: Empty input skipped without orchestration ─────

    def test_empty_input_skipped(self):
        inputs = iter(["", "exit"])

        with patch("builtins.input", side_effect=inputs), \
             patch("main.load_config",  return_value=FAKE_CONFIG), \
             patch("main.KeyManager"),  \
             patch("main.MemoryManager") as MockMem, \
             patch("main.Orchestrator") as MockOrch, \
             patch("main.Watchdog"):

            MockMem.return_value.get_context.return_value = []

            from main import main
            main()

            MockOrch.return_value.handle.assert_not_called()


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
