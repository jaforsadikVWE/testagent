"""
test_phase5.py
────────────────────────────────────────────────────────────
Unit tests for Phase 5: MemoryManager and Watchdog.

Uses real file I/O in a temporary directory.
Run with:  python test_phase5.py
"""

import json
import logging
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.memory   import MemoryManager
from core.watchdog import Watchdog


# ══════════════════════════════════════════════════════════════
#  MemoryManager Tests
# ══════════════════════════════════════════════════════════════

class TestMemoryManager(unittest.TestCase):

    def setUp(self):
        """Each test gets its own temp directory so tests are isolated."""
        self._tmp = tempfile.mkdtemp()
        self._history_file = os.path.join(self._tmp, "data", "history.json")

    def _make_mm(self, max_turns=15) -> MemoryManager:
        return MemoryManager(
            history_file=self._history_file,
            max_turns=max_turns,
        )

    # ── Test 1: Persistence ───────────────────────────────────

    def test_persistence_across_instances(self):
        """
        Write a message with one MemoryManager instance,
        then verify it exists when loading a new instance.
        """
        mm1 = self._make_mm()
        mm1.add_message("user", "Hello from test!")

        # Create a completely new instance pointing to the same file
        mm2 = self._make_mm()
        context = mm2.get_context(limit=10)

        self.assertEqual(len(context), 1)
        self.assertEqual(context[0]["role"],    "user")
        self.assertEqual(context[0]["content"], "Hello from test!")

    def test_timestamp_saved(self):
        mm = self._make_mm()
        mm.add_message("user", "timestamped message")
        mm2 = self._make_mm()
        self.assertIn("timestamp", mm2.get_context()[0])

    # ── Test 2: Sliding window pruning ───────────────────────

    def test_pruning_keeps_last_n(self):
        """
        Add 20 messages with max_turns=15.
        Only the last 15 should be in the file.
        get_context(limit=5) should return only 5.
        """
        mm = self._make_mm(max_turns=15)
        for i in range(20):
            mm.add_message("user", f"message {i}")

        # File should only have 15 entries
        with open(self._history_file, "r", encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(len(saved), 15)

        # get_context with limit=5 returns last 5
        context = mm.get_context(limit=5)
        self.assertEqual(len(context), 5)
        self.assertEqual(context[-1]["content"], "message 19")
        self.assertEqual(context[0]["content"],  "message 15")

    def test_get_context_returns_all_when_fewer_than_limit(self):
        mm = self._make_mm()
        mm.add_message("user",  "a")
        mm.add_message("model", "b")
        context = mm.get_context(limit=100)
        self.assertEqual(len(context), 2)

    # ── Test 3: Unicode & emoji support ──────────────────────

    def test_emoji_and_unicode_roundtrip(self):
        mm = self._make_mm()
        msg = "مرحبا 👋 こんにちは 🤖"
        mm.add_message("user", msg)

        mm2     = self._make_mm()
        context = mm2.get_context()
        self.assertEqual(context[0]["content"], msg)

        # Verify the file itself contains the real chars (not \uXXXX escapes)
        with open(self._history_file, "r", encoding="utf-8") as f:
            raw = f.read()
        self.assertIn("مرحبا", raw)
        self.assertIn("👋",    raw)

    # ── Test 4: clear_memory ──────────────────────────────────

    def test_clear_memory_wipes_history(self):
        mm = self._make_mm()
        mm.add_message("user", "something")
        mm.clear_memory()
        self.assertEqual(len(mm), 0)
        mm2 = self._make_mm()
        self.assertEqual(len(mm2), 0)

    # ── Test 5: Missing file handled gracefully ───────────────

    def test_missing_file_returns_empty_list(self):
        mm = self._make_mm()   # file doesn't exist yet
        history = mm.load_history()
        self.assertEqual(history, [])

    # ── Test 6: Corrupted file handled gracefully ─────────────

    def test_corrupted_file_starts_fresh(self):
        os.makedirs(os.path.dirname(self._history_file), exist_ok=True)
        with open(self._history_file, "w") as f:
            f.write("NOT VALID JSON {{{{")
        mm      = self._make_mm()
        context = mm.get_context()
        self.assertEqual(context, [])


# ══════════════════════════════════════════════════════════════
#  Watchdog Tests
# ══════════════════════════════════════════════════════════════

class TestWatchdog(unittest.TestCase):

    def setUp(self):
        self._tmp      = tempfile.mkdtemp()
        self._log_file = os.path.join(self._tmp, "logs", "audit.log")

    def _make_wd(self) -> Watchdog:
        # Use fresh logger name per test to avoid handler accumulation
        wd = Watchdog(log_file=self._log_file)
        # Reset handlers so each test gets a clean logger
        log = logging.getLogger("watchdog.audit")
        log.handlers.clear()
        wd = Watchdog(log_file=self._log_file)
        return wd

    def _read_log(self) -> str:
        with open(self._log_file, "r", encoding="utf-8") as f:
            return f.read()

    # ── Test 3: Auditing ──────────────────────────────────────

    def test_log_risk_creates_file(self):
        wd = self._make_wd()
        wd.log_risk(
            action_type="script",
            risk_level="high",
            command="rm -rf /",
            outcome="DENIED",
            requires_root=True,
        )
        self.assertTrue(os.path.exists(self._log_file))

    def test_log_risk_contains_expected_fields(self):
        wd = self._make_wd()
        wd.log_risk(
            action_type="script",
            risk_level="high",
            command="rm -rf /",
            outcome="DENIED",
        )
        content = self._read_log()
        self.assertIn("RISK: HIGH",  content)
        self.assertIn("rm -rf /",    content)
        self.assertIn("DENIED",      content)
        self.assertIn("TYPE: script", content)

    def test_log_failure_written(self):
        wd = self._make_wd()
        wd.log_failure("Executor", "subprocess.TimeoutExpired")
        content = self._read_log()
        self.assertIn("FAILURE",               content)
        self.assertIn("subprocess.TimeoutExpired", content)

    def test_low_risk_still_logged(self):
        wd = self._make_wd()
        wd.log_risk(
            action_type="read_only",
            risk_level="low",
            command="ls /sdcard",
            outcome="APPROVED",
        )
        content = self._read_log()
        self.assertIn("RISK: LOW", content)
        self.assertIn("APPROVED",  content)

    def test_root_flag_logged(self):
        wd = self._make_wd()
        wd.log_risk(
            action_type="script",
            risk_level="medium",
            command="chmod 777 /data",
            outcome="APPROVED",
            requires_root=True,
        )
        content = self._read_log()
        self.assertIn("ROOT: YES", content)

    def test_logs_directory_created_automatically(self):
        """Watchdog creates the logs/ directory if it doesn't exist."""
        self.assertFalse(os.path.exists(os.path.dirname(self._log_file)))
        wd = self._make_wd()
        wd.log_info("startup")
        self.assertTrue(os.path.exists(os.path.dirname(self._log_file)))


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
