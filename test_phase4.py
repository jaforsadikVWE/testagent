"""
test_phase4.py
────────────────────────────────────────────────────────────
Unit tests for Phase 4: RootGuard and Executor.

All subprocess calls and user input are mocked.
No real commands are executed.
Run with:  python test_phase4.py
"""

import hashlib
import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, call

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from execution.root_guard import execute_with_root, verify_hash
from execution.executor import execute_plan
from execution.policy import compute_payload_hash


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def make_plan(
    commands=None,
    action_type="script",
    requires_root=False,
    script_type="bash",
    risk_level="low",
    confidence=0.9,
):
    if commands is None:
        commands = ["echo hello"]
    return {
        "agent":          "Code",
        "action_type":    action_type,
        "parameters":     {"script_type": script_type, "commands": commands},
        "requires_root":  requires_root,
        "confidence_score": confidence,
        "risk_level":     risk_level,
        "dry_run_safe":   True,
        "payload_hash":   compute_payload_hash(commands),
    }


def make_subprocess_result(stdout="ok\n", stderr="", returncode=0):
    m = MagicMock()
    m.stdout     = stdout
    m.stderr     = stderr
    m.returncode = returncode
    return m


# ══════════════════════════════════════════════════════════════
#  RootGuard Tests
# ══════════════════════════════════════════════════════════════

class TestRootGuard(unittest.TestCase):

    # ── Hash verification ─────────────────────────────────────

    def test_verify_hash_correct(self):
        cmd = "rm -rf /tmp/test"
        h   = sha256_str(cmd)
        self.assertTrue(verify_hash(cmd, h))

    def test_verify_hash_wrong(self):
        self.assertFalse(verify_hash("echo hello", "deadbeef" * 8))

    # ── Tamper detection (hash mismatch → ABORT) ──────────────

    def test_hash_mismatch_aborts(self):
        result = execute_with_root("rm -rf /important", "wrong_hash")
        self.assertEqual(result["status"], "error")
        self.assertIn("Hash mismatch", result["error"])

    # ── User denies (N) ───────────────────────────────────────

    @patch("execution.root_guard.subprocess.run")
    @patch("builtins.input", return_value="N")
    def test_user_denial_no_su_call(self, mock_input, mock_subproc):
        cmd  = "rm -rf /sdcard/test"
        h    = sha256_str(cmd)
        result = execute_with_root(cmd, h)
        self.assertEqual(result["status"], "denied")
        mock_subproc.assert_not_called()

    # ── User approves (y) ─────────────────────────────────────

    @patch("execution.root_guard.subprocess.run",
           return_value=make_subprocess_result("done\n"))
    @patch("builtins.input", return_value="y")
    def test_user_approval_calls_su(self, mock_input, mock_subproc):
        cmd  = "chmod 777 /sdcard/myfile"
        h    = sha256_str(cmd)
        result = execute_with_root(cmd, h)
        self.assertEqual(result["status"], "success")
        # Must use ['su', '-c', command] format
        mock_subproc.assert_called_once_with(
            ["su", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=30,
        )

    # ── Root command timeout ──────────────────────────────────

    @patch("execution.root_guard.subprocess.run",
           side_effect=__import__("subprocess").TimeoutExpired("su", 30))
    @patch("builtins.input", return_value="y")
    def test_root_timeout_returns_error(self, mock_input, mock_subproc):
        cmd  = "sleep 999"
        h    = sha256_str(cmd)
        result = execute_with_root(cmd, h)
        self.assertEqual(result["status"], "error")
        self.assertIn("timed out", result["error"].lower())


# ══════════════════════════════════════════════════════════════
#  Executor Tests
# ══════════════════════════════════════════════════════════════

class TestExecutor(unittest.TestCase):

    # ── Read-only safe execution ──────────────────────────────

    @patch("execution.executor.subprocess.run",
           return_value=make_subprocess_result("file1\nfile2\n"))
    def test_read_only_runs_without_su(self, mock_subproc):
        plan = make_plan(
            commands=["ls /sdcard"],
            action_type="read_only",
        )
        result = execute_plan(plan)
        self.assertEqual(result["status"], "success")
        self.assertIn("file1", result["stdout"])
        # subprocess must NOT be called with 'su'
        args = mock_subproc.call_args[0][0]
        self.assertNotIn("su", args)

    # ── Standard script execution ─────────────────────────────

    @patch("execution.executor.subprocess.run",
           return_value=make_subprocess_result("hello\n"))
    def test_standard_script_runs(self, mock_subproc):
        plan   = make_plan(commands=["echo hello"])
        result = execute_plan(plan)
        self.assertEqual(result["status"], "success")
        self.assertIn("hello", result["stdout"])

    # ── Root plan → delegates to RootGuard ───────────────────

    @patch("execution.executor.execute_with_root",
           return_value={"status": "success", "stdout": "done", "stderr": "", "exit_code": 0})
    def test_root_plan_calls_root_guard(self, mock_rg):
        plan = make_plan(
            commands=["rm -rf /tmp/test"],
            requires_root=True,
        )
        result = execute_plan(plan)
        mock_rg.assert_called_once()
        self.assertEqual(result["status"], "success")

    # ── Root denial propagated ────────────────────────────────

    @patch("execution.executor.execute_with_root",
           return_value={"status": "denied", "error": "User refused root execution."})
    def test_root_denial_propagated(self, mock_rg):
        plan = make_plan(commands=["rm /important"], requires_root=True)
        result = execute_plan(plan)
        self.assertEqual(result["status"], "denied")

    # ── Tamper check propagated ───────────────────────────────

    @patch("execution.executor.execute_with_root",
           return_value={"status": "error", "error": "Security Breach: Hash mismatch"})
    def test_tamper_check_error_propagated(self, mock_rg):
        plan = make_plan(commands=["dangerous cmd"], requires_root=True)
        plan["payload_hash"] = "wrong_hash"
        result = execute_plan(plan)
        self.assertEqual(result["status"], "error")

    # ── Timeout handling ──────────────────────────────────────

    @patch("execution.executor.subprocess.run",
           side_effect=__import__("subprocess").TimeoutExpired("cmd", 30))
    def test_timeout_returns_error(self, mock_subproc):
        plan   = make_plan(commands=["sleep 999"])
        result = execute_plan(plan)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["exit_code"], -1)

    # ── Multiple commands accumulate output ───────────────────

    @patch("execution.executor.subprocess.run")
    def test_multiple_commands_accumulate_stdout(self, mock_subproc):
        mock_subproc.side_effect = [
            make_subprocess_result("line1\n"),
            make_subprocess_result("line2\n"),
        ]
        plan   = make_plan(commands=["echo line1", "echo line2"])
        result = execute_plan(plan)
        self.assertEqual(result["status"], "success")
        self.assertIn("line1", result["stdout"])
        self.assertIn("line2", result["stdout"])

    # ── First failing command stops execution ─────────────────

    @patch("execution.executor.subprocess.run")
    def test_stops_on_first_failure(self, mock_subproc):
        mock_subproc.side_effect = [
            make_subprocess_result("ok\n"),
            make_subprocess_result("", "error msg", returncode=1),
            make_subprocess_result("should not run\n"),   # never called
        ]
        plan   = make_plan(commands=["cmd1", "cmd2", "cmd3"])
        result = execute_plan(plan)
        self.assertEqual(result["status"], "error")
        self.assertEqual(mock_subproc.call_count, 2)  # third cmd not called

    # ── Termux-API tool not found ─────────────────────────────

    @patch("execution.executor.shutil.which", return_value=None)
    def test_termux_api_tool_not_found(self, mock_which):
        plan = make_plan(
            commands=["termux-battery-status"],
            script_type="termux-api",
        )
        result = execute_plan(plan)
        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result["stderr"])

    # ── Termux-API tool found → executes ─────────────────────

    @patch("execution.executor.subprocess.run",
           return_value=make_subprocess_result('{"health":"GOOD"}'))
    @patch("execution.executor.shutil.which", return_value="/data/data/com.termux/files/usr/bin/termux-battery-status")
    def test_termux_api_tool_found_executes(self, mock_which, mock_subproc):
        plan = make_plan(
            commands=["termux-battery-status"],
            script_type="termux-api",
        )
        result = execute_plan(plan)
        self.assertEqual(result["status"], "success")

    # ── Explain / knowledge plan returns text without execution ──

    def test_explain_action_type_no_subprocess(self):
        plan = make_plan(
            commands=["Python is a high-level programming language."],
            action_type="explain",
        )
        with patch("execution.executor.subprocess.run") as mock_subproc:
            result = execute_plan(plan)
            mock_subproc.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertIn("Python", result["stdout"])


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
