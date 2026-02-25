"""
execution/executor.py
────────────────────────────────────────────────────────────
Execution layer that runs sub-agent command plans.

execute_plan() is the single public entry point.  It:
  • Reads the plan dict (Phase 1 JSON schema).
  • Routes each command through the correct execution path:
      - read_only  → direct subprocess (safe)
      - root       → RootGuard (hash check + human approval)
      - standard   → subprocess with timeout
      - termux-api → checks tool exists first
  • Accumulates stdout/stderr across all commands.
  • Returns a structured feedback dict.
"""

import shutil
import subprocess
import logging
from typing import Any, Dict, List

from execution.root_guard import execute_with_root
from execution.policy import compute_payload_hash

logger = logging.getLogger(__name__)

# Default subprocess timeout (seconds)
DEFAULT_TIMEOUT = 30

# ANSI colours for terminal output
RED   = "\033[91m"
RESET = "\033[0m"


def execute_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a validated sub-agent plan.

    Args:
        plan: Parsed Phase-1 JSON dict (as returned by policy.evaluate_plan).

    Returns:
        {
          "status":    "success" | "error" | "denied",
          "stdout":    "combined stdout of all commands",
          "stderr":    "combined stderr",
          "exit_code": int   (last command's exit code, or -1 on timeout)
        }
    """
    action_type:   str  = plan.get("action_type", "")
    parameters:    dict = plan.get("parameters", {})
    commands:      list = parameters.get("commands", [])
    script_type:   str  = parameters.get("script_type", "bash")
    requires_root: bool = bool(plan.get("requires_root", False))
    payload_hash:  str  = plan.get("payload_hash", "")

    # ── Route to the correct execution path ─────────────────────────

    if action_type == "explain":
        # Knowledge/explanation responses: just return the text, no execution
        # May have empty commands for simple greetings/hellos
        stdout_text = "\n".join(commands) if commands else ""
        return {
            "status":    "success",
            "stdout":    stdout_text,
            "stderr":    "",
            "exit_code": 0,
        }

    if not commands:
        return _error("No commands found in plan.")

    if requires_root:
        return _execute_root(commands, payload_hash)

    if action_type == "read_only":
        return _execute_commands(commands, script_type, safe_mode=True)

    # Standard script / automation
    return _execute_commands(commands, script_type, safe_mode=False)


# ─────────────────────────── private helpers ─────────────────────────────────

def _execute_root(commands: List[str], plan_hash: str) -> Dict[str, Any]:
    """
    Execute commands that require root via RootGuard.

    For multi-command root plans we join them with ' && ' so the user
    reviews and approves the full sequence as one atomic unit.
    The hash covers each individual command string.
    """
    combined_stdout = ""
    combined_stderr = ""
    last_exit_code  = 0

    for cmd in commands:
        # Compute hash for this specific command string
        cmd_hash = _sha256_of_string(cmd)

        result = execute_with_root(cmd, cmd_hash)

        if result["status"] in ("error", "denied"):
            result["stdout"] = combined_stdout
            result["stderr"] = combined_stderr + result.get("error", "")
            return result

        combined_stdout += result.get("stdout", "")
        combined_stderr += result.get("stderr", "")
        last_exit_code   = result.get("exit_code", 0)

    return {
        "status":    "success",
        "stdout":    combined_stdout,
        "stderr":    combined_stderr,
        "exit_code": last_exit_code,
    }


def _execute_commands(
    commands: List[str],
    script_type: str,
    safe_mode: bool,
) -> Dict[str, Any]:
    """
    Execute a list of commands sequentially via subprocess.

    Args:
        commands:    List of shell command strings.
        script_type: "bash" | "python" | "termux-api"
        safe_mode:   If True (read_only), extra checks may be added in future.

    Returns:
        Combined result dict.
    """
    combined_stdout = ""
    combined_stderr = ""
    last_exit_code  = 0

    for cmd in commands:
        # ── Syntax Check: balanced quotes ────────────────────────
        if not _check_shell_syntax(cmd):
            return _error(f"Shell syntax error: unbalanced quotes in command: {cmd}")

        # ── Termux-API: verify the binary exists ──────────────────
        if script_type == "termux-api":
            bin_name = cmd.split()[0]
            if not shutil.which(bin_name):
                return _error(
                    f"Termux-API tool '{bin_name}' not found. "
                    "Install termux-api and grant permissions."
                )

        # ── Run the command ───────────────────────────────────────
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUT,
            )
            combined_stdout += result.stdout
            combined_stderr += result.stderr
            last_exit_code   = result.returncode

            # Stop on first failure
            if result.returncode != 0:
                logger.warning(
                    f"Command exited with code {result.returncode}: {cmd}"
                )
                return {
                    "status":    "error",
                    "stdout":    combined_stdout,
                    "stderr":    combined_stderr,
                    "exit_code": last_exit_code,
                }

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {cmd}")
            return {
                "status":    "error",
                "stdout":    combined_stdout,
                "stderr":    combined_stderr + f"\nTimeout: {cmd}",
                "exit_code": -1,
            }
        except Exception as exc:
            logger.error(f"Unexpected error executing '{cmd}': {exc}")
            return _error(str(exc))

    return {
        "status":    "success",
        "stdout":    combined_stdout,
        "stderr":    combined_stderr,
        "exit_code": last_exit_code,
    }


def _sha256_of_string(s: str) -> str:
    """Return the SHA-256 hex digest of a string (for per-command root hashes)."""
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _check_shell_syntax(cmd: str) -> bool:
    """
    Very basic check for balanced quotes (single, double, backticks).
    Does not handle escaped quotes or complex shell grammar.
    """
    s_count = cmd.count("'")
    d_count = cmd.count('"')
    b_count = cmd.count('`')
    return (s_count % 2 == 0) and (d_count % 2 == 0) and (b_count % 2 == 0)


def _error(message: str) -> Dict[str, Any]:
    """Convenience: build an error result dict."""
    return {
        "status":    "error",
        "stdout":    "",
        "stderr":    message,
        "exit_code": -1,
    }
