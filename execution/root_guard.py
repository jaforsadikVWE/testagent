"""
execution/root_guard.py
────────────────────────────────────────────────────────────
The physical barrier between the AI and the `su` command.

execute_with_root() is the ONLY path through which a root
command may be executed.  It:
  1. Verifies the SHA-256 hash to detect tampering.
  2. Prints the command in bold red so the user can review it.
  3. Asks for explicit [y/N] confirmation.
  4. Runs `su -c <command>` ONLY if the user types "y".

Never call subprocess.run(['su', ...]) from anywhere else.
"""

import hashlib
import json
import subprocess
from typing import Dict, Any

# ANSI colour codes
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def verify_hash(command: str, provided_hash: str) -> bool:
    """
    Compute the SHA-256 of `command` and compare it to `provided_hash`.

    Args:
        command:       The raw command string to hash.
        provided_hash: The hash that the sub-agent embedded in the plan.

    Returns:
        True if hashes match, False otherwise.
    """
    computed = hashlib.sha256(command.encode("utf-8")).hexdigest()
    return computed == provided_hash


def execute_with_root(command: str, provided_hash: str) -> Dict[str, Any]:
    """
    Gate-keep root execution with hash verification + HITL confirmation.

    Args:
        command:       The shell command to run as root.
        provided_hash: SHA-256 of the command (from the sub-agent plan).
                       NOTE: For root_guard the hash covers the single
                       command string, not the full commands array.

    Returns:
        dict with keys: status, stdout, stderr, exit_code  (on success)
        dict with keys: status, error                      (on denial / error)
    """
    # ── 1. Tamper check ──────────────────────────────────────────────
    if not verify_hash(command, provided_hash):
        print(f"{RED}{BOLD}[ROOT GUARD] ⛔ SECURITY BREACH: Hash mismatch!{RESET}")
        print(f"{RED}  Expected hash for command does not match plan hash.{RESET}")
        print(f"{RED}  Execution ABORTED.{RESET}")
        return {
            "status": "error",
            "error":  "Security Breach: Hash mismatch. Execution aborted.",
        }

    # ── 2. Display command in bold red ───────────────────────────────
    print(f"\n{RED}{BOLD}⚠  ROOT COMMAND REQUESTED:{RESET}")
    print(f"{RED}{BOLD}   {command}{RESET}\n")

    # ── 3. Human confirmation ────────────────────────────────────────
    try:
        answer = input(
            f"{RED}⚠️  Review this Root Command. Execute? [y/N]: {RESET}"
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print(f"\n{RED}[ROOT GUARD] Input cancelled. Execution denied.{RESET}")
        return {"status": "denied", "error": "Input cancelled by user."}

    if answer != "y":
        print(f"{RED}[ROOT GUARD] Execution denied by user.{RESET}")
        return {"status": "denied", "error": "User refused root execution."}

    # ── 4. Execute via su ────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["su", "-c", command],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "status":    "success" if result.returncode == 0 else "error",
            "stdout":    result.stdout,
            "stderr":    result.stderr,
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "status":    "error",
            "error":     "Root command timed out after 30 seconds.",
            "exit_code": -1,
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "error":  "su binary not found. Is this device rooted?",
        }
    except Exception as exc:
        return {
            "status": "error",
            "error":  f"Unexpected error during root execution: {exc}",
        }
