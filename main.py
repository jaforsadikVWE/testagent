"""
main.py
────────────────────────────────────────────────────────────
Entry point for the Termux AI Agent.

Implements a colorised REPL (Read-Eval-Print Loop) that:
  1. Initialises Memory, Orchestrator, and Config.
  2. Accepts user input from the terminal.
  3. Routes input through the Orchestrator → sub-agent → Executor.
  4. Prints results with ANSI colour coding.
  5. Persists each turn to MemoryManager.
  6. Never crashes on runtime errors.

Usage:
  python main.py
  GROQ_API_KEY=gsk_... python main.py
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, List

# ── Internal imports ──────────────────────────────────────────
from core.key_manager  import KeyManager
from core.orchestrator import Orchestrator
from core.memory       import MemoryManager
from core.watchdog     import Watchdog
from execution.executor import execute_plan
from execution.policy   import evaluate_plan, STATUS_APPROVED, STATUS_APPROVAL_REQ

# ── Logging setup (file + stderr) ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger("main")

# ── ANSI colour helpers ───────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

CONFIG_PATH = "config.json"

BANNER = rf"""
{CYAN}{BOLD}
 _____ _____ ____  __  __ _   ___  ___
|_   _| ____|  _ \|  \/  | | | \ \/ /
  | | |  _| | |_) | |\/| | | | |>  < 
  | | | |___|  _ <| |  | | |_| / _ \ 
  |_| |_____|_| \_\_|  |_|\___/_/ \_\
     AI Agent  .  Groq-Powered  .  HITL
{RESET}"""


# ══════════════════════════════════════════════════════════════
#  Initialisation helpers
# ══════════════════════════════════════════════════════════════

def load_config(path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Load config.json and return as dict."""
    if not os.path.exists(path):
        print(f"{RED}[ERROR] config.json not found at '{path}'.{RESET}")
        print(f"{YELLOW}Create it from the template and add your Groq API key(s).{RESET}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bootstrap() -> tuple:
    """
    Initialise all system components.

    Returns:
        (config, key_manager, memory, orchestrator, watchdog)
    """
    config = load_config()

    try:
        km = KeyManager(CONFIG_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{RED}[ERROR] KeyManager failed: {exc}{RESET}")
        print(
            f"{YELLOW}Hint: Replace placeholder keys in config.json "
            "with real Groq API keys.{RESET}"
        )
        sys.exit(1)

    max_turns = config.get("memory", {}).get("max_turns", 15)
    hist_file = config.get("memory", {}).get("history_file", "data/history.json")
    memory    = MemoryManager(history_file=hist_file, max_turns=max_turns)

    log_file     = config.get("logging", {}).get("audit_log",     "logs/audit.log")
    max_bytes    = config.get("logging", {}).get("max_bytes",      1_048_576)
    backup_count = config.get("logging", {}).get("backup_count",   5)
    watchdog     = Watchdog(log_file=log_file, max_bytes=max_bytes, backup_count=backup_count)

    orchestrator = Orchestrator(key_manager=km)

    return config, km, memory, orchestrator, watchdog


# ══════════════════════════════════════════════════════════════
#  Display helpers
# ══════════════════════════════════════════════════════════════

def print_plan(plan: Dict[str, Any]) -> None:
    """Pretty-print the agent's plan in yellow."""
    agent      = plan.get("_selected_agent", plan.get("agent", "Unknown"))
    action     = plan.get("action_type",    "?")
    risk       = plan.get("risk_level",     "?")
    confidence = plan.get("confidence_score", 0.0)
    commands   = plan.get("parameters", {}).get("commands", [])

    print(f"\n{YELLOW}{BOLD}> Agent selected : {agent}{RESET}")
    print(f"{YELLOW}  Action type    : {action}{RESET}")
    print(f"{YELLOW}  Risk level     : {risk}  |  Confidence: {confidence:.0%}{RESET}")
    if commands:
        print(f"{YELLOW}  Commands ({len(commands)}):{RESET}")
        for i, cmd in enumerate(commands, 1):
            display = cmd if len(cmd) <= 120 else cmd[:117] + "..."
            print(f"{YELLOW}    [{i}] {display}{RESET}")


def print_result(result: Dict[str, Any]) -> None:
    """Pretty-print execution result in cyan (or red on error)."""
    status = result.get("status", "unknown")
    stdout = result.get("stdout", "").strip()
    stderr = result.get("stderr", "").strip()

    if status == "success":
        print(f"\n{CYAN}{BOLD}[OK] Result:{RESET}")
        if stdout:
            print(f"{CYAN}{stdout}{RESET}")
        else:
            print(f"{CYAN}(no output){RESET}")
    elif status == "denied":
        print(f"\n{RED}{BOLD}[DENIED] Execution denied by user.{RESET}")
    else:
        print(f"\n{RED}{BOLD}[ERROR] Execution error:{RESET}")
        if stderr:
            print(f"{RED}{stderr}{RESET}")
        if stdout:
            print(f"{RED}{stdout}{RESET}")


def approval_prompt(plan: Dict[str, Any]) -> bool:
    """
    Present an APPROVAL_REQUIRED prompt to the user.

    Returns True if the user approves, False otherwise.
    """
    reason   = plan.get("policy_reason", "")
    commands = plan.get("parameters", {}).get("commands", [])

    print(f"\n{RED}{BOLD}⚠  HUMAN APPROVAL REQUIRED{RESET}")
    print(f"{RED}   Reason: {reason}{RESET}")
    print(f"{RED}   Commands to run:{RESET}")
    for cmd in commands:
        print(f"{RED}     → {cmd}{RESET}")

    try:
        answer = input(f"\n{RED}Approve execution? [y/N]: {RESET}").strip().lower()
        return answer == "y"
    except (EOFError, KeyboardInterrupt):
        return False


# ══════════════════════════════════════════════════════════════
#  Main REPL
# ══════════════════════════════════════════════════════════════

def main() -> None:
    print(BANNER)

    # ── Bootstrap ────────────────────────────────────────────
    config, km, memory, orchestrator, watchdog = bootstrap()
    print(f"{GREEN}[OK] System initialised. Type 'help' for tips, 'exit' to quit.{RESET}\n")

    history: List[Dict[str, str]] = []

    # ── REPL Loop ─────────────────────────────────────────────
    while True:
        try:
            # ── Get user input ────────────────────────────────
            try:
                user_input = input(
                    f"{GREEN}{BOLD}root@termux-agent:~$ {RESET}"
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{YELLOW}Goodbye!{RESET}")
                break

            if not user_input:
                continue

            # ── Built-in commands ──────────────────────────────
            if user_input.lower() in ("exit", "quit", "q"):
                print(f"{YELLOW}Goodbye!{RESET}")
                break

            if user_input.lower() == "help":
                _print_help()
                continue

            if user_input.lower() == "clear":
                memory.clear_memory()
                history.clear()
                print(f"{CYAN}Memory cleared.{RESET}")
                continue

            if user_input.lower() == "history":
                _print_history(memory)
                continue

            # ── Orchestrate ───────────────────────────────────
            print(f"\n{YELLOW}{DIM}Thinking...{RESET}", end="", flush=True)
            t0   = time.time()
            plan = orchestrator.handle(user_input, history)
            elapsed = time.time() - t0
            print(f"\r{DIM}(planned in {elapsed:.1f}s){RESET}          ")

            print_plan(plan)

            # ── Policy evaluation ─────────────────────────────
            policy_status, plan = evaluate_plan(plan, CONFIG_PATH)

            # ── HITL gate ─────────────────────────────────────
            if policy_status == STATUS_APPROVAL_REQ:
                approved = approval_prompt(plan)
                if not approved:
                    watchdog.log_risk(
                        action_type=plan.get("action_type", "unknown"),
                        risk_level=plan.get("risk_level", "high"),
                        command=str(plan.get("parameters", {}).get("commands", [])),
                        outcome="DENIED_BY_USER",
                        requires_root=plan.get("requires_root", False),
                    )
                    print(f"{RED}Execution cancelled.{RESET}")
                    memory.add_message("user",  user_input)
                    memory.add_message("model", "Execution cancelled by user.")
                    history.append({"role": "user",  "content": user_input})
                    history.append({"role": "model", "content": "Execution cancelled."})
                    continue

            # ── Execute ───────────────────────────────────────
            result = execute_plan(plan)
            print_result(result)

            # ── Watchdog: log risky executions ────────────────
            if (plan.get("risk_level") == "high" or plan.get("requires_root")):
                watchdog.log_risk(
                    action_type=plan.get("action_type", "unknown"),
                    risk_level=plan.get("risk_level", "high"),
                    command=str(plan.get("parameters", {}).get("commands", [])),
                    outcome=result.get("status", "unknown").upper(),
                    requires_root=plan.get("requires_root", False),
                )

            # ── Persist to memory ─────────────────────────────
            result_summary = (
                result.get("stdout", "") or result.get("error", "")
            )[:300]

            memory.add_message("user",  user_input)
            memory.add_message("model", result_summary)
            history.append({"role": "user",  "content": user_input})
            history.append({"role": "model", "content": result_summary})

        except Exception as exc:
            # ── Never crash the REPL ──────────────────────────
            print(f"\n{RED}[ERROR] {exc}{RESET}", file=sys.stderr)
            logger.exception("Unhandled error in REPL loop")
            watchdog.log_failure("REPL", str(exc))


# ══════════════════════════════════════════════════════════════
#  Helper display functions
# ══════════════════════════════════════════════════════════════

def _print_help() -> None:
    print(f"""
{CYAN}{BOLD}Termux AI Agent – Tips{RESET}
{CYAN}  exit / quit     → Exit the agent
  clear            → Wipe conversation memory
  history          → Show recent memory
  <any text>       → Send to the AI agent

{YELLOW}Example prompts:{RESET}
{YELLOW}  List all files in /sdcard/Download
  Write a Python script to rename files
  What is my battery status?
  Explain how chmod works
  Move all .jpg files to /sdcard/Photos{RESET}
""")


def _print_history(memory: MemoryManager) -> None:
    context = memory.get_context(limit=10)
    if not context:
        print(f"{DIM}(no history){RESET}")
        return
    print(f"\n{CYAN}{BOLD}Recent memory ({len(context)} turns):{RESET}")
    for turn in context:
        role    = turn.get("role", "?").upper()
        content = turn.get("content", "")
        short   = content[:120] + "..." if len(content) > 120 else content
        colour  = GREEN if role == "USER" else YELLOW
        print(f"  {colour}[{role}]{RESET} {short}")
    print()


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
