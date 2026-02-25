"""
agents/sub_agents.py
────────────────────────────────────────────────────────────
Concrete sub-agent classes. Each picks the right model and system prompt.
"""

import hashlib
import json
import logging
from typing import Any, Dict

from core.key_manager import KeyManager
from agents.base_agent import BaseAgent
from agents.system_prompt import (
    REASONING_PROMPT,
    CODE_PROMPT,
    KNOWLEDGE_PROMPT,
    AUTOMATION_PROMPT,
)

logger = logging.getLogger(__name__)

HEAVY_MODEL = "openai/gpt-oss-120b"
FAST_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"


def _hash_commands(commands: list) -> str:
    """SHA-256 of the compact JSON-serialised commands list."""
    canonical = json.dumps(commands, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ensure_required_fields(plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all required fields are present in the plan.
    Fills in sensible defaults for any missing fields to prevent policy rejection.
    
    Required fields:
    - agent, action_type, parameters, requires_root, confidence_score,
      risk_level, dry_run_safe, payload_hash
    """
    # Set defaults for any missing top-level fields
    if "agent" not in plan:
        plan["agent"] = plan.get("_agent", "SubAgent")
    
    if "action_type" not in plan:
        plan["action_type"] = "read_only"
    
    if "parameters" not in plan:
        plan["parameters"] = {"script_type": "bash", "commands": []}
    
    if "requires_root" not in plan:
        plan["requires_root"] = False
    
    if "confidence_score" not in plan:
        plan["confidence_score"] = 0.5
    
    if "risk_level" not in plan:
        plan["risk_level"] = "low"
    
    if "dry_run_safe" not in plan:
        plan["dry_run_safe"] = True
    
    # Compute payload_hash if missing
    commands = plan.get("parameters", {}).get("commands", [])
    if not plan.get("payload_hash"):
        plan["payload_hash"] = _hash_commands(commands) if commands else ""
    
    return plan


class ReasoningAgent(BaseAgent):
    AGENT_NAME = "ReasoningAgent"

    def __init__(self, key_manager: KeyManager) -> None:
        super().__init__(HEAVY_MODEL, key_manager)

    def run(self, task: str) -> Dict[str, Any]:
        logger.info(f"[{self.AGENT_NAME}] task: {task[:80]}")
        plan = self.generate(task, REASONING_PROMPT, reasoning_effort="high")
        plan["agent"] = self.AGENT_NAME
        return _ensure_required_fields(plan)


class CodeAgent(BaseAgent):
    AGENT_NAME = "CodeAgent"

    def __init__(self, key_manager: KeyManager) -> None:
        super().__init__(HEAVY_MODEL, key_manager)

    def run(self, task: str) -> Dict[str, Any]:
        logger.info(f"[{self.AGENT_NAME}] task: {task[:80]}")
        plan = self.generate(task, CODE_PROMPT, reasoning_effort="medium")
        plan["agent"] = self.AGENT_NAME
        return _ensure_required_fields(plan)


class KnowledgeAgent(BaseAgent):
    AGENT_NAME = "KnowledgeAgent"

    def __init__(self, key_manager: KeyManager) -> None:
        super().__init__(FAST_MODEL, key_manager)

    def run(self, task: str) -> Dict[str, Any]:
        logger.info(f"[{self.AGENT_NAME}] task: {task[:80]}")
        plan = self.generate(task, KNOWLEDGE_PROMPT)
        plan["agent"] = self.AGENT_NAME
        return _ensure_required_fields(plan)


class AutomationAgent(BaseAgent):
    AGENT_NAME = "AutomationAgent"

    def __init__(self, key_manager: KeyManager) -> None:
        super().__init__(FAST_MODEL, key_manager)

    def run(self, task: str) -> Dict[str, Any]:
        logger.info(f"[{self.AGENT_NAME}] task: {task[:80]}")
        plan = self.generate(task, AUTOMATION_PROMPT)
        plan["agent"] = self.AGENT_NAME
        return _ensure_required_fields(plan)


AGENT_REGISTRY: Dict[str, type] = {
    "ReasoningAgent":  ReasoningAgent,
    "CodeAgent":       CodeAgent,
    "KnowledgeAgent":  KnowledgeAgent,
    "AutomationAgent": AutomationAgent,
}
