"""
core/orchestrator.py
────────────────────────────────────────────────────────────
Main controller: routes requests to sub-agents via a fast Router model.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Tuple

import groq

from core.key_manager import KeyManager
from agents.base_agent import BaseAgent
from agents.system_prompt import ROUTER_SYSTEM_PROMPT
from agents.sub_agents import AGENT_REGISTRY

logger = logging.getLogger(__name__)

FALLBACK_AGENT = "KnowledgeAgent"
ROUTER_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"


class Orchestrator:
    def __init__(self, key_manager: KeyManager, max_retries: int = 3) -> None:
        self.key_manager = key_manager
        self.max_retries = max_retries

        self._router = BaseAgent(
            model_name=ROUTER_MODEL,
            key_manager=key_manager,
        )

        self._agents: Dict[str, BaseAgent] = {
            name: cls(key_manager)
            for name, cls in AGENT_REGISTRY.items()
        }
        logger.info(f"Orchestrator ready. Agents: {list(self._agents.keys())}")

    # ─────────────────────────── public ──────────────────────────────

    def route_request(
        self,
        user_input: str,
        history: List[Dict[str, str]],
    ) -> Tuple[str, str]:
        """Ask the Router which agent to use and get a task summary."""
        history_text  = self._format_history(history)
        router_prompt = (
            f"Conversation so far:\n{history_text}\n\n"
            f"New user request: {user_input}\n\n"
            "Pick the best agent and summarize the task."
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._router.generate(
                    prompt=router_prompt,
                    system_instruction=ROUTER_SYSTEM_PROMPT,
                )
                logger.info(f"Router raw result: {result}")

                agent_name = str(result.get("selected_agent", "")).strip()
                context    = str(result.get(
                    "synthesized_context",
                    result.get("context", user_input)
                )).strip() or user_input

                if agent_name not in self._agents:
                    logger.warning(
                        f"Router returned unknown agent '{agent_name}'. "
                        f"Attempt {attempt}/{self.max_retries}. "
                        f"Valid: {list(self._agents.keys())}"
                    )
                    continue

                logger.info(f"Routing to {agent_name}: {context[:100]}")
                return agent_name, context

            except (groq.RateLimitError, RuntimeError, ValueError) as exc:
                logger.warning(f"Router error on attempt {attempt}: {exc}")
                time.sleep(1)

        logger.error(
            f"Router failed after {self.max_retries} attempts. "
            f"Falling back to {FALLBACK_AGENT}."
        )
        return FALLBACK_AGENT, user_input

    def handle(
        self,
        user_input: str,
        history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """Route + dispatch. Returns Phase-1 JSON plan dict."""
        agent_name, context = self.route_request(user_input, history)
        logger.info(f"Dispatching to {agent_name}.")
        agent = self._agents[agent_name]

        try:
            plan = agent.run(context)
            plan["_selected_agent"] = agent_name
            return plan
        except Exception as exc:
            logger.error(f"Sub-agent {agent_name} failed: {exc}")
            return self._error_plan(agent_name, str(exc))

    # ─────────────────────────── private ─────────────────────────────

    @staticmethod
    def _format_history(history: List[Dict[str, str]], limit: int = 6) -> str:
        recent = history[-limit:] if len(history) > limit else history
        if not recent:
            return "(no prior context)"
        return "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in recent
        )

    @staticmethod
    def _error_plan(agent_name: str, error_msg: str) -> Dict[str, Any]:
        return {
            "_selected_agent":  agent_name,
            "_error":           error_msg,
            "agent":            agent_name,
            "action_type":      "explain",
            "parameters":       {"script_type": "bash", "commands": [error_msg]},
            "requires_root":    False,
            "confidence_score": 0.0,
            "risk_level":       "low",
            "dry_run_safe":     True,
            "payload_hash":     "",
        }
