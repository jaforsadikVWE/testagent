"""
agents/base_agent.py
────────────────────────────────────────────────────────────
Groq API wrapper with automatic key rotation on RateLimitError.
"""

import json
import logging
import re
import time
from typing import Any, Dict, Optional

import groq

from core.key_manager import KeyManager

logger = logging.getLogger(__name__)

MAX_RETRIES: int = 3

# Models that support the reasoning_effort parameter (Groq specific)
# Most Groq models don't support this, but including it doesn't hurt – it's just ignored
REASONING_MODELS = set()


class BaseAgent:
    def __init__(
        self,
        model_name: str,
        key_manager: KeyManager,
        max_retries: int = MAX_RETRIES,
    ) -> None:
        self.model_name  = model_name
        self.key_manager = key_manager
        self.max_retries = max_retries
        self._client     = key_manager.make_client()

    # ─────────────────────────── public ──────────────────────────────

    def generate(
        self,
        prompt: str,
        system_instruction: str,
        reasoning_effort: str = "medium",
    ) -> Dict[str, Any]:
        """
        Send a prompt and return a parsed JSON dict.
        reasoning_effort is ONLY passed for REASONING_MODELS.
        """
        last_error: Optional[Exception] = None
        last_error_detail: str = ""

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"[{self.model_name}] attempt {attempt}/{self.max_retries}")

                raw_text = self._call_api(prompt, system_instruction, reasoning_effort)
                logger.debug(f"[{self.model_name}] raw response: {repr(raw_text[:200])}")

                if not raw_text or not raw_text.strip():
                    raise ValueError(
                        f"Model returned empty content on attempt {attempt}."
                    )

                return self._parse_json(raw_text)

            except groq.RateLimitError as exc:
                logger.warning(f"RateLimitError on attempt {attempt}: {exc}. Rotating key…")
                last_error = exc
                last_error_detail = str(exc)
                new_key = self.key_manager.rotate_key()
                if new_key is None:
                    raise RuntimeError("All API keys exhausted.") from exc
                self._client = self.key_manager.make_client()
                time.sleep(1)

            except groq.APIError as exc:
                logger.error(f"APIError on attempt {attempt}: {exc}")
                last_error = exc
                last_error_detail = str(exc)
                time.sleep(2)

            except (json.JSONDecodeError, ValueError) as exc:
                logger.error(f"Parse error on attempt {attempt}: {exc}")
                last_error = exc
                last_error_detail = str(exc)

        error_msg = (
            f"generate() failed after {self.max_retries} attempts using model={self.model_name}. "
            f"Last error: {last_error_detail}"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # ─────────────────────────── private ─────────────────────────────

    def _call_api(
        self,
        prompt: str,
        system_instruction: str,
        reasoning_effort: str,
    ) -> str:
        """Fire the Groq API call and return raw text content."""
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user",   "content": prompt},
        ]

        kwargs: Dict[str, Any] = {
            "model":    self.model_name,
            "messages": messages,
            "temperature": 1,
            "top_p":    1,
            "stream":   False,
            "stop":     None,
        }

        # ── Model-specific params ────────────────────────────────────
        if self.model_name in REASONING_MODELS:
            # Heavy reasoning models: large budget + reasoning_effort
            kwargs["max_completion_tokens"] = 8192
            kwargs["reasoning_effort"]      = reasoning_effort
        else:
            # Fast models (llama-4-scout etc.): NO reasoning_effort param
            kwargs["max_completion_tokens"] = 2048

        response = self._client.chat.completions.create(**kwargs)
        choice   = response.choices[0]
        content  = choice.message.content

        # Reasoning models sometimes place output only in reasoning_content
        # and return None for content.  Try to recover it.
        if not content:
            reasoning = getattr(choice.message, "reasoning_content", None)
            if reasoning:
                logger.warning(
                    f"[{self.model_name}] content is empty but reasoning_content "
                    "has data – extracting JSON from reasoning."
                )
                content = reasoning
            else:
                logger.warning(
                    f"[{self.model_name}] returned empty content. "
                    f"finish_reason={choice.finish_reason}"
                )
                return ""

        return content

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        """
        Robustly extract a JSON object from model output.

        Tries in order:
          1. Strip ``` fences, parse whole string.
          2. Find first {...} block with regex (handles prose wrappers).
          3. Raise with helpful message.
        """
        # Step 1 – strip markdown fences
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        cleaned = re.sub(r"```",          "", cleaned).strip()

        # Step 2 – direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Step 3 – extract first {...} block
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", cleaned, re.DOTALL)
        if not match:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)   # greedy fallback

        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(
            f"No valid JSON found in model output.\n"
            f"Raw (first 400 chars):\n{raw[:400]}"
        )
