"""
core/key_manager.py
────────────────────────────────────────────────────────────
Multi-key API rotator for Groq.

Holds a list of API keys and advances to the next one whenever
a groq.RateLimitError (HTTP 429) is raised.  The caller simply
calls `get_current_key()`, makes its API call, and on failure
calls `rotate_key()` before retrying – no user interruption.
"""

import json
import os
import logging
from typing import List, Optional

import groq  # pip install groq

logger = logging.getLogger(__name__)


class KeyManager:
    """Manages a pool of Groq API keys with automatic rotation."""

    def __init__(self, config_path: str = "config.json"):
        """
        Load API keys from config.json.

        Args:
            config_path: Path to config.json (relative to cwd or absolute).

        Raises:
            FileNotFoundError: If config file doesn't exist.
            ValueError: If no valid keys are found in config.
        """
        self._keys: List[str] = []
        self._index: int = 0
        self._load_keys(config_path)

    # ─────────────────────────── private ────────────────────────────

    def _load_keys(self, config_path: str) -> None:
        """Parse and validate API keys from config.json."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        raw_keys: List[str] = config.get("groq_api_keys", [])

        # Filter out placeholder / empty keys
        valid_keys = [
            k for k in raw_keys
            if k and not k.startswith("gsk_YOUR")
        ]

        if not valid_keys:
            raise ValueError(
                "No valid Groq API keys found in config.json. "
                "Please replace the placeholder values."
            )

        self._keys = valid_keys
        self._index = 0
        logger.info(f"KeyManager loaded {len(self._keys)} key(s).")

    # ─────────────────────────── public ─────────────────────────────

    def get_current_key(self) -> str:
        """Return the currently active API key."""
        return self._keys[self._index]

    def rotate_key(self) -> Optional[str]:
        """
        Advance to the next available key.

        Returns:
            The new active key, or None if all keys are exhausted.
        """
        next_index = self._index + 1

        if next_index >= len(self._keys):
            logger.error("All API keys exhausted – no more keys to rotate to.")
            return None

        self._index = next_index
        logger.warning(
            f"Key rotated → using key index {self._index} "
            f"(of {len(self._keys)})."
        )
        return self._keys[self._index]

    def reset(self) -> None:
        """Reset the rotator back to the first key."""
        self._index = 0

    def keys_remaining(self) -> int:
        """Return how many keys are still available after the current one."""
        return len(self._keys) - self._index - 1

    def total_keys(self) -> int:
        """Return total number of loaded keys."""
        return len(self._keys)

    def make_client(self) -> "groq.Groq":
        """
        Convenience: create and return a Groq client using the current key.

        Usage:
            client = key_manager.make_client()
            response = client.chat.completions.create(...)
        """
        return groq.Groq(api_key=self.get_current_key())
