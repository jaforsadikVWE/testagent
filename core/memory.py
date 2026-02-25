"""
core/memory.py
────────────────────────────────────────────────────────────
Conversation memory with sliding-window persistence.

Saves every turn to data/history.json so context survives
between sessions.  A configurable window limit keeps the file
small and prevents token exhaustion in the Orchestrator.

Supports full Unicode including emojis (ensure_ascii=False).
"""

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_HISTORY_FILE = "data/history.json"
DEFAULT_MAX_TURNS    = 15


class MemoryManager:
    """
    Manages persistent conversation history.

    Args:
        history_file: Path to the JSON file used for storage.
        max_turns:    Sliding-window size (number of turns to keep).
    """

    def __init__(
        self,
        history_file: str = DEFAULT_HISTORY_FILE,
        max_turns: int    = DEFAULT_MAX_TURNS,
    ) -> None:
        self.history_file = history_file
        self.max_turns    = max_turns
        self._history: List[Dict[str, Any]] = []
        self._ensure_directory()
        self._history = self.load_history()

    # ─────────────────────────── private ─────────────────────────────

    def _ensure_directory(self) -> None:
        """Create the data/ directory if it doesn't exist."""
        directory = os.path.dirname(self.history_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

    # ─────────────────────────── public ──────────────────────────────

    def load_history(self) -> List[Dict[str, Any]]:
        """
        Load conversation history from disk.

        Returns an empty list if the file doesn't exist or is corrupted.
        """
        if not os.path.exists(self.history_file):
            logger.debug(
                f"No history file at {self.history_file}. "
                "Starting with empty history."
            )
            return []

        try:
            with open(self.history_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning("history.json is not a list – resetting.")
                return []
            logger.debug(f"Loaded {len(data)} turns from history.")
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"Could not load history: {exc}. Starting fresh.")
            return []

    def save_history(self) -> None:
        """
        Persist the in-memory history list to disk.

        Uses ensure_ascii=False to correctly store emojis and non-ASCII
        characters (Arabic, Chinese, etc.).
        """
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self._history, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self._history)} turns to history.")
        except OSError as exc:
            logger.error(f"Failed to save history: {exc}")

    def add_message(self, role: str, content: str) -> None:
        """
        Append a new message and persist to disk.

        The sliding window is enforced: if the history exceeds
        max_turns, the oldest entries are dropped.

        Args:
            role:    "user" or "model"
            content: The message text.
        """
        entry = {
            "role":      role,
            "content":   content,
            "timestamp": int(time.time()),
        }
        self._history.append(entry)

        # Apply sliding window
        if len(self._history) > self.max_turns:
            excess = len(self._history) - self.max_turns
            self._history = self._history[excess:]
            logger.debug(
                f"Pruned {excess} old turn(s). "
                f"History size: {len(self._history)}."
            )

        self.save_history()

    def get_context(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return the last `limit` messages for use as context.

        Args:
            limit: Maximum number of recent turns to return.

        Returns:
            List of message dicts, oldest first.
        """
        return self._history[-limit:] if len(self._history) > limit \
            else list(self._history)

    def clear_memory(self) -> None:
        """
        Wipe all history from memory and disk.
        Useful for starting a fresh session.
        """
        self._history = []
        self.save_history()
        logger.info("Memory cleared.")

    def __len__(self) -> int:
        return len(self._history)
