"""
core/watchdog.py
────────────────────────────────────────────────────────────
Immutable audit logger for high-risk and root actions.

Every dangerous action, root command, and failure is appended
to logs/audit.log using Python's RotatingFileHandler so the log
never fills the device's storage.

This module contains NO AI logic – it is purely deterministic.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional

DEFAULT_LOG_DIR  = "logs"
DEFAULT_LOG_FILE = "logs/audit.log"
MAX_BYTES        = 1_048_576   # 1 MB
BACKUP_COUNT     = 5


class Watchdog:
    """
    Audit logger for the HITL agent system.

    Args:
        log_file:     Path to the rotating audit log.
        max_bytes:    Maximum size before log rotation.
        backup_count: Number of backup log files to keep.
    """

    def __init__(
        self,
        log_file:     str = DEFAULT_LOG_FILE,
        max_bytes:    int = MAX_BYTES,
        backup_count: int = BACKUP_COUNT,
    ) -> None:
        self.log_file = log_file
        self._ensure_log_directory()
        self._logger  = self._setup_logger(log_file, max_bytes, backup_count)

    # ─────────────────────────── private ─────────────────────────────

    def _ensure_log_directory(self) -> None:
        """Create the logs/ directory if it doesn't exist."""
        directory = os.path.dirname(self.log_file)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    @staticmethod
    def _setup_logger(
        log_file:     str,
        max_bytes:    int,
        backup_count: int,
    ) -> logging.Logger:
        """Configure a dedicated rotating file logger for audit events."""
        logger = logging.getLogger("watchdog.audit")
        logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers if Watchdog is instantiated multiple times
        if logger.handlers:
            return logger

        handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        # Plain-text format – no Python logging boilerplate
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

        # Also suppress propagation so audit lines don't appear in root log
        logger.propagate = False

        return logger

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ─────────────────────────── public ──────────────────────────────

    def log_risk(
        self,
        action_type:   str,
        risk_level:    str,
        command:       str,
        outcome:       str,
        requires_root: bool = False,
    ) -> None:
        """
        Record a risk event to the audit log.

        Logs with level CRITICAL when risk_level is "high" or
        requires_root is True; WARNING otherwise.

        Format:
            [2024-01-01 12:00:00] [RISK: HIGH] [ROOT: YES]
            [TYPE: script] [CMD: rm -rf /] [OUTCOME: DENIED]

        Args:
            action_type:   Plan's action_type field.
            risk_level:    "low" | "medium" | "high"
            command:       The command string (or summary).
            outcome:       Human-readable outcome, e.g. "APPROVED", "DENIED".
            requires_root: Whether the command needed root.
        """
        ts         = self._timestamp()
        risk_tag   = risk_level.upper()
        root_tag   = "YES" if requires_root else "NO"
        is_critical = risk_level.lower() == "high" or requires_root

        log_line = (
            f"[{ts}] "
            f"[RISK: {risk_tag}] "
            f"[ROOT: {root_tag}] "
            f"[TYPE: {action_type}] "
            f"[CMD: {command}] "
            f"[OUTCOME: {outcome}]"
        )

        if is_critical:
            self._logger.critical(log_line)
        else:
            self._logger.warning(log_line)

    def log_failure(self, context: str, error: str) -> None:
        """
        Record an unexpected system failure.

        Args:
            context: Where the failure occurred (e.g. "Executor").
            error:   Error message or traceback summary.
        """
        ts       = self._timestamp()
        log_line = f"[{ts}] [FAILURE] [CONTEXT: {context}] [ERROR: {error}]"
        self._logger.critical(log_line)

    def log_info(self, message: str) -> None:
        """Log a general informational audit event."""
        ts = self._timestamp()
        self._logger.info(f"[{ts}] [INFO] {message}")
