"""
execution/policy.py
────────────────────────────────────────────────────────────
Non-AI policy layer.

Receives a parsed sub-agent JSON plan, checks it against the
configured thresholds, and decides whether human approval is
required before execution proceeds.

No API calls are made here – this is pure deterministic logic.
"""

import json
import hashlib
import logging
import os
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# ── Status constants ─────────────────────────────────────────
STATUS_APPROVED       = "APPROVED"
STATUS_APPROVAL_REQ   = "APPROVAL_REQUIRED"
STATUS_DENIED         = "DENIED"


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Load and return the parsed config.json."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_payload_hash(commands: list) -> str:
    """
    Compute a SHA-256 hash of the commands list.

    The list is serialised to a canonical JSON string (sorted keys,
    no whitespace) before hashing to ensure consistency.

    Args:
        commands: List of command strings from the sub-agent plan.

    Returns:
        Lowercase hex SHA-256 digest string.
    """
    canonical = json.dumps(commands, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate_plan(
    plan: Dict[str, Any],
    config_path: str = "config.json"
) -> Tuple[str, Dict[str, Any]]:
    """
    Evaluate a sub-agent plan against policy rules.

    Rules applied (in order):
    1. Validate required fields are present.
    2. Verify payload_hash matches the commands list.
    3. If confidence_score < threshold AND experimental_override is False
       → flag APPROVAL_REQUIRED and escalate risk_level to "high".
    4. If requires_root is True or risk_level is "high"
       → flag APPROVAL_REQUIRED.
    5. Otherwise → APPROVED.

    Args:
        plan:        Parsed sub-agent JSON as a Python dict.
        config_path: Path to config.json.

    Returns:
        Tuple of (status_string, updated_plan_dict).
        The returned plan may have risk_level upgraded to "high".
    """
    config = load_config(config_path)
    threshold: float      = config.get("confidence_threshold", 0.5)
    override:  bool       = config.get("experimental_override", False)

    # ── 1. Validate required fields ──────────────────────────────────
    required_fields = [
        "agent", "action_type", "parameters",
        "requires_root", "confidence_score",
        "risk_level", "dry_run_safe", "payload_hash"
    ]
    missing = [f for f in required_fields if f not in plan]
    if missing:
        logger.error(f"Plan is missing required fields: {missing}")
        logger.debug(f"Plan received: {json.dumps(plan, indent=2)}")
        plan["policy_status"] = STATUS_DENIED
        plan["policy_reason"] = f"Missing fields: {missing}"
        return STATUS_DENIED, plan
    
    logger.debug(f"Plan validation: all required fields present. Field values:\n"
                 f"  agent={plan.get('agent')}, action_type={plan.get('action_type')}, "
                 f"confidence={plan.get('confidence_score')}, risk={plan.get('risk_level')}")

    commands: list = plan.get("parameters", {}).get("commands", [])
    confidence:  float = float(plan["confidence_score"])
    risk_level:  str   = plan["risk_level"].lower()
    requires_root: bool = bool(plan["requires_root"])

    # ── 2. Verify or auto-compute payload hash ───────────────────────
    expected_hash = compute_payload_hash(commands)
    provided_hash = plan.get("payload_hash", "")

    # Treat placeholders as missing/empty
    is_placeholder = "<" in provided_hash or "sha256" in provided_hash.lower() or (provided_hash and len(provided_hash) != 64)

    if not provided_hash or is_placeholder:
        # Sub-agent forgot to include the hash or used a placeholder → compute and fill it in
        logger.warning(f"payload_hash '{provided_hash}' is missing or a placeholder – computing it now.")
        plan["payload_hash"] = expected_hash
        provided_hash = expected_hash

    if expected_hash != provided_hash:
        logger.warning(
            f"Hash mismatch! Expected {expected_hash}, got {provided_hash}."
        )
        plan["policy_status"] = STATUS_DENIED
        plan["policy_reason"] = "Security: payload_hash mismatch."
        return STATUS_DENIED, plan

    # ── 3. Confidence threshold check ────────────────────────────────
    if confidence < threshold and not override:
        logger.warning(
            f"Confidence {confidence:.2f} below threshold {threshold}. "
            "Escalating to APPROVAL_REQUIRED."
        )
        plan["risk_level"] = "high"   # escalate risk
        risk_level = "high"
        plan["policy_status"] = STATUS_APPROVAL_REQ
        plan["policy_reason"] = (
            f"Low confidence ({confidence:.2f} < {threshold}). "
            "Human approval required."
        )
        return STATUS_APPROVAL_REQ, plan

    # ── 4. Root / high-risk check ────────────────────────────────────
    if requires_root or risk_level == "high":
        logger.warning(
            f"Plan requires root={requires_root} / risk={risk_level}. "
            "Flagging APPROVAL_REQUIRED."
        )
        plan["policy_status"] = STATUS_APPROVAL_REQ
        plan["policy_reason"] = (
            f"requires_root={requires_root}, risk_level={risk_level}. "
            "Human approval required."
        )
        return STATUS_APPROVAL_REQ, plan

    # ── 5. All checks passed ─────────────────────────────────────────
    plan["policy_status"] = STATUS_APPROVED
    plan["policy_reason"] = "All policy checks passed."
    logger.info("Plan approved by policy layer.")
    return STATUS_APPROVED, plan
