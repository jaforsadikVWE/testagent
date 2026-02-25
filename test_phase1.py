"""
test_phase1.py
────────────────────────────────────────────────────────────
Unit tests for Phase 1: KeyManager and Policy Layer.

All tests use unittest.mock – no real API calls are made.
Run with:  python test_phase1.py
"""

import json
import os
import sys
import hashlib
import unittest
from unittest.mock import patch, MagicMock

# ── Make sure project root is on the path ────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Groq import: use real SDK or offline mock ────────────────
try:
    import groq
except ImportError:
    import mock_groq as groq
    import sys; sys.modules["groq"] = groq


from core.key_manager import KeyManager
from execution.policy import evaluate_plan, compute_payload_hash


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

FAKE_CONFIG = {
    "groq_api_keys": ["gsk_key_alpha", "gsk_key_beta", "gsk_key_gamma"],
    "models": {
        "heavy": "openai/gpt-oss-120b",
        "fast":  "meta-llama/llama-4-scout-17b-16e-instruct"
    },
    "confidence_threshold": 0.5,
    "experimental_override": False,
    "max_retries": 3,
    "request_timeout": 30,
    "memory":  {"history_file": "data/history.json", "max_turns": 15},
    "logging": {"audit_log": "logs/audit.log", "max_bytes": 1048576, "backup_count": 5}
}


def make_valid_plan(commands=None, confidence=0.9, risk="low", root=False):
    """Build a syntactically valid sub-agent plan dict."""
    if commands is None:
        commands = ["echo hello"]
    payload_hash = compute_payload_hash(commands)
    return {
        "agent": "Code",
        "action_type": "script",
        "parameters": {
            "script_type": "bash",
            "commands": commands
        },
        "requires_root": root,
        "confidence_score": confidence,
        "risk_level": risk,
        "dry_run_safe": True,
        "payload_hash": payload_hash
    }


# ══════════════════════════════════════════════════════════════
#  KeyManager Tests
# ══════════════════════════════════════════════════════════════

class TestKeyManager(unittest.TestCase):

    def _make_km(self) -> KeyManager:
        """Create a KeyManager backed by FAKE_CONFIG (no disk I/O)."""
        with patch("builtins.open", unittest.mock.mock_open(
            read_data=json.dumps(FAKE_CONFIG)
        )), patch("os.path.exists", return_value=True):
            return KeyManager("config.json")

    # ── Basic initialisation ──────────────────────────────────

    def test_loads_valid_keys(self):
        km = self._make_km()
        self.assertEqual(km.total_keys(), 3)
        self.assertEqual(km.get_current_key(), "gsk_key_alpha")

    def test_rotate_advances_key(self):
        km = self._make_km()
        new_key = km.rotate_key()
        self.assertEqual(new_key, "gsk_key_beta")
        self.assertEqual(km.get_current_key(), "gsk_key_beta")

    def test_rotate_twice(self):
        km = self._make_km()
        km.rotate_key()
        km.rotate_key()
        self.assertEqual(km.get_current_key(), "gsk_key_gamma")

    def test_rotate_exhausted_returns_none(self):
        km = self._make_km()
        km.rotate_key()
        km.rotate_key()
        result = km.rotate_key()   # no keys left
        self.assertIsNone(result)

    def test_reset_goes_back_to_first(self):
        km = self._make_km()
        km.rotate_key()
        km.reset()
        self.assertEqual(km.get_current_key(), "gsk_key_alpha")

    def test_keys_remaining(self):
        km = self._make_km()
        self.assertEqual(km.keys_remaining(), 2)
        km.rotate_key()
        self.assertEqual(km.keys_remaining(), 1)

    # ── Rotation on simulated RateLimitError ──────────────────

    def test_rotation_on_rate_limit_error(self):
        """
        Simulate the caller pattern:
          try: call API with current key
          except RateLimitError: rotate_key(), retry
        """
        import groq as groq_module

        km = self._make_km()

        call_count = {"n": 0}

        def fake_api_call(key):
            call_count["n"] += 1
            if key == "gsk_key_alpha":
                # First key raises rate limit
                raise groq_module.RateLimitError(
                    "rate limit", response=MagicMock(), body={}
                )
            return "success"   # second key works

        result = None
        for attempt in range(km.total_keys()):
            try:
                result = fake_api_call(km.get_current_key())
                break
            except groq_module.RateLimitError:
                new_key = km.rotate_key()
                if new_key is None:
                    break

        self.assertEqual(result, "success")
        self.assertEqual(km.get_current_key(), "gsk_key_beta")
        self.assertEqual(call_count["n"], 2)

    # ── Edge cases ────────────────────────────────────────────

    def test_missing_config_raises(self):
        with patch("os.path.exists", return_value=False):
            with self.assertRaises(FileNotFoundError):
                KeyManager("nonexistent.json")

    def test_placeholder_keys_filtered(self):
        config = dict(FAKE_CONFIG)
        config["groq_api_keys"] = ["gsk_YOUR_KEY_1_HERE", "gsk_real_key"]
        with patch("builtins.open", unittest.mock.mock_open(
            read_data=json.dumps(config)
        )), patch("os.path.exists", return_value=True):
            km = KeyManager("config.json")
        self.assertEqual(km.total_keys(), 1)
        self.assertEqual(km.get_current_key(), "gsk_real_key")

    def test_all_placeholder_keys_raises(self):
        config = dict(FAKE_CONFIG)
        config["groq_api_keys"] = ["gsk_YOUR_KEY_1_HERE"]
        with patch("builtins.open", unittest.mock.mock_open(
            read_data=json.dumps(config)
        )), patch("os.path.exists", return_value=True):
            with self.assertRaises(ValueError):
                KeyManager("config.json")


# ══════════════════════════════════════════════════════════════
#  Policy Layer Tests
# ══════════════════════════════════════════════════════════════

class TestPolicyLayer(unittest.TestCase):

    def _mock_config(self, overrides=None):
        """Patch load_config to return FAKE_CONFIG (optionally modified)."""
        cfg = dict(FAKE_CONFIG)
        if overrides:
            cfg.update(overrides)
        return patch("execution.policy.load_config", return_value=cfg)

    # ── Approved path ─────────────────────────────────────────

    def test_safe_plan_approved(self):
        plan = make_valid_plan(confidence=0.9, risk="low", root=False)
        with self._mock_config():
            status, result = evaluate_plan(plan)
        self.assertEqual(status, "APPROVED")
        self.assertEqual(result["policy_status"], "APPROVED")

    # ── Low-confidence escalation ─────────────────────────────

    def test_low_confidence_requires_approval(self):
        plan = make_valid_plan(confidence=0.3)   # below 0.5 threshold
        with self._mock_config():
            status, result = evaluate_plan(plan)
        self.assertEqual(status, "APPROVAL_REQUIRED")
        self.assertEqual(result["risk_level"], "high")   # escalated

    def test_low_confidence_with_override_approved(self):
        plan = make_valid_plan(confidence=0.2)
        with self._mock_config({"experimental_override": True}):
            status, result = evaluate_plan(plan)
        # Override allows low confidence through; but still flagged for root check
        # confidence=0.2, risk=low, root=False → should APPROVE when override=True
        self.assertEqual(status, "APPROVED")

    # ── Root / high-risk escalation ───────────────────────────

    def test_root_plan_requires_approval(self):
        plan = make_valid_plan(confidence=0.9, risk="low", root=True)
        with self._mock_config():
            status, result = evaluate_plan(plan)
        self.assertEqual(status, "APPROVAL_REQUIRED")

    def test_high_risk_plan_requires_approval(self):
        plan = make_valid_plan(confidence=0.9, risk="high", root=False)
        with self._mock_config():
            status, result = evaluate_plan(plan)
        self.assertEqual(status, "APPROVAL_REQUIRED")

    # ── Hash tamper detection ─────────────────────────────────

    def test_tampered_hash_denied(self):
        plan = make_valid_plan(confidence=0.9)
        plan["payload_hash"] = "deadbeef" * 8   # wrong hash
        with self._mock_config():
            status, result = evaluate_plan(plan)
        self.assertEqual(status, "DENIED")
        self.assertIn("mismatch", result["policy_reason"].lower())

    # ── Missing fields ────────────────────────────────────────

    def test_missing_field_denied(self):
        plan = make_valid_plan()
        del plan["confidence_score"]
        with self._mock_config():
            status, result = evaluate_plan(plan)
        self.assertEqual(status, "DENIED")

    # ── Hash helper ───────────────────────────────────────────

    def test_compute_payload_hash_deterministic(self):
        cmds = ["ls -la", "echo hello"]
        h1 = compute_payload_hash(cmds)
        h2 = compute_payload_hash(cmds)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 64)   # SHA-256 hex = 64 chars

    def test_compute_payload_hash_sensitive_to_order(self):
        h1 = compute_payload_hash(["a", "b"])
        h2 = compute_payload_hash(["b", "a"])
        self.assertNotEqual(h1, h2)


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
