# Termux AI Agent
### Groq-Powered · Human-in-the-Loop · Multi-Agent · Android Termux

A modular, safety-first AI operating layer for Android Termux.  
Sub-agents plan commands in structured JSON; a policy layer enforces human approval for anything risky; the executor runs them safely.

---

## Quick Start

### 1. Install
```bash
# In Termux (or any Python 3.11+ environment)
pip install groq
```

### 2. Configure
Edit `config.json` and replace the placeholder API keys:
```json
{
  "groq_api_keys": ["gsk_YOUR_REAL_KEY_HERE"]
}
```
Get a free key at https://console.groq.com

### 3. Run
```bash
python main.py
```

---

## Project Structure

```
termux-ai-agent/
├── main.py                  # REPL entry point (Phase 6)
├── config.json              # API keys, thresholds, model config
├── requirements.txt
│
├── core/
│   ├── key_manager.py       # Multi-key rotation (Phase 1)
│   ├── orchestrator.py      # Router + agent dispatcher (Phase 3)
│   ├── memory.py            # Sliding-window persistence (Phase 5)
│   └── watchdog.py          # Immutable audit logger (Phase 5)
│
├── agents/
│   ├── system_prompt.py     # All system prompts (Phase 2)
│   ├── base_agent.py        # Groq API wrapper + retry (Phase 2)
│   └── sub_agents.py        # ReasoningAgent, CodeAgent, … (Phase 2)
│
├── execution/
│   ├── policy.py            # Risk evaluation + hash check (Phase 1)
│   ├── root_guard.py        # su gate with HITL confirmation (Phase 4)
│   └── executor.py          # Command runner (Phase 4)
│
├── data/
│   └── history.json         # Auto-created by MemoryManager
└── logs/
    └── audit.log            # Auto-created by Watchdog
```

---

## Architecture

```
User Input
    │
    ▼
Orchestrator  ──(Router: llama-4-scout)──►  Agent selection
    │
    ▼
Sub-Agent (ReasoningAgent | CodeAgent | KnowledgeAgent | AutomationAgent)
    │  openai/gpt-oss-120b (heavy) or llama-4-scout (fast)
    ▼
Structured JSON Plan
    │
    ▼
Policy Layer  ──(hash check + confidence threshold)──►  APPROVE / DENY / HITL
    │
    ▼
Executor  ──(read_only | standard | root via RootGuard)──►  Result
    │
    ▼
Memory + Watchdog
```

---

## Agents

| Agent | Model | Use case |
|---|---|---|
| ReasoningAgent | `openai/gpt-oss-120b` | Complex multi-step analysis |
| CodeAgent | `openai/gpt-oss-120b` | Writing bash/python scripts |
| KnowledgeAgent | `meta-llama/llama-4-scout-17b-16e-instruct` | Factual Q&A |
| AutomationAgent | `meta-llama/llama-4-scout-17b-16e-instruct` | File management, Termux-API |

---

## Safety Features

- **Multi-key rotation** – automatically switches API keys on rate-limit errors
- **SHA-256 payload hashing** – detects tampering between planning and execution
- **Confidence threshold** – low-confidence plans require human approval
- **HITL root gate** – root commands require explicit `y` confirmation
- **Immutable audit log** – every risky action logged to `logs/audit.log`
- **Crash-proof REPL** – errors are caught and logged; the loop never exits

---

## Running Tests

```bash
python test_phase1.py   # KeyManager + Policy
python test_phase2.py   # BaseAgent + Sub-agents
python test_phase3.py   # Orchestrator routing
python test_phase4.py   # RootGuard + Executor
python test_phase5.py   # Memory + Watchdog
python test_phase6.py   # REPL integration

# Or run all at once
python -m unittest discover -p "test_phase*.py" -v
```

---

## Built-in REPL Commands

| Command | Action |
|---|---|
| `exit` / `quit` | Exit the agent |
| `clear` | Wipe conversation memory |
| `history` | Show recent memory turns |
| `help` | Show usage tips |
| `<any text>` | Send to the AI agent |
