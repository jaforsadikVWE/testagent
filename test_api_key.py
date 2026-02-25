#!/usr/bin/env python3
"""Quick test to verify Groq API key and model availability."""

import json
import sys
from groq import Groq

# Load config
try:
    with open("config.json", "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("ERROR: config.json not found")
    sys.exit(1)

# Get first valid key
keys = [k for k in config.get("groq_api_keys", []) if k and not k.startswith("gsk_YOUR")]
if not keys:
    print("ERROR: No valid API keys in config.json")
    sys.exit(1)

api_key = keys[0]
model = "llama-3.3-70b-versatile"  # Current model

print(f"Testing with:")
print(f"  API Key: {api_key[:20]}...{api_key[-10:]}")
print(f"  Model:   {model}")
print()

try:
    client = Groq(api_key=api_key)
    print("[OK] Groq client initialized successfully")
    
    # Try a simple completion
    print("\nSending test request...")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say 'Hello, Groq API works!'"}],
        max_tokens=100,
        temperature=1,
    )
    
    print(f"Full response: {response}")
    print(f"Response type: {type(response)}")
    
    choice = response.choices[0]
    print(f"\nChoice message: {choice.message}")
    print(f"Message content type: {type(choice.message.content)}")
    print(f"Message content: {repr(choice.message.content)}")
    
    content = choice.message.content
    print(f"\n[OK] API Response: {content}")
    
    if content and content.strip() and content != "{}":
        print("[OK] Response looks valid!")
    else:
        print("[WARN] Warning: Response appears empty or is just '{}'")
    
except Exception as exc:
    import traceback
    print(f"[ERROR] Error: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)

print("\n[OK] API key test completed!")

