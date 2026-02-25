"""Quick debug script to test raw Groq API response."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.key_manager import KeyManager
import groq

km = KeyManager("config.json")
client = km.make_client()

print("=== Testing llama-4-scout (router model) ===")
try:
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": "You are a router. Respond with JSON only."},
            {"role": "user",   "content": "Route this task: say hello"}
        ],
        temperature=1,
        max_completion_tokens=256,
        top_p=1,
        stream=False,
        stop=None,
    )
    print(f"choices[0].message.content = {repr(resp.choices[0].message.content)}")
    print(f"finish_reason = {resp.choices[0].finish_reason}")
    # Check all fields
    msg = resp.choices[0].message
    print(f"message dir: {[a for a in dir(msg) if not a.startswith('_')]}")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
