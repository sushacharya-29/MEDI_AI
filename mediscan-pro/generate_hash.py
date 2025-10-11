# =====================================================================
# FILE: generate_hash.py
# Description: Generates a secure hash for your API key
# =====================================================================

import os
import sys
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.utils.security import security  # import your hashing function

# Load .env
load_dotenv()

# Get your actual API key from .env
api_key = os.getenv("GROK_API_KEY")

if not api_key:
    print("❌ GROK_API_KEY not found in .env")
    sys.exit(1)

# Generate hashed key
hashed_key = security.hash_api_key(api_key)

print("🔑 GROK_API_KEY:", api_key)
print("✅ Hashed API Key:", hashed_key)
print("\n👉 Copy this value into your .env as:")
print(f"API_KEY_HASH={hashed_key}")
