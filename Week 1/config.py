"""
Central config for the multi-agent system.
Uses Gemini via its OpenAI-compatible endpoint (same setup you used in the
earlier Gemini-agent exercises).
"""
import os
from openai import OpenAI

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError(
        "Set GEMINI_API_KEY in your environment before running "
        "(e.g. `export GEMINI_API_KEY=your_key_here`). Get a free-tier key "
        "from Google AI Studio."
    )

# Adjust if Google renames/retires this model -- check https://ai.google.dev/models
MODEL_NAME = "gemini-2.0-flash"

client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
