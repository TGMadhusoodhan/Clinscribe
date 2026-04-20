"""
ClinScribe configuration loader.
All environment variables are loaded here once. Import config everywhere else.
Fails fast with a clear error if required vars are missing.
"""

from dotenv import load_dotenv, find_dotenv
import os

# find_dotenv() searches parent directories — works from any subdirectory
load_dotenv(find_dotenv())

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise EnvironmentError(
        "ANTHROPIC_API_KEY is not set.\n"
        "Copy .env.example to .env and add your Anthropic API key.\n"
        "Get one at: https://console.anthropic.com"
    )

ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENMRS_BASE_URL = os.getenv("OPENMRS_BASE_URL", "http://localhost:8080/openmrs")
OPENMRS_USER = os.getenv("OPENMRS_USER", "admin")
OPENMRS_PASSWORD = os.getenv("OPENMRS_PASSWORD", "Admin123")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "hi")

# Derived — never set directly; always computed from OPENMRS_BASE_URL
FHIR_BASE_URL = f"{OPENMRS_BASE_URL}/ws/fhir2/R4"
