"""
Centralized configuration for AccessClass.

Every model name and environment-variable name used anywhere in the project
is defined here ONCE, so nothing is scattered across files (proposal rule #12).
"""

import os

from dotenv import load_dotenv

# Load variables from a local .env file if one exists (does nothing in
# production environments where real env vars are already set).
load_dotenv()

# Quiet down CrewAI's telemetry/tracing console noise before crewai is
# imported anywhere else in the app. These do not disable functionality,
# only background network telemetry calls this project doesn't need.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# Text reasoning (Stage 3: Subject Interpreter) — served via Groq.
# Groq exposes an OpenAI-compatible endpoint, so this is configured as the
# "openai" provider pointed at Groq's base URL rather than a native "groq"
# provider (this CrewAI version has no native Groq integration; see README
# notes shipped with Portion 1 for the full explanation).
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Visual/multimodal reasoning (Stage 4: Explanation Agent) — used for images.
GEMINI_MODEL = "gemini-3.5-flash"

# Text embeddings (Stage 6: Grounded Learning Agent). gemini-embedding-001 is
# the current generally-available embedding model — the newer
# gemini-embedding-2* models have a known SDK batching bug (they silently
# return one embedding regardless of input count), so 001 is the correct,
# safe choice here, not a downgrade.
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # truncated via Matryoshka Representation Learning

# Edge TTS voice for chapter audio (Stage 5).
TTS_VOICE = "en-US-EmmaMultilingualNeural"


# ---------------------------------------------------------------------------
# Environment variable names (single source of truth for the names)
# ---------------------------------------------------------------------------
ENV_GROQ_API_KEY = "GROQ_API_KEY"
ENV_GEMINI_API_KEY = "GEMINI_API_KEY"
ENV_QDRANT_URL = "QDRANT_URL"
ENV_QDRANT_API_KEY = "QDRANT_API_KEY"
ENV_QDRANT_COLLECTION = "QDRANT_COLLECTION"
ENV_SUPABASE_URL = "SUPABASE_URL"
ENV_SUPABASE_KEY = "SUPABASE_KEY"
ENV_SUPABASE_BUCKET = "SUPABASE_BUCKET"

# Vars Portion 1 needs to run.
REQUIRED_FOR_PORTION_1 = [
    ENV_GROQ_API_KEY,
    ENV_SUPABASE_URL,
    ENV_SUPABASE_KEY,
]

# Additional vars Portion 2 needs (Stages 4-6).
REQUIRED_FOR_PORTION_2 = REQUIRED_FOR_PORTION_1 + [
    ENV_GEMINI_API_KEY,
    ENV_QDRANT_URL,
    ENV_QDRANT_API_KEY,
]

DEFAULT_SUPABASE_BUCKET = "lectures"
DEFAULT_QDRANT_COLLECTION = "accessclass_lecture_chunks"


def get_groq_api_key() -> str:
    return os.environ[ENV_GROQ_API_KEY]


def get_gemini_api_key() -> str:
    return os.environ[ENV_GEMINI_API_KEY]


def get_qdrant_url() -> str:
    return os.environ[ENV_QDRANT_URL]


def get_qdrant_api_key() -> str:
    return os.environ[ENV_QDRANT_API_KEY]


def get_qdrant_collection() -> str:
    return os.environ.get(ENV_QDRANT_COLLECTION, DEFAULT_QDRANT_COLLECTION)


def get_supabase_url() -> str:
    return os.environ[ENV_SUPABASE_URL]


def get_supabase_key() -> str:
    return os.environ[ENV_SUPABASE_KEY]


def get_supabase_bucket() -> str:
    return os.environ.get(ENV_SUPABASE_BUCKET, DEFAULT_SUPABASE_BUCKET)


def _validate(required: list[str]) -> None:
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise EnvironmentError(
            "AccessClass cannot start: missing required environment "
            f"variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and fill these in before running the app."
        )


def validate_portion1_env() -> None:
    """Stages 1-3 only. Call this at FastAPI startup."""
    _validate(REQUIRED_FOR_PORTION_1)


def validate_portion2_env() -> None:
    """Stages 4-6 additionally need Gemini + Qdrant credentials."""
    _validate(REQUIRED_FOR_PORTION_2)


# ---------------------------------------------------------------------------
# Subject / content-type vocabularies (used by Stage 3 and by the Pydantic
# models in models/subject.py — defined here so both stay in sync).
# ---------------------------------------------------------------------------
SUPPORTED_SUBJECTS = [
    "programming_fundamentals_cpp",
    "data_structures_algorithms",
    "calculus",
    "ict",
    "oop",
    "database_systems",
    "operating_systems",
    "computer_networks",
    "artificial_intelligence_ml",
    "unclassified",
]

SUPPORTED_CONTENT_TYPES = [
    "text",
    "code",
    "table",
    "equation",
    "image",
]