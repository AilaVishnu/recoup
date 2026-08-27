"""Runtime configuration, loaded once from the environment.

Deliberately thin. Anything that constrains what the agent is *allowed to do*
lives in recoup/policy/rules.py instead, where it can be reviewed as policy
rather than buried among connection strings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The decision model. Opus 5 with adaptive thinking - recovery decisions are
# cheap to make and expensive to get wrong, and the volume that reaches the LLM
# at all is small by design (see recoup/agent/brain.py on the routing split).
DECISION_MODEL = "claude-opus-5"
DECISION_EFFORT = "medium"


def _is_placeholder(value: str) -> bool:
    """True for a value that is obviously a stand-in rather than a credential.

    Fixing .env.example is not enough on its own: an .env copied from an older
    revision, or a key someone meant to paste over and did not, produces the same
    failure. And that failure is nasty out of proportion to its cause - a dummy
    key is indistinguishable from a real one to an SDK, so the setup check
    reports "rejected" rather than "absent", every gateway call fails against
    credentials that were never meant to work, and the headline result comes out
    negative for reasons that have nothing to do with the system being measured.

    Treating these as absent puts the project back on the path it handles well:
    no key means simulate, say so, and carry on.
    """
    v = value.strip().lower()
    if not v:
        return True
    return "xxxx" in v or v in {"changeme", "your-key-here", "todo", "none"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    razorpay_key_id: str = Field(default="", alias="RAZORPAY_KEY_ID")
    razorpay_key_secret: str = Field(default="", alias="RAZORPAY_KEY_SECRET")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    llm_provider: str = Field(default="", alias="RECOUP_LLM_PROVIDER")
    """anthropic | openai. Empty means anthropic.

    The second covers any OpenAI-compatible /chat/completions endpoint - Groq,
    Gemini, OpenRouter, Together, DeepSeek, a local Ollama - via llm_base_url.
    Which one answered is deliberately invisible to the policy engine: a
    proposal from a frontier model and one from a free tier clear the same
    thirteen bounds or neither does.
    """

    llm_base_url: str = Field(default="", alias="RECOUP_LLM_BASE_URL")
    llm_model_override: str = Field(default="", alias="RECOUP_LLM_MODEL")

    db_url: str = Field(default="sqlite:///data/recoup.db", alias="RECOUP_DB_URL")
    seed: int = Field(default=42, alias="RECOUP_SEED")

    dry_run: bool = Field(default=False, alias="RECOUP_DRY_RUN")
    """When true, executors log intended calls instead of hitting Razorpay.

    Useful for replaying the pipeline over a seeded dataset without creating
    hundreds of Payment Links in the test account.
    """

    @field_validator("razorpay_key_id")
    @classmethod
    def _refuse_live_keys(cls, v: str) -> str:
        """Hard stop on live credentials.

        Recoup contacts customers and spends incentive budget. It has no business
        holding a live key, and a project that only ever runs in test mode should
        make that structurally impossible rather than merely intended.
        """
        if v.startswith("rzp_live_"):
            raise ValueError(
                "RAZORPAY_KEY_ID is a LIVE key. Recoup refuses to run against live "
                "credentials - it sends messages and spends money. Use a rzp_test_ key."
            )
        return v

    @property
    def razorpay_configured(self) -> bool:
        return not (
            _is_placeholder(self.razorpay_key_id)
            or _is_placeholder(self.razorpay_key_secret)
        )

    @property
    def anthropic_configured(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def llm_api_key(self) -> str:
        """The key for whichever provider is active, or "" if it is a placeholder."""
        if (self.llm_provider or "anthropic").strip().lower() == "openai":
            key = self.openai_api_key
        else:
            key = self.anthropic_api_key
        return "" if _is_placeholder(key) else key

    @property
    def llm_model(self) -> str:
        """Explicit override, else the sensible default for the active provider.

        A model id is provider-specific, so carrying one default across both
        would guarantee a 404 on whichever provider did not own it.
        """
        if self.llm_model_override:
            return self.llm_model_override
        if (self.llm_provider or "anthropic").strip().lower() == "openai":
            return ""  # adapter picks its own default
        return DECISION_MODEL

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key or self.llm_base_url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
