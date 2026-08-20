"""Groq-backed judge model for DeepEval.

DeepEval has no native Groq provider — only OpenAI/Azure/Anthropic/Gemini/
Ollama/LiteLLM/Bedrock ship out of the box — so every DeepEval metric used by
rag/evaluation.py (FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric,
ContextualPrecisionMetric, ContextualRecallMetric, GEval) is pointed at a
GroqJudge instance instead of a bare model-name string.

`generate()`/`a_generate()` are DeepEval's only call sites into the judge, so
this is the one place Groq's per-minute rate limits and truncated-JSON replies
get handled, instead of every metric reimplementing it.
"""

from __future__ import annotations

import logging
import os
import re
import time

from deepeval.models import DeepEvalBaseLLM
from groq import Groq

import config

logger = logging.getLogger(__name__)

# Parse Groq's "Please try again in 7m48.72s" / "3.5s" / "120ms" hint.
_RETRY_AFTER_RE = re.compile(r"try again in (?:(\d+)m)?([\d.]+)(ms|s)\b", re.IGNORECASE)


def _retry_after_seconds(message: str) -> float | None:
    m = _RETRY_AFTER_RE.search(message or "")
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    value = float(m.group(2))
    seconds = value / 1000.0 if m.group(3).lower() == "ms" else value
    return minutes * 60 + seconds


class GroqJudge(DeepEvalBaseLLM):
    """A DeepEval judge model that calls Groq under the hood.

    Falls back to GROQ_API_KEY (the generation key, from rag.generator) if
    GROQ_API_KEY_RAGAS isn't set, so this works out of the box with a single key.
    """

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or config.DEEPEVAL_JUDGE_MODEL
        super().__init__(self._model_name)

    def load_model(self) -> Groq:
        key = os.getenv("GROQ_API_KEY_RAGAS") or os.getenv("GROQ_API_KEY")
        if not key or key.startswith("gsk_your"):
            raise RuntimeError(
                "No Groq API key set for DeepEval judge calls. Add GROQ_API_KEY_RAGAS "
                "(or GROQ_API_KEY) to your .env file."
            )
        return Groq(api_key=key)

    def get_model_name(self) -> str:
        return self._model_name

    def _call(self, prompt: str, schema=None, max_tokens: int = 1500) -> str:
        """One judge call, retrying transient per-minute rate limits (honoring
        the API's suggested wait, bounded by DEEPEVAL_MAX_RETRY_WAIT) and
        truncated-JSON replies (doubling the token budget, capped at
        DEEPEVAL_MAX_TOKENS_CAP) — same handling rag/evaluation.py used to do
        for its own hand-rolled judge calls, just centralized here now that
        every metric funnels through this one method.
        """
        current_max_tokens = max_tokens
        kwargs = dict(
            model=self._model_name,
            temperature=0.0,
            max_tokens=current_max_tokens,
            timeout=config.DEEPEVAL_TIMEOUT_S,
            # DEEPEVAL_JUDGE_MODEL is a reasoning model (gpt-oss) — without
            # this it can spend the whole token budget on hidden reasoning
            # and return an empty response (confirmed live), which would
            # otherwise look like a truncated-JSON failure here. See
            # config.REASONING_EFFORT.
            reasoning_effort=config.REASONING_EFFORT,
            messages=[{"role": "user", "content": prompt}],
        )
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(config.DEEPEVAL_MAX_RETRIES + 1):
            try:
                kwargs["max_tokens"] = current_max_tokens
                resp = self.model.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                msg = str(e)
                is_rate_limit = "429" in msg or "rate_limit" in msg.lower()
                is_token_limit = "json_validate_failed" in msg or "max completion tokens" in msg.lower()
                wait = _retry_after_seconds(msg)
                if (
                    is_rate_limit
                    and attempt < config.DEEPEVAL_MAX_RETRIES
                    and wait is not None
                    and wait <= config.DEEPEVAL_MAX_RETRY_WAIT
                ):
                    time.sleep(wait + 0.5)
                    continue
                if (
                    is_token_limit
                    and attempt < config.DEEPEVAL_MAX_RETRIES
                    and current_max_tokens < config.DEEPEVAL_MAX_TOKENS_CAP
                ):
                    current_max_tokens = min(current_max_tokens * 2, config.DEEPEVAL_MAX_TOKENS_CAP)
                    continue
                logger.warning("DeepEval judge call failed", exc_info=True)
                raise

    def generate(self, prompt: str, schema=None):
        raw = self._call(prompt, schema=schema)
        if schema is None:
            return raw
        return schema.model_validate_json(raw)

    async def a_generate(self, prompt: str, schema=None):
        # async_mode=False on every metric (see rag/evaluation.py) means this
        # is never actually awaited concurrently, but DeepEvalBaseLLM requires
        # an implementation regardless.
        return self.generate(prompt, schema=schema)
