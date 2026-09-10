"""
Groq-backed judge/generator for DeepEval, with optional OmniRoute gateway
support for Q&A generation (Synthesizer) only.

DeepEval defaults to OpenAI. This adapter lets every DeepEval feature —
synthesizer and metrics alike — run on the Groq key the rest of the app
already uses, with no OpenAI key involved.

For Q&A generation specifically, if `LLM_GATEWAY_URL` is set and the gateway
is reachable, `build_gateway_llm()` is returned instead of the Groq judge so
OmniRoute's provider-level fallback handles availability. The DeepEval *judge*
(metric scoring) always stays on Groq so scores remain comparable across runs.

The contract that matters: DeepEval hands each step a prompt and, for the
steps that count, a Pydantic `schema` it needs back. Returning a validated
instance of that schema is what lets DeepEval do its arithmetic in Python
instead of trusting a number the model wrote in prose.
"""
import os
import logging

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Opt out of DeepEval's telemetry before the package is imported anywhere.
# This is an evaluation harness handling user documents; nothing about a run
# should leave the machine except the LLM calls we make deliberately.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("ERROR_REPORTING", "NO")

from backend.services.groq_keys import make_client, primary_key, key_pool_size

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Google Gemini configuration (Priority 1 for Q&A Generation)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")

# Nvidia NIM configuration (Priority 2 for Q&A Generation and Metrics Evaluation)
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "moonshotai/kimi-k3")

# OpenRouter Nvidia configuration for metrics evaluation (separate from Q&A generation)
OPENROUTER_EVAL_API_KEY = os.getenv("OPENROUTER_EVAL_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_EVAL_MODEL", "nvidia/nemotron-3.5-lightning:free")

# The judge/generator model. Every DeepEval step parses its reply against a
# schema, so strict JSON adherence matters far more here than raw capability.
#
# This is deliberately ONE constant, not a routed or auto-selected model. Metric
# scores are only comparable between runs when the thing producing them is held
# fixed; a judge that changes underneath you turns a score change into noise of
# unknown origin.
DEEPEVAL_MODEL = os.getenv("DEEPEVAL_MODEL", "openai/gpt-oss-120b")

# Groq's tokens-per-day quota is scoped per model AND per organisation, so a
# different model is a *separate* daily budget — an independent axis from key
# rotation. When every key is exhausted for one model, moving to the next model
# keeps the run alive instead of aborting it.
#
# These are chosen from models Groq actually still serves. The Llama family that
# used to head this list (llama-3.3-70b-versatile) has been REMOVED by Groq, not
# merely deprecated — leaving it here meant the "fallback" was itself a failure.
#
# Every entry must hold strict JSON under load: this adapter enforces the schema
# itself (json_mode + Pydantic validation + a repair retry), but a model that
# trails prose after the object burns the repair attempt on every call.
DEEPEVAL_MODEL_FALLBACKS = [
    m.strip() for m in os.getenv(
        "DEEPEVAL_MODEL_FALLBACKS",
        "",
    ).split(",") if m.strip()
]

# Last-resort judge, used only when every Groq model above is exhausted. Off by
# default: see the measurement note in `_gateway_judge_model` before enabling.
GATEWAY_JUDGE_MODEL = os.getenv("GATEWAY_JUDGE_MODEL", "").strip()


def _gateway_judge_model() -> str:
    """Judge model to try on the gateway once Groq is exhausted, or "".

    Measured against OmniRoute 3.8.49 on its keyless free tier, at the serial
    pace a judging run actually calls at (~6s between requests):

      * `auto/*` needs NO provider key. 10/10 short and 8/8 long-context
        (~8.8k char) schema-shaped judge calls returned valid, correctly shaped
        JSON. Bursting the same calls back-to-back does trip 429s, but a serial
        run never bursts.
      * A *pinned* free model (e.g. `oc/deepseek-v4-flash-free`) is NOT usable:
        pinning defeats the routing that would fall back, so it 429s almost
        immediately — 0/10 even when paced. Use an `auto/*` alias here.
      * Latency is 7-10s per call versus well under a second on Groq, and a run
        makes roughly ten judge calls per QA pair.

    So this stays a last resort: it reliably finishes a run that would otherwise
    abort, but as a primary judge it is far slower than Groq and its `auto/`
    alias offers no contractual guarantee of resolving to the same model twice
    (observed: it held `hy3-free` across all 8 long-context calls). Whatever
    actually serves the call is recorded via `_note_model_fallback`, so a drift
    shows up in the run's notes instead of silently moving the scores.
    """
    return GATEWAY_JUDGE_MODEL


def _model_chain(primary: str) -> list[str]:
    chain, seen = [], set()
    for m in [primary, *DEEPEVAL_MODEL_FALLBACKS]:
        if m and m not in seen:
            seen.add(m)
            chain.append(m)
    return chain


def groq_ready() -> bool:
    return key_pool_size() > 0


def gemini_ready() -> bool:
    """Check if Gemini API key is configured and looks valid."""
    key = os.getenv("GEMINI_API_KEY", "").strip()
    return bool(key and len(key) > 10 and not key.startswith("your_"))


def nvidia_ready() -> bool:
    for idx in range(1, 10):
        var = "NVIDIA_API_KEY" if idx == 1 else f"NVIDIA_API_KEY_{idx}"
        key = os.getenv(var, "").strip()
        if key and not key.startswith("your_"):
            return True
    return False


# Records which substitute models had to serve requests, so a run can report
# that its questions were not all produced by the model that was requested.
_model_fallbacks: dict[str, int] = {}


def _note_model_fallback(model: str) -> None:
    _model_fallbacks[model] = _model_fallbacks.get(model, 0) + 1


def model_fallback_stats() -> dict:
    return dict(_model_fallbacks)


def reset_model_fallback_stats() -> None:
    _model_fallbacks.clear()


def build_groq_judge(model_name: str | None = None):
    """Construct a DeepEvalBaseLLM backed by Groq.

    Imported lazily so that importing this module costs nothing and cannot
    fail if deepeval is absent.
    """
    from deepeval.models import DeepEvalBaseLLM
    from openai import OpenAI
    from pydantic import BaseModel

    resolved = model_name or DEEPEVAL_MODEL

    class UnifiedFallbackLLM(DeepEvalBaseLLM):
        def __init__(self, model: str):
            self.model_name = model
            # Groq Client (auto-rotates keys via make_client)
            self._groq_client = OpenAI(
                api_key=primary_key(),
                base_url=GROQ_BASE_URL,
                http_client=make_client(timeout=120.0),
                max_retries=0,
            )
            
            # Nvidia NIM Clients (Priority 1)
            self._nvidia_slots = []
            k1 = os.getenv("NVIDIA_API_KEY", "").strip()
            if k1 and not k1.startswith("your_"):
                self._nvidia_slots.append({
                    "client": OpenAI(
                        api_key=k1,
                        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip(),
                        max_retries=2, timeout=120.0
                    ),
                    "model": os.getenv("NVIDIA_MODEL", "moonshotai/kimi-k3").strip(),
                })
            for idx in range(2, 10):
                k = os.getenv(f"NVIDIA_API_KEY_{idx}", "").strip()
                if k and not k.startswith("your_"):
                    default_m = "deepseek/deepseek-v4-pro-0813" if idx == 2 else "moonshotai/kimi-k3"
                    self._nvidia_slots.append({
                        "client": OpenAI(
                            api_key=k,
                            base_url=os.getenv(f"NVIDIA_BASE_URL_{idx}", os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")).strip(),
                            max_retries=2, timeout=120.0
                        ),
                        "model": os.getenv(f"NVIDIA_MODEL_{idx}", default_m).strip(),
                    })

            # OR1 Client
            or1_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            self._or1_client = None
            if or1_key:
                self._or1_client = OpenAI(
                    api_key=or1_key,
                    base_url="https://openrouter.ai/api/v1",
                    max_retries=0, timeout=120.0
                )
            self._or1_model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free").strip()
            
            # OR2 Client
            or2_key = os.getenv("OPENROUTER_API_KEY_2", "").strip()
            self._or2_client = None
            if or2_key:
                self._or2_client = OpenAI(
                    api_key=or2_key,
                    base_url="https://openrouter.ai/api/v1",
                    max_retries=0, timeout=120.0
                )
            self._or2_model = os.getenv("OPENROUTER_MODEL_2", "google/gemma-4-26b-a4b-it:free").strip()
            
            # AIML API Client (stealth/ox-alpha)
            aiml_key = os.getenv("AIMLAPI_KEY", "").strip()
            self._aiml_client = None
            if aiml_key:
                self._aiml_client = OpenAI(
                    api_key=aiml_key,
                    base_url=os.getenv("AIMLAPI_BASE_URL", "https://api.aimlapi.com/v1").strip(),
                    max_retries=0, timeout=120.0
                )
            self._aiml_model = os.getenv("AIMLAPI_MODEL", "stealth/ox-alpha").strip()
            
            super().__init__(model)

        def load_model(self):
            return self._groq_client

        def get_model_name(self) -> str:
            return f"Unified/{self.model_name}"

        def _call(self, client, model, messages, json_mode, max_tokens, use_reasoning=False) -> str:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if use_reasoning:
                kwargs["reasoning"] = {"enabled": True}
                
            res = client.chat.completions.create(**kwargs)
            served = getattr(res, "model", None) or model
            if served.split("/")[-1] != self.model_name.split("/")[-1]:
                _note_model_fallback(served)
            return res.choices[0].message.content or ""

        def _chat(self, messages, json_mode: bool, max_tokens: int) -> str:
            from openai import RateLimitError, NotFoundError
            import time

            last: Exception | None = None
            
            # 1. Try Nvidia NIM slots (Priority 1)
            for slot_idx, slot in enumerate(self._nvidia_slots):
                logger.info("Attempting Nvidia NIM API (Priority 1, slot #%d) with model %s", slot_idx + 1, slot["model"])
                try:
                    return self._call(slot["client"], slot["model"], messages, json_mode, max_tokens)
                except Exception as e:
                    last = e
                    logger.warning("Nvidia NIM API model '%s' failed: %s", slot["model"], str(e)[:100])

            # 2. Try Groq keys
            for attempt in range(3):
                try:
                    return self._call(self._groq_client, self.model_name, messages, json_mode, max_tokens)
                except NotFoundError as e:
                    logger.warning("Model '%s' not found on Groq.", self.model_name)
                    last = e
                    break
                except Exception as e:
                    last = e
                    logger.warning("Groq rate-limited/failed: %s", str(e)[:100])
                    
            # 2. Try AIML API (stealth/ox-alpha) if key is provided
            if self._aiml_client:
                logger.info("Attempting AIML API with model %s", self._aiml_model)
                try:
                    return self._call(self._aiml_client, self._aiml_model, messages, json_mode, max_tokens)
                except Exception as e:
                    last = e
                    logger.warning("AIML API model '%s' failed: %s", self._aiml_model, str(e)[:100])

            # 3. Try OpenRouter 1
            if self._or1_client:
                logger.info("Falling back to OpenRouter 1 with model %s", self._or1_model)
                try:
                    return self._call(self._or1_client, self._or1_model, messages, json_mode, max_tokens)
                except Exception as e:
                    last = e
                    logger.warning("OpenRouter 1 failed: %s", str(e)[:100])
                    
            # 4. Try OpenRouter 2 (configured model, e.g. Gemma or Nemotron reasoning)
            if self._or2_client:
                models_to_try = [self._or2_model]
                if "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free" not in models_to_try:
                    models_to_try.append("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free")
                
                for or_m in models_to_try:
                    logger.info("Attempting OpenRouter 2 with model %s", or_m)
                    try:
                        return self._call(self._or2_client, or_m, messages, json_mode, max_tokens, use_reasoning=True)
                    except Exception as e:
                        last = e
                        logger.warning("OpenRouter 2 model '%s' failed: %s", or_m, str(e)[:100])
                    
            raise last if last else RuntimeError("No models available in unified fallback chain")

        def generate(self, prompt: str, schema: type[BaseModel] | None = None):
            if schema is None:
                return self._chat(
                    [{"role": "user", "content": prompt}],
                    json_mode=False, max_tokens=2048,
                )

            instruction = (
                f"{prompt}\n\n"
                "CRITICAL INSTRUCTIONS:\n"
                "1. You must generate real data and fill in the required fields.\n"
                "2. DO NOT output the schema definitions or '$defs'.\n"
                "3. Respond with a single valid JSON object and nothing else. It must validate against this JSON schema:\n"
                f"{schema.model_json_schema()}"
            )
            messages = [{"role": "user", "content": instruction}]
            
            use_json_mode = True
            for attempt in range(2):
                try:
                    raw = self._chat(messages, json_mode=use_json_mode, max_tokens=3000)
                except Exception as e:
                    import openai
                    if isinstance(e, openai.BadRequestError) and "json_validate_failed" in str(e):
                        if attempt:
                            raise
                        use_json_mode = False
                        messages = messages + [{"role": "user", "content": "Please ensure your output is valid JSON."}]
                        continue
                    raise

                try:
                    import re
                    match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
                    if match:
                        raw = match.group(1)
                    return schema.model_validate_json(raw)
                except Exception as e:
                    if attempt:
                        logger.warning("Schema validation failed twice: %s", e)
                        raise
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": (
                            f"That JSON failed schema validation: {e}. "
                            "Return the corrected JSON object only."
                        )},
                    ]

        async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
            return self.generate(prompt, schema=schema)

    return UnifiedFallbackLLM(resolved)


# ─────────────────────────────────────────────────────────────────────────────
# OmniRoute / gateway LLM — for Q&A generation ONLY
# ─────────────────────────────────────────────────────────────────────────────
# The gateway is NOT used for the DeepEval judge (metric scoring). Metric
# scores are only comparable when the judge is held fixed; a gateway that
# silently swaps providers would make that invisible. Q&A generation has no
# such constraint — it just needs capable LLM calls, and the gateway adds
# provider-level fallback on top of key rotation.

# Model the gateway should use for Q&A generation. Defaults to `auto/best`
# which lets OmniRoute pick the best available provider.
# Set GATEWAY_GENERATOR_MODEL to a specific id to pin it (e.g. `oc/deepseek-v4-flash`).
GATEWAY_GENERATOR_MODEL = os.getenv("GATEWAY_GENERATOR_MODEL", "auto/best").strip()


def build_gateway_llm(model: str | None = None):
    """Return a DeepEvalBaseLLM backed by the configured gateway (OmniRoute).

    Raises RuntimeError when no gateway URL is set or the gateway is offline,
    so call sites can fall back to build_groq_judge() gracefully.

    Use ONLY for Q&A synthesis (Synthesizer), never for the DeepEval
    judge / metric scoring — see module docstring.
    """
    from backend.services.llm_endpoint import resolve_endpoint_sync

    ep = resolve_endpoint_sync()
    if not ep.via_gateway:
        raise RuntimeError(
            "No LLM gateway configured or gateway is offline — "
            "set LLM_GATEWAY_URL in your .env to use OmniRoute for Q&A generation."
        )

    from deepeval.models import DeepEvalBaseLLM
    from openai import OpenAI
    from pydantic import BaseModel

    resolved_model = model or GATEWAY_GENERATOR_MODEL
    # Capture ep in the closure so the class does not need to re-resolve it.
    _ep = ep

    class GatewayGeneratorLLM(DeepEvalBaseLLM):
        """DeepEval LLM backed by OmniRoute (or any OpenAI-compatible gateway).

        Intended for Synthesizer (Q&A generation) only. JSON schema enforcement
        and a single repair retry mirror the GroqJudge behaviour so the
        Synthesizer works identically regardless of which backend is active.
        """

        def __init__(self):
            self.model_name = resolved_model
            self._client = OpenAI(
                api_key=_ep.token or "unused",  # keyless gateways still need a non-empty string
                base_url=_ep.base_url,
                max_retries=1,
            )
            super().__init__(resolved_model)

        def load_model(self):
            return self._client

        def get_model_name(self) -> str:
            return f"Gateway/{self.model_name}"

        def _call(self, messages: list[dict], json_mode: bool, max_tokens: int) -> str:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "stream": False,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            res = self._client.chat.completions.create(**kwargs)
            served = getattr(res, "model", None) or self.model_name
            if served != self.model_name:
                logger.info("Gateway served Q&A generation with model '%s'", served)
                _note_model_fallback(served)
            return res.choices[0].message.content or ""

        def generate(self, prompt: str, schema=None):
            if schema is None:
                return self._call(
                    [{"role": "user", "content": prompt}],
                    json_mode=False, max_tokens=2000,
                )
            instruction = (
                f"{prompt}\n\n"
                "Respond with a single JSON object and nothing else. It must "
                "validate against this JSON schema:\n"
                f"{schema.model_json_schema()}"
            )
            messages = [{"role": "user", "content": instruction}]
            use_json_mode = True
            for attempt in range(2):
                try:
                    raw = self._call(messages, json_mode=use_json_mode, max_tokens=3000)
                except Exception as e:
                    import openai
                    if isinstance(e, openai.BadRequestError) and "json_validate_failed" in str(e):
                        if attempt:
                            raise
                        use_json_mode = False
                        messages = messages + [{"role": "user", "content": "Please ensure your output is valid JSON."}]
                        continue
                    raise

                try:
                    return schema.model_validate_json(raw)
                except Exception as e:
                    if attempt:
                        logger.warning("Gateway generator schema validation failed twice: %s", e)
                        raise
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": (
                            f"That JSON failed schema validation: {e}. "
                            "Return the corrected JSON object only."
                        )},
                    ]

        async def a_generate(self, prompt: str, schema=None):
            # Synthesizer runs with async_mode=False (serial for rate limits),
            # so this only satisfies the abstract interface.
            return self.generate(prompt, schema=schema)

    return GatewayGeneratorLLM()


def build_generator_llm():
    """Return the best available LLM for Q&A generation (Synthesizer use only).

    Priority:
      1. Google Gemini (GEMINI_API_KEY) — Priority 1, latest Google models
      2. Nvidia NIM (NVIDIA_API_KEY with moonshotai/kimi-k3) — Priority 2
      3. AIHubMix (Premium key) — when PREFER_AIHUBMIX=1 is set
      4. OpenAI (OPENAI_API_KEY) — fast, reliable
      5. Groq direct — fast, reliable, 3 API keys with rotation
      6. OmniRoute gateway — when LLM_GATEWAY_URL is set and gateway responds
      7. OpenRouter Nvidia — only when PREFER_OPENROUTER=1 is set

    NEVER pass the result of this function to the DeepEval judge / metrics.
    For scoring always use build_groq_judge() so runs stay comparable.
    """
    # ── Priority 1: Google Gemini override ───────────────────────────────────
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp").strip()
    # Only try Gemini if key is present and looks valid (not empty, not placeholder)
    if gemini_key and len(gemini_key) > 10 and not gemini_key.startswith("your_"):
        try:
            llm = build_gemini_llm(gemini_key, gemini_model)
            logger.info("Q&A generation: using Google Gemini (Priority 1) with model '%s'", gemini_model)
            return llm
        except RuntimeError as e:
            # Python 3.14 compatibility error or missing package
            logger.warning("Gemini provider not available: %s - Trying Nvidia NIM...", str(e)[:200])
        except Exception as e:
            logger.warning("Gemini provider failed: %s - Trying Nvidia NIM...", str(e)[:100])

    # ── Priority 2: Nvidia NIM override ──────────────────────────────────────
    nvidia_slots = []
    k1 = os.getenv("NVIDIA_API_KEY", "").strip()
    if k1 and not k1.startswith("your_"):
        nvidia_slots.append({
            "key": k1,
            "model": os.getenv("NVIDIA_MODEL", "moonshotai/kimi-k3").strip(),
            "base_url": os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip(),
        })
    for idx in range(2, 10):
        k = os.getenv(f"NVIDIA_API_KEY_{idx}", "").strip()
        if k and not k.startswith("your_"):
            default_m = "deepseek/deepseek-v4-pro-0813" if idx == 2 else "moonshotai/kimi-k3"
            nvidia_slots.append({
                "key": k,
                "model": os.getenv(f"NVIDIA_MODEL_{idx}", default_m).strip(),
                "base_url": os.getenv(f"NVIDIA_BASE_URL_{idx}", os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")).strip(),
            })

    if nvidia_slots:
        llm = build_nvidia_llm(slots=nvidia_slots)
        slot_desc = ", ".join(f"{s['model']} (key #{i+1})" for i, s in enumerate(nvidia_slots))
        logger.info("Q&A generation: using Nvidia NIM (Priority 1) with %d slot(s): %s", len(nvidia_slots), slot_desc)
        return llm

    # ── Premium: AIHubMix override ───────────────────────────────────────────
    if os.getenv("PREFER_AIHUBMIX", "").strip() == "1":
        aihubmix_key = os.getenv("AIHUBMIX_API_KEY", "").strip()
        aihubmix_model = os.getenv("AIHUBMIX_MODEL", "ox-alpha").strip()
        aihubmix_base = os.getenv("AIHUBMIX_BASE_URL", "https://aihubmix.com/v1").strip()
        if aihubmix_key:
            llm = build_aihubmix_llm(aihubmix_key, aihubmix_model, aihubmix_base)
            logger.info("Q&A generation: using AIHubMix with model '%s'", aihubmix_model)
            return llm

    # ── Optional: OpenRouter override ────────────────────────────────────────
    # Set PREFER_OPENROUTER=1 in .env to try OpenRouter first. Off by default
    # because it has been consistently timing out from this network.
    if os.getenv("PREFER_OPENROUTER", "").strip() == "1":
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        openrouter_model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free").strip()
        if openrouter_key and openrouter_key.startswith("sk-"):
            llm = build_openrouter_llm(openrouter_key, openrouter_model)
            logger.info("Q&A generation: using OpenRouter (PREFER_OPENROUTER=1) with model '%s'", openrouter_model)
            return llm

    # ── OpenAI Provider ──────────────────────────────────────────────────────
    # Collect all OPENAI_API_KEY, OPENAI_API_KEY_2 … _9 into a pool for
    # automatic failover on rate-limit (429) or dead-key (401/403) errors.
    _openai_keys: list[str] = []
    for _slot in [""] + [f"_{i}" for i in range(2, 10)]:
        _k = os.getenv(f"OPENAI_API_KEY{_slot}", "").strip()
        if _k and not _k.startswith("your_") and _k not in _openai_keys:
            _openai_keys.append(_k)
    if _openai_keys:
        openai_model = os.getenv("OPENAI_MODEL", "openai/gpt-oss-120b").strip()
        openai_base = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1").strip()
        try:
            llm = build_openai_llm(_openai_keys, openai_model, openai_base)
            logger.info(
                "Q&A generation: using OpenAI provider with model '%s' (%d key(s))",
                openai_model, len(_openai_keys),
            )
            return llm
        except Exception as e:
            logger.warning("OpenAI provider failed: %s - Trying Groq...", str(e)[:100])

    # ── Primary: Groq direct ─────────────────────────────────────────────────
    if groq_ready():
        logger.info("Q&A generation: using Groq direct with %d API key(s)", key_pool_size())
        return build_groq_judge()

    # ── Fallback: gateway ────────────────────────────────────────────────────
    try:
        llm = build_gateway_llm()
        logger.info(
            "Q&A generation: routing through gateway (model: %s)",
            llm.model_name,
        )
        return llm
    except RuntimeError as e:
        logger.info("Q&A generation: gateway unavailable (%s) — using Groq.", str(e)[:100])
        return build_groq_judge()



def build_aihubmix_llm(api_key: str, model_name: str, base_url: str):
    """Build a DeepEval-compatible LLM wrapper for AIHubMix."""
    from deepeval.models import DeepEvalBaseLLM
    
    class AIHubMixLLM(DeepEvalBaseLLM):
        def __init__(self, api_key: str, model: str, base_url: str):
            self._aihubmix_model = model
            self.api_key = api_key
            self.base_url = base_url
            self._client = None
            super().__init__(model=model)
        
        def load_model(self):
            return None
            
        def get_model_name(self) -> str:
            return self._aihubmix_model
        
        def _get_client(self):
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_retries=2,
                    timeout=120.0,
                )
            return self._client
        
        def generate(self, prompt: str, schema: type = None):
            client = self._get_client()
            model_to_use = self._aihubmix_model or getattr(self, 'model', None) or "ox-alpha"
            
            if schema is not None:
                prompt += f"\n\nRespond with a single JSON object and nothing else. It must validate against this JSON schema:\n{schema.model_json_schema()}"
                
            logger.info(f"AIHubMix sync generate using model: {model_to_use}")
            response = client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content or ""
            
            if schema is not None:
                try:
                    import re
                    match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
                    if match:
                        raw = match.group(1)
                    return schema.model_validate_json(raw)
                except Exception as e:
                    logger.error(f"Failed to parse schema in AIHubMix generate: {e}")
                    raise
            return raw
        
        async def a_generate(self, prompt: str, schema: type = None):
            from openai import AsyncOpenAI
            import asyncio
            
            model_to_use = self._aihubmix_model or getattr(self, 'model', None) or "ox-alpha"
            
            if schema is not None:
                prompt += f"\n\nRespond with a single JSON object and nothing else. It must validate against this JSON schema:\n{schema.model_json_schema()}"
                
            logger.info(f"Calling AIHubMix with model={model_to_use}, prompt_len={len(prompt)}")
            
            async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                max_retries=2,
                timeout=120.0,
            )
            
            response = await async_client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content or ""
            
            if schema is not None:
                try:
                    import re
                    match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
                    if match:
                        raw = match.group(1)
                    return schema.model_validate_json(raw)
                except Exception as e:
                    logger.error(f"Failed to parse schema in AIHubMix a_generate: {e}")
                    raise
            return raw
    
    return AIHubMixLLM(api_key, model_name, base_url)

def build_openrouter_llm(api_key: str, model_name: str):
    """Build a DeepEval-compatible LLM wrapper for OpenRouter Nvidia API."""
    from deepeval.models import DeepEvalBaseLLM
    
    class OxAlphaLLM(DeepEvalBaseLLM):
        def __init__(self, api_key: str, model: str):
            # Store model BEFORE calling super() to prevent it from being overwritten
            self._openrouter_model = model
            self.api_key = api_key
            self._client = None
            # Call super().__init__() LAST and pass the model
            super().__init__(model=model)
        
        def load_model(self):
            """Required by DeepEvalBaseLLM - called during initialization.
            Returns None to avoid serialization issues."""
            return None
        
        def _get_client(self):
            """Lazy client creation to avoid serialization issues."""
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1",
                    max_retries=2,
                    timeout=60.0,
                )
            return self._client
        
        def generate(self, prompt: str, schema: type = None):
            """Synchronous generation - required by DeepEvalBaseLLM.
            
            Tries OpenRouter first; on any failure, falls back to Groq.
            """
            try:
                client = self._get_client()
                # Use stored model name
                model_to_use = self._openrouter_model or getattr(self, 'model', None) or "nvidia/nemotron-3.5-lightning:free"
                
                if schema is not None:
                    prompt += f"\n\nRespond with a single JSON object and nothing else. It must validate against this JSON schema:\n{schema.model_json_schema()}"
                    
                logger.info(f"OpenRouter sync generate using model: {model_to_use}")
                response = client.chat.completions.create(
                    model=model_to_use,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2048,
                )
                raw = response.choices[0].message.content or ""
                
                if schema is not None:
                    try:
                        import re
                        match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
                        if match:
                            raw = match.group(1)
                        return schema.model_validate_json(raw)
                    except Exception as e:
                        logger.error(f"Failed to parse schema in OpenRouter generate: {e}")
                        raise
                return raw
            except Exception as e:
                logger.warning(
                    "OpenRouter sync generation failed (%s: %s) — falling back to Groq",
                    type(e).__name__, str(e)[:120],
                )
                try:
                    groq_llm = build_groq_judge()
                    result = groq_llm.generate(prompt, schema)
                    return result
                except Exception as groq_err:
                    logger.error(
                        "Groq fallback also failed: %s: %s",
                        type(groq_err).__name__, str(groq_err)[:160],
                    )
                    raise e

        
        async def a_generate(self, prompt: str, schema: type = None):
            """Async generation - required by DeepEvalBaseLLM.
            
            Tries OpenRouter first; on timeout/network/API errors, falls back
            to Groq so the Synthesizer run is not aborted.
            """
            try:
                from openai import AsyncOpenAI
                import asyncio
                
                # Use stored model name with fallback
                model_to_use = self._openrouter_model or getattr(self, 'model', None) or "nvidia/nemotron-3.5-lightning:free"
                
                if schema is not None:
                    prompt += f"\n\nRespond with a single JSON object and nothing else. It must validate against this JSON schema:\n{schema.model_json_schema()}"
                    
                logger.info(f"Calling OpenRouter with model={model_to_use}, prompt_len={len(prompt)}")
                
                # Create client with aggressive timeout for network issues
                async_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1",
                    max_retries=0,  # We'll handle retries ourselves
                    timeout=90.0,  # Reduced from 120 - fail faster on network issues
                )
                
                # Try with timeout - fail fast if network is unreliable
                try:
                    response = await asyncio.wait_for(
                        async_client.chat.completions.create(
                            model=model_to_use,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=2048,
                        ),
                        timeout=90.0  # Match client timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("OpenRouter API call timed out after 90 seconds - network connectivity issue")
                    raise TimeoutError("OpenRouter API timed out")
                except ConnectionError as e:
                    logger.error(f"OpenRouter connection error: {e}")
                    raise
                
                # Check if response is None
                if response is None:
                    logger.error("OpenRouter returned None response - likely network issue")
                    raise ConnectionError("OpenRouter returned None")
                
                logger.info(f"Response type: {type(response)}, has choices: {hasattr(response, 'choices') if response else 'N/A'}")
                
                if not hasattr(response, 'choices') or not response.choices:
                    logger.error(f"Response is missing choices: {response}")
                    raise ValueError(f"Invalid response from OpenRouter: {type(response)}")
                
                raw = response.choices[0].message.content or ""
                logger.info(f"OpenRouter Nvidia async generated {len(raw)} chars")
                
                if schema is not None:
                    try:
                        import re
                        match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
                        if match:
                            raw = match.group(1)
                        return schema.model_validate_json(raw)
                    except Exception as e:
                        logger.error(f"Failed to parse schema in OpenRouter a_generate: {e}")
                        raise
                return raw
                
            except Exception as e:
                # ── Groq fallback ─────────────────────────────────────────────
                # OpenRouter failed (timeout, network, API error). Rather than
                # aborting the entire Synthesizer run, fall back to the Groq
                # judge which has its own key rotation and model chain.
                logger.warning(
                    "OpenRouter generation failed (%s: %s) — falling back to Groq",
                    type(e).__name__, str(e)[:120],
                )
                try:
                    import asyncio
                    groq_llm = build_groq_judge()
                    result = await asyncio.to_thread(groq_llm.generate, prompt, schema)
                    return result
                except Exception as groq_err:
                    logger.error(
                        "Groq fallback also failed: %s: %s",
                        type(groq_err).__name__, str(groq_err)[:160],
                    )
                    # Re-raise the original OpenRouter error
                    raise e

        
        def get_model_name(self) -> str:
            """Required by DeepEvalBaseLLM."""
            return self._openrouter_model or getattr(self, 'model', None) or "nvidia/nemotron-3.5-lightning:free"
    
    return OxAlphaLLM(api_key=api_key, model=model_name)


def build_openai_llm(api_keys, model_name: str, base_url: str):
    """Build a DeepEval-compatible LLM wrapper for OpenAI API (including Groq
    OpenAI-compatible endpoint) with automatic key-pool failover.

    *api_keys* may be a single string or a list of strings.  When multiple keys
    are supplied, a 429 (rate limit) or 401/403 (dead key) on one key triggers
    an immediate retry with the next key — the same pattern Groq rotation uses.
    """
    import time
    from deepeval.models import DeepEvalBaseLLM

    # Normalise to list
    if isinstance(api_keys, str):
        api_keys = [api_keys]

    # Status codes worth trying the next key for.
    _ROTATE_ON = {401, 403, 429}

    class OpenAILLM(DeepEvalBaseLLM):
        def __init__(self, keys: list[str], model: str, base_url: str):
            self._openai_model = model
            self._keys = keys
            self.api_key = keys[0]           # kept for any code that reads it
            self.base_url = base_url
            self._clients: dict[str, object] = {}   # lazy, per-key
            super().__init__(model=model)

        def load_model(self):
            """Required by DeepEvalBaseLLM - called during initialization."""
            return None

        def get_model_name(self) -> str:
            """Required by DeepEvalBaseLLM."""
            return self._openai_model or getattr(self, 'model', None) or "openai/gpt-oss-120b"

        def _get_client(self, key: str):
            """Lazy, per-key client creation."""
            if key not in self._clients:
                from openai import OpenAI
                self._clients[key] = OpenAI(
                    api_key=key,
                    base_url=self.base_url,
                    max_retries=2,
                    timeout=90.0,
                )
            return self._clients[key]

        # ── helpers ───────────────────────────────────────────────────────

        @staticmethod
        def _is_rotatable(exc) -> bool:
            """Return True if the exception signals a rate-limit or dead key."""
            status = getattr(exc, 'status_code', None)
            if status in _ROTATE_ON:
                return True
            # openai.RateLimitError, openai.AuthenticationError
            cls_name = type(exc).__name__
            if cls_name in ('RateLimitError', 'AuthenticationError'):
                return True
            return False

        @staticmethod
        def _parse_response(raw: str, schema):
            if schema is not None:
                import re
                match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
                if match:
                    raw = match.group(1)
                return schema.model_validate_json(raw)
            return raw

        # ── sync generate ─────────────────────────────────────────────────

        def generate(self, prompt: str, schema: type = None):
            """Synchronous generation with key-pool rotation."""
            model_to_use = self._openai_model or getattr(self, 'model', None) or "openai/gpt-oss-120b"
            full_prompt = prompt
            if schema is not None:
                full_prompt += (
                    f"\n\nRespond with a single JSON object and nothing else. "
                    f"It must validate against this JSON schema:\n"
                    f"{schema.model_json_schema()}"
                )

            last_exc = None
            for attempt, key in enumerate(self._keys):
                try:
                    client = self._get_client(key)
                    logger.info(
                        "OpenAI provider sync generate using model: %s (key #%d/%d)",
                        model_to_use, attempt + 1, len(self._keys),
                    )
                    response = client.chat.completions.create(
                        model=model_to_use,
                        messages=[{"role": "user", "content": full_prompt}],
                        temperature=0.7,
                        max_tokens=2048,
                    )
                    raw = response.choices[0].message.content or ""
                    return self._parse_response(raw, schema)
                except Exception as e:
                    last_exc = e
                    if self._is_rotatable(e) and attempt + 1 < len(self._keys):
                        logger.warning(
                            "OpenAI key #%d hit %s — rotating to key #%d",
                            attempt + 1, type(e).__name__, attempt + 2,
                        )
                        time.sleep(0.5)
                        continue
                    break  # non-rotatable error or last key exhausted

            # All OpenAI keys exhausted — fall back to Groq
            logger.warning(
                "OpenAI provider sync generation failed (%s: %s) — falling back to Groq",
                type(last_exc).__name__, str(last_exc)[:120],
            )
            try:
                groq_llm = build_groq_judge()
                return groq_llm.generate(prompt, schema)
            except Exception as groq_err:
                logger.error(
                    "Groq fallback also failed: %s: %s",
                    type(groq_err).__name__, str(groq_err)[:160],
                )
                raise last_exc

        # ── async generate ────────────────────────────────────────────────

        async def a_generate(self, prompt: str, schema: type = None):
            """Async generation with key-pool rotation."""
            from openai import AsyncOpenAI
            import asyncio

            model_to_use = self._openai_model or getattr(self, 'model', None) or "openai/gpt-oss-120b"
            full_prompt = prompt
            if schema is not None:
                full_prompt += (
                    f"\n\nRespond with a single JSON object and nothing else. "
                    f"It must validate against this JSON schema:\n"
                    f"{schema.model_json_schema()}"
                )

            last_exc = None
            for attempt, key in enumerate(self._keys):
                try:
                    logger.info(
                        "OpenAI provider async generate using model=%s (key #%d/%d), prompt_len=%d",
                        model_to_use, attempt + 1, len(self._keys), len(full_prompt),
                    )
                    async_client = AsyncOpenAI(
                        api_key=key,
                        base_url=self.base_url,
                        max_retries=2,
                        timeout=90.0,
                    )
                    response = await asyncio.wait_for(
                        async_client.chat.completions.create(
                            model=model_to_use,
                            messages=[{"role": "user", "content": full_prompt}],
                            temperature=0.7,
                            max_tokens=2048,
                        ),
                        timeout=90.0,
                    )
                    if response is None or not hasattr(response, 'choices') or not response.choices:
                        raise ValueError(f"Invalid response from OpenAI provider: {type(response)}")

                    raw = response.choices[0].message.content or ""
                    logger.info("OpenAI provider async generated %d chars", len(raw))
                    return self._parse_response(raw, schema)
                except Exception as e:
                    last_exc = e
                    if self._is_rotatable(e) and attempt + 1 < len(self._keys):
                        logger.warning(
                            "OpenAI key #%d hit %s — rotating to key #%d",
                            attempt + 1, type(e).__name__, attempt + 2,
                        )
                        await asyncio.sleep(0.5)
                        continue
                    break

            # All OpenAI keys exhausted — fall back to Groq
            logger.warning(
                "OpenAI provider generation failed (%s: %s) — falling back to Groq",
                type(last_exc).__name__, str(last_exc)[:120],
            )
            try:
                import asyncio as _aio
                groq_llm = build_groq_judge()
                return await _aio.to_thread(groq_llm.generate, prompt, schema)
            except Exception as groq_err:
                logger.error(
                    "Groq fallback also failed: %s: %s",
                    type(groq_err).__name__, str(groq_err)[:160],
                )
                raise last_exc

    return OpenAILLM(keys=api_keys, model=model_name, base_url=base_url)


def build_nvidia_llm(
    api_key: str | list | None = None,
    model_name: str | None = None,
    base_url: str = "https://integrate.api.nvidia.com/v1",
    slots: list[dict] | None = None,
):
    """Build a DeepEval-compatible LLM wrapper for Nvidia NIM with multi-slot key/model failover.

    Supports multiple key+model combinations (e.g. key 1 -> moonshotai/kimi-k3,
    key 2 -> deepseek/deepseek-v4-pro-0813). On rate limit (429) or error, automatically
    fails over to the next configured slot, and only falls back to Groq if all Nvidia slots fail.
    """
    import time
    from deepeval.models import DeepEvalBaseLLM

    # Normalise slots
    normalized_slots: list[dict] = []
    if slots:
        normalized_slots = list(slots)
    elif isinstance(api_key, list):
        for item in api_key:
            if isinstance(item, dict):
                normalized_slots.append(item)
            elif isinstance(item, str) and item.strip():
                normalized_slots.append({
                    "key": item.strip(),
                    "model": model_name or "moonshotai/kimi-k3",
                    "base_url": base_url or "https://integrate.api.nvidia.com/v1",
                })
    elif isinstance(api_key, str) and api_key.strip():
        normalized_slots.append({
            "key": api_key.strip(),
            "model": model_name or "moonshotai/kimi-k3",
            "base_url": base_url or "https://integrate.api.nvidia.com/v1",
        })

    if not normalized_slots:
        normalized_slots = [{
            "key": "",
            "model": model_name or "moonshotai/kimi-k3",
            "base_url": base_url or "https://integrate.api.nvidia.com/v1",
        }]

    class NvidiaLLM(DeepEvalBaseLLM):
        def __init__(self, slots: list[dict]):
            self._slots = slots
            self._primary = slots[0]
            self._nvidia_model = self._primary["model"]
            self.api_key = self._primary["key"]
            self.base_url = self._primary["base_url"]
            self._clients: dict[str, object] = {}
            super().__init__(model=self._nvidia_model)

        def load_model(self):
            """Required by DeepEvalBaseLLM - called during initialization."""
            return None

        def get_model_name(self) -> str:
            return self._nvidia_model or getattr(self, 'model', None) or "moonshotai/kimi-k3"

        def _get_client(self, slot: dict):
            """Lazy client creation per key/base_url to avoid serialization issues."""
            ckey = f"{slot['key']}@{slot['base_url']}"
            if ckey not in self._clients:
                from openai import OpenAI
                self._clients[ckey] = OpenAI(
                    api_key=slot["key"],
                    base_url=slot["base_url"],
                    max_retries=2,
                    timeout=90.0,
                )
            return self._clients[ckey]

        @staticmethod
        def _parse_response(raw: str, schema):
            if schema is not None:
                import re
                match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
                if match:
                    raw = match.group(1)
                return schema.model_validate_json(raw)
            return raw

        def generate(self, prompt: str, schema: type = None):
            """Synchronous generation with multi-slot failover."""
            full_prompt = prompt
            if schema is not None:
                full_prompt += (
                    f"\n\nRespond with a single JSON object and nothing else. "
                    f"It must validate against this JSON schema:\n"
                    f"{schema.model_json_schema()}"
                )

            last_exc = None
            for attempt, slot in enumerate(self._slots):
                model_to_use = slot.get("model") or "moonshotai/kimi-k3"
                try:
                    client = self._get_client(slot)
                    logger.info(
                        "Nvidia NIM sync generate using model: %s (slot #%d/%d)",
                        model_to_use, attempt + 1, len(self._slots),
                    )
                    response = client.chat.completions.create(
                        model=model_to_use,
                        messages=[{"role": "user", "content": full_prompt}],
                        temperature=0.7,
                        max_tokens=2048,
                    )
                    raw = response.choices[0].message.content or ""
                    return self._parse_response(raw, schema)
                except Exception as e:
                    last_exc = e
                    if attempt + 1 < len(self._slots):
                        next_model = self._slots[attempt + 1].get("model", "unknown")
                        logger.warning(
                            "Nvidia slot #%d (%s) hit %s — rotating to slot #%d (%s)",
                            attempt + 1, model_to_use, type(e).__name__, attempt + 2, next_model,
                        )
                        time.sleep(0.5)
                        continue
                    break

            logger.warning(
                "Nvidia sync generation failed across all %d slot(s) (%s: %s) — falling back to Groq/Unified judge",
                len(self._slots), type(last_exc).__name__, str(last_exc)[:120],
            )
            try:
                groq_llm = build_groq_judge()
                return groq_llm.generate(prompt, schema)
            except Exception as fallback_err:
                logger.error(
                    "Groq fallback also failed: %s: %s",
                    type(fallback_err).__name__, str(fallback_err)[:160],
                )
                raise last_exc

        async def a_generate(self, prompt: str, schema: type = None):
            """Async generation with multi-slot failover."""
            from openai import AsyncOpenAI
            import asyncio

            full_prompt = prompt
            if schema is not None:
                full_prompt += (
                    f"\n\nRespond with a single JSON object and nothing else. "
                    f"It must validate against this JSON schema:\n"
                    f"{schema.model_json_schema()}"
                )

            last_exc = None
            for attempt, slot in enumerate(self._slots):
                model_to_use = slot.get("model") or "moonshotai/kimi-k3"
                try:
                    logger.info(
                        "Calling Nvidia NIM with model=%s (slot #%d/%d), prompt_len=%d",
                        model_to_use, attempt + 1, len(self._slots), len(full_prompt),
                    )
                    async_client = AsyncOpenAI(
                        api_key=slot["key"],
                        base_url=slot["base_url"],
                        max_retries=2,
                        timeout=90.0,
                    )
                    response = await asyncio.wait_for(
                        async_client.chat.completions.create(
                            model=model_to_use,
                            messages=[{"role": "user", "content": full_prompt}],
                            temperature=0.7,
                            max_tokens=2048,
                        ),
                        timeout=90.0,
                    )
                    if response is None or not hasattr(response, 'choices') or not response.choices:
                        raise ValueError(f"Invalid response from Nvidia NIM: {type(response)}")

                    raw = response.choices[0].message.content or ""
                    return self._parse_response(raw, schema)
                except Exception as e:
                    last_exc = e
                    if attempt + 1 < len(self._slots):
                        next_model = self._slots[attempt + 1].get("model", "unknown")
                        logger.warning(
                            "Nvidia slot #%d (%s) hit %s — rotating to slot #%d (%s)",
                            attempt + 1, model_to_use, type(e).__name__, attempt + 2, next_model,
                        )
                        await asyncio.sleep(0.5)
                        continue
                    break

            logger.warning(
                "Nvidia async generation failed across all %d slot(s) (%s: %s) — falling back to Groq/Unified judge",
                len(self._slots), type(last_exc).__name__, str(last_exc)[:120],
            )
            try:
                groq_llm = build_groq_judge()
                return await asyncio.to_thread(groq_llm.generate, prompt, schema)
            except Exception as fallback_err:
                logger.error(
                    "Groq fallback also failed: %s: %s",
                    type(fallback_err).__name__, str(fallback_err)[:160],
                )
                raise last_exc

    return NvidiaLLM(normalized_slots)


def build_gemini_llm(api_key: str, model_name: str):
    """Build a DeepEval-compatible LLM wrapper for Google Gemini API.
    
    Uses the google-generativeai Python SDK to call Gemini models.
    Supports schema-based generation for DeepEval's structured outputs.
    
    NOTE: Requires Python < 3.14 due to protobuf compatibility.
    """
    import time
    from deepeval.models import DeepEvalBaseLLM
    
    # Check Python version compatibility
    import sys
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "Google Generative AI SDK is not compatible with Python 3.14+. "
            "Please use Python 3.12 or earlier, or use alternative providers (Nvidia NIM, OpenAI, Groq)."
        )
    
    class GeminiLLM(DeepEvalBaseLLM):
        def __init__(self, api_key: str, model: str):
            self._gemini_model = model
            self.api_key = api_key
            self._model = None
            super().__init__(model=model)
        
        def load_model(self):
            """Required by DeepEvalBaseLLM - called during initialization."""
            return None
        
        def get_model_name(self) -> str:
            """Required by DeepEvalBaseLLM."""
            return self._gemini_model or getattr(self, 'model', None) or "gemini-2.0-flash-exp"
        
        def _get_model(self):
            """Lazy model initialization to avoid serialization issues."""
            if self._model is None:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    self._model = genai.GenerativeModel(self._gemini_model)
                except ImportError:
                    raise RuntimeError(
                        "google-generativeai package not installed. "
                        "Install it with: pip install google-generativeai"
                    )
            return self._model
        
        @staticmethod
        def _parse_response(raw: str, schema):
            """Parse and validate response against schema if provided."""
            if schema is not None:
                import re
                # Remove markdown code blocks if present
                match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
                if match:
                    raw = match.group(1)
                return schema.model_validate_json(raw)
            return raw
        
        def generate(self, prompt: str, schema: type = None):
            """Synchronous generation with Gemini API."""
            model = self._get_model()
            model_to_use = self._gemini_model or "gemini-2.0-flash-exp"
            
            full_prompt = prompt
            if schema is not None:
                full_prompt += (
                    f"\n\nIMPORTANT: Respond with a single valid JSON object and nothing else. "
                    f"The JSON must validate against this schema:\n"
                    f"{schema.model_json_schema()}\n\n"
                    f"Do not include any markdown formatting, explanations, or additional text. "
                    f"Only return the JSON object."
                )
            
            try:
                logger.info("Gemini sync generate using model: %s", model_to_use)
                
                # Configure generation parameters
                generation_config = {
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                }
                
                # For schema-based generation, request JSON output
                if schema is not None:
                    generation_config["response_mime_type"] = "application/json"
                
                response = model.generate_content(
                    full_prompt,
                    generation_config=generation_config,
                )
                
                raw = response.text
                logger.info("Gemini generated %d chars", len(raw))
                return self._parse_response(raw, schema)
                
            except Exception as e:
                logger.error("Gemini generation failed: %s: %s", type(e).__name__, str(e)[:120])
                raise
        
        async def a_generate(self, prompt: str, schema: type = None):
            """Async generation with Gemini API."""
            import asyncio
            
            model = self._get_model()
            model_to_use = self._gemini_model or "gemini-2.0-flash-exp"
            
            full_prompt = prompt
            if schema is not None:
                full_prompt += (
                    f"\n\nIMPORTANT: Respond with a single valid JSON object and nothing else. "
                    f"The JSON must validate against this schema:\n"
                    f"{schema.model_json_schema()}\n\n"
                    f"Do not include any markdown formatting, explanations, or additional text. "
                    f"Only return the JSON object."
                )
            
            try:
                logger.info(
                    "Calling Gemini with model=%s, prompt_len=%d",
                    model_to_use, len(full_prompt),
                )
                
                # Configure generation parameters
                generation_config = {
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                }
                
                # For schema-based generation, request JSON output
                if schema is not None:
                    generation_config["response_mime_type"] = "application/json"
                
                # Run in thread pool to avoid blocking
                response = await asyncio.to_thread(
                    model.generate_content,
                    full_prompt,
                    generation_config=generation_config,
                )
                
                raw = response.text
                logger.info("Gemini async generated %d chars", len(raw))
                return self._parse_response(raw, schema)
                
            except Exception as e:
                logger.error("Gemini async generation failed: %s: %s", type(e).__name__, str(e)[:120])
                raise
    
    return GeminiLLM(api_key=api_key, model=model_name)
