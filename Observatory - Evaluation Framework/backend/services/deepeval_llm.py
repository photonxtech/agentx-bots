"""
Groq-backed judge/generator for DeepEval.

DeepEval defaults to OpenAI. This adapter lets every DeepEval feature —
synthesizer and metrics alike — run on the Groq key the rest of the app
already uses, with no OpenAI key involved.

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
        "qwen/qwen3.6-27b,openai/gpt-oss-20b",
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

    class GroqJudge(DeepEvalBaseLLM):
        def __init__(self, model: str):
            self.model_name = model
            # The injected client rotates to another Groq account on 429/401,
            # so a rate limit on one key does not abort the run.
            self._client = OpenAI(
                api_key=primary_key(),
                base_url=GROQ_BASE_URL,
                http_client=make_client(timeout=120.0),
                max_retries=0,   # rotation already retries; don't double up
            )
            self._gw_client = None
            super().__init__(model)

        def load_model(self):
            return self._client

        def get_model_name(self) -> str:
            return f"Groq/{self.model_name}"

        def _gateway_client(self):
            """Lazily built client for the last-resort gateway tier."""
            if self._gw_client is None:
                from openai import OpenAI
                from backend.services.llm_endpoint import resolve_endpoint_sync

                ep = resolve_endpoint_sync()
                if not ep.via_gateway:
                    return None
                self._gw_client = OpenAI(
                    api_key=ep.token or "unused",   # keyless gateways still need a non-empty value
                    base_url=ep.base_url,
                    max_retries=0,
                )
            return self._gw_client

        def _call(self, client, model, messages, json_mode, max_tokens) -> str:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                # Explicit rather than omitted: some gateway providers read a
                # missing `stream` as streaming and reply with SSE, which the
                # SDK cannot parse into a completion.
                "stream": False,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            res = client.chat.completions.create(**kwargs)

            # Trust what the response says served it, not what we asked for. A
            # gateway may satisfy a request with a different model than the id
            # implies, and an unrecorded judge swap is exactly what makes two
            # runs quietly incomparable.
            served = getattr(res, "model", None) or model
            if served.split("/")[-1] != self.model_name.split("/")[-1]:
                _note_model_fallback(served)
            return res.choices[0].message.content or ""

        def _chat(self, messages, json_mode: bool, max_tokens: int) -> str:
            """Try each Groq model in the chain, then the gateway if configured.

            Key rotation (in the transport) has already been exhausted by the
            time a RateLimitError surfaces here, so the remaining budget is a
            different model, and after that a different provider entirely."""
            from openai import RateLimitError

            last: Exception | None = None
            for model in _model_chain(self.model_name):
                try:
                    out = self._call(self._client, model, messages, json_mode, max_tokens)
                    if model != self.model_name:
                        logger.warning(
                            "Model '%s' rate-limited on every key — served by '%s' instead",
                            self.model_name, model,
                        )
                    return out
                except RateLimitError as e:
                    last = e
                    logger.warning("Model '%s' rate-limited on all keys: %s",
                                   model, str(e)[:160])
                    continue

            gw_model = _gateway_judge_model()
            if gw_model:
                client = self._gateway_client()
                if client is not None:
                    logger.warning(
                        "Every Groq judge model is exhausted — falling back to gateway "
                        "model '%s'. Scores from this run are not strictly comparable "
                        "with runs judged on Groq.", gw_model,
                    )
                    try:
                        return self._call(client, gw_model, messages, json_mode, max_tokens)
                    except Exception as e:
                        logger.error("Gateway judge '%s' also failed: %s", gw_model, str(e)[:160])
                        last = e

            raise last if last else RuntimeError("No model available")

        def generate(self, prompt: str, schema: type[BaseModel] | None = None):
            if schema is None:
                return self._chat(
                    [{"role": "user", "content": prompt}],
                    json_mode=False, max_tokens=2000,
                )

            # Groq's JSON mode guarantees syntactically valid JSON but not the
            # right shape, so the shape is stated in the prompt and enforced by
            # Pydantic on the way back. One repair attempt: a near-miss is
            # cheap to fix and expensive to throw away — and in a synthesizer
            # run, one unrecoverable step discards everything before it.
            instruction = (
                f"{prompt}\n\n"
                "Respond with a single JSON object and nothing else. It must "
                "validate against this JSON schema:\n"
                f"{schema.model_json_schema()}"
            )
            messages = [{"role": "user", "content": instruction}]
            for attempt in range(2):
                raw = self._chat(messages, json_mode=True, max_tokens=3000)
                try:
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
            # Everything here runs with async_mode=False and is kept serial for
            # Groq's rate limit, so this only satisfies the abstract base class.
            return self.generate(prompt, schema=schema)

    return GroqJudge(resolved)
