import json
from functools import lru_cache

from openai import AsyncOpenAI

from app.config import settings


@lru_cache
def get_llm_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def generate_structured(messages: list[dict], schema: dict, usage_holder: dict | None = None) -> dict:
    # Structured Outputs (strict JSON schema) trades real-time token streaming for a
    # guaranteed response shape — the API call itself is non-streaming, since parsing a
    # partial/incomplete JSON string safely mid-stream isn't worth the complexity here.
    # Callers that need a "typing" UX replay the returned text in chunks instead (see
    # chat_service._chunk_for_replay), which gets the same visual effect either way.
    client = get_llm_client()
    response = await client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.2,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "chat_response", "schema": schema, "strict": True},
        },
    )
    if usage_holder is not None and response.usage is not None:
        usage_holder["prompt_tokens"] = response.usage.prompt_tokens
        usage_holder["completion_tokens"] = response.usage.completion_tokens
        usage_holder["total_tokens"] = response.usage.total_tokens

    return json.loads(response.choices[0].message.content)
