"""LLM answer generation (STEP 7).

Calls OpenAI Chat Completions with the grounded prompt and returns the answer
text. Retrieval provides the citations, so this module's only job is faithful,
low-temperature synthesis over the supplied context.
"""

from __future__ import annotations

import time

from openai import OpenAI, OpenAIError

from app.core.exceptions import LLMError
from app.core.logger import get_logger
from app.rag.prompt import (
    CHAT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_chat_prompt,
    build_user_prompt,
)
from app.schemas.documents import RetrievedChunk

logger = get_logger(__name__)


class LLMClient:
    """Generate grounded answers from retrieved context."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 800,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(self, question: str, chunks: list[RetrievedChunk]) -> str:
        """Return the model's answer for ``question`` given ``chunks``.

        Raises:
            LLMError: If the chat-completions call fails.
        """
        user_prompt = build_user_prompt(question, chunks)
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except OpenAIError as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        elapsed = time.perf_counter() - start
        answer = (response.choices[0].message.content or "").strip()
        logger.info("LLM responded in %.2fs (%d chars)", elapsed, len(answer))
        return answer

    def chat(self, message: str, intent_hint: str) -> str:
        """Generate a natural small-talk reply as Buddy (no retrieval).

        Used for greetings, "how are you", identity, thanks, etc. Runs at a
        higher temperature than answers so replies feel varied and human, and
        is guarded by the persona prompt against stating PhotonX facts.

        Raises:
            LLMError: If the chat-completions call fails.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=0.7,
                max_tokens=160,
                messages=[
                    {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_chat_prompt(message, intent_hint),
                    },
                ],
            )
        except OpenAIError as exc:
            raise LLMError(f"Chat request failed: {exc}") from exc

        reply = (response.choices[0].message.content or "").strip()
        logger.info("Chat reply generated (%d chars)", len(reply))
        return reply
