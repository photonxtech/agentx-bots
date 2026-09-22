"""Thin wrapper around the Groq API for RAG answer generation."""

from groq import Groq


class Answerer:
    def __init__(self, api_key: str, model: str):
        self.client = Groq(api_key=api_key)
        self.model = model

    def summarize(self, title: str, text: str) -> str:
        """Generates a short document-level summary at ingestion time.

        Stored as its own chunk so broad questions like "what does this
        contain" retrieve something that actually answers that, instead
        of one random narrow chunk winning on embedding similarity.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Write a concise 2-4 sentence summary of the following "
                        "content. Be specific about what it actually covers "
                        "and its main points, not generic."
                    ),
                },
                {"role": "user", "content": f"Title: {title}\n\nContent:\n{text[:6000]}"},
            ],
            max_tokens=250,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _build_messages(self, question: str, hits: list[dict]):
        """Shared prompt construction for both answer() and answer_stream().
        Returns None if there are no hits — caller decides what to say then.
        """
        if not hits:
            return None

        context_blocks = []
        for h in hits:
            title = h["metadata"].get("title", h["metadata"].get("source", "source"))
            context_blocks.append(f"[Source: {title}]\n{h['text']}")
        context = "\n\n---\n\n".join(context_blocks)

        distinct_sources = {
            h["metadata"].get("title", h["metadata"].get("source", "source")) for h in hits
        }

        system_prompt = (
            "You are a helpful assistant answering questions using ONLY the "
            "context provided below, which was previously shared by this user. "
            "When the question refers to 'this link', 'this article', 'this "
            "document', 'this file', 'this image', 'it', or similar phrasing, "
            "treat that as referring to the source(s) in the context below — "
            "answer using what that source actually discusses. Don't get stuck "
            "on literal wording; 'what does this link contain' means 'what is "
            "this source about', not a technical question about a hyperlink. "
            "If the context genuinely doesn't contain the answer, say so plainly "
            "instead of guessing. Cite which source(s) you used by title. Be concise."
        )

        if len(distinct_sources) > 1:
            # Multiple genuinely different sources matched — don't let the
            # model quietly blend them into one anonymous voice. Only kicks
            # in when there's real cross-source content to call out.
            system_prompt += (
                f" The context below comes from {len(distinct_sources)} different "
                "sources the user shared. If more than one is actually relevant "
                "to the question, say so explicitly: name each relevant source "
                "and what it says, and state clearly whether they agree, add to "
                "each other, or disagree. Don't force this framing if only one "
                "source actually answers the question — in that case just answer "
                "normally from it."
            )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]

    def answer(self, question: str, hits: list[dict]) -> str:
        messages = self._build_messages(question, hits)
        if messages is None:
            return (
                "I don't have anything saved yet that relates to that. "
                "Share a link, document, or image with me first and ask again."
            )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=800,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def answer_stream(self, question: str, hits: list[dict]):
        """Same as answer(), but yields text as it's generated instead of
        waiting for the full response — lets the caller progressively
        update a Slack message rather than the person staring at nothing
        for however long the full completion takes.
        """
        messages = self._build_messages(question, hits)
        if messages is None:
            yield (
                "I don't have anything saved yet that relates to that. "
                "Share a link, document, or image with me first and ask again."
            )
            return

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=800,
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta