from __future__ import annotations

from openai import OpenAI


class OpenAIGenerator:
    """Generator seam implementation backed by a real OpenAI chat completion call.

    Constructed with an explicit api_key/model_name rather than reading Settings
    itself, so it stays a plain Generator implementation - the seam RAGPipeline
    depends on - and doesn't need to know where its config came from.
    """

    def __init__(self, api_key: str, model_name: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content
