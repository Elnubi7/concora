from __future__ import annotations

import httpx


class GitHubModelsError(RuntimeError):
    pass


class GitHubModelsClient:
    def __init__(self, settings) -> None:
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.GITHUB_TOKEN)

    def _headers(self, json_body: bool = False) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.settings.GITHUB_TOKEN}",
            "X-GitHub-Api-Version": self.settings.GITHUB_API_VERSION,
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _chat_path(self) -> str:
        if self.settings.GITHUB_MODELS_ORG:
            return f"/orgs/{self.settings.GITHUB_MODELS_ORG}/inference/chat/completions"
        return "/inference/chat/completions"

    def _embedding_path(self) -> str:
        if self.settings.GITHUB_MODELS_ORG:
            return f"/orgs/{self.settings.GITHUB_MODELS_ORG}/inference/embeddings"
        return "/inference/embeddings"

    async def chat(self, messages: list[dict], temperature: float | None = None, max_tokens: int | None = None) -> str:
        if not self.enabled:
            raise GitHubModelsError("GITHUB_TOKEN is missing.")

        payload = {
            "model": self.settings.GITHUB_CHAT_MODEL,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.DEFAULT_TEMPERATURE,
            "max_tokens": max_tokens or self.settings.MAX_RESPONSE_TOKENS,
        }

        async with httpx.AsyncClient(
            base_url=self.settings.GITHUB_MODELS_BASE_URL,
            timeout=self.settings.REQUEST_TIMEOUT_SECONDS,
        ) as client:
            response = await client.post(self._chat_path(), json=payload, headers=self._headers(json_body=True))

        if response.status_code >= 400:
            raise GitHubModelsError(f"GitHub Models chat error {response.status_code}: {response.text}")

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise GitHubModelsError(f"Unexpected chat response format: {data}") from exc

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            raise GitHubModelsError("GITHUB_TOKEN is missing.")

        payload = {
            "model": self.settings.GITHUB_EMBEDDING_MODEL,
            "input": texts,
        }

        async with httpx.AsyncClient(
            base_url=self.settings.GITHUB_MODELS_BASE_URL,
            timeout=self.settings.REQUEST_TIMEOUT_SECONDS,
        ) as client:
            response = await client.post(self._embedding_path(), json=payload, headers=self._headers(json_body=True))

        if response.status_code >= 400:
            raise GitHubModelsError(f"GitHub Models embedding error {response.status_code}: {response.text}")

        data = response.json()
        try:
            rows = sorted(data["data"], key=lambda row: row["index"])
            return [row["embedding"] for row in rows]
        except (KeyError, TypeError) as exc:
            raise GitHubModelsError(f"Unexpected embedding response format: {data}") from exc