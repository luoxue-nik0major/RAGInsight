import asyncio
import httpx
import json
from typing import List, Dict, Any, AsyncGenerator
from app.core.config import get_settings

settings = get_settings()

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.
When using information from the context, cite the source using [ref:chunk_INDEX] format where INDEX is the chunk number.
For example: "The capital of France is Paris [ref:chunk_0]."
If the context doesn't contain enough information, say so honestly.
Be concise and accurate."""

# Retry policy for transient failures (rate limit / gateway errors / timeouts)
_MAX_RETRIES = 3
_RETRY_BACKOFF = 1.0  # seconds, doubles each attempt
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class DeepSeekClient:
    def __init__(self):
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url.rstrip("/")
        self.model = settings.deepseek_model
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
    ) -> str:
        """Non-streaming chat completion with exponential-backoff retry."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        last_exc: Exception = RuntimeError("unreachable")
        for attempt in range(_MAX_RETRIES):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(url, headers=self.headers, json=payload)
                    if resp.status_code in _RETRYABLE_STATUS:
                        # Treat retryable HTTP status as a transport-level failure
                        raise httpx.TransportError(f"retryable status {resp.status_code}")
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_RETRY_BACKOFF * (2 ** attempt))
        raise last_exc

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion. Retries only before the first token;
        a mid-stream failure surfaces immediately to avoid duplicated content."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        attempt = 0
        while True:
            got_token = False
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", url, headers=self.headers, json=payload) as response:
                        if response.status_code in _RETRYABLE_STATUS:
                            raise httpx.TransportError(f"retryable status {response.status_code}")
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str == "[DONE]":
                                    break
                                try:
                                    data = json.loads(data_str)
                                    delta = data["choices"][0]["delta"]
                                    if "content" in delta:
                                        got_token = True
                                        yield delta["content"]
                                except (json.JSONDecodeError, KeyError, IndexError):
                                    continue
                return
            except (httpx.TimeoutException, httpx.TransportError):
                if got_token or attempt >= _MAX_RETRIES - 1:
                    raise
                attempt += 1
                await asyncio.sleep(_RETRY_BACKOFF * (2 ** (attempt - 1)))

    def _build_answer_messages(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        context_text = "\n\n".join(
            f"[chunk_{i}] {chunk['content']}" for i, chunk in enumerate(context_chunks)
        )
        user_prompt = f"""Context:
{context_text}

Question: {query}

Please answer the question based on the context above. Cite sources using [ref:chunk_INDEX]."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    async def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> str:
        """Generate answer with citations."""
        return await self.chat(self._build_answer_messages(query, context_chunks))

    async def generate_answer_stream(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> AsyncGenerator[str, None]:
        """Streaming variant of generate_answer; yields answer tokens."""
        async for token in self.chat_stream(self._build_answer_messages(query, context_chunks)):
            yield token


deepseek_client = DeepSeekClient()
