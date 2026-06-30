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
        """Non-streaming chat completion."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=self.headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion (not used in Phase 1 but prepared)."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=self.headers, json=payload) as response:
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
                                yield delta["content"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

    async def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
    ) -> str:
        """Generate answer with citations."""
        context_text = "\n\n".join(
            f"[chunk_{i}] {chunk['content']}" for i, chunk in enumerate(context_chunks)
        )
        user_prompt = f"""Context:
{context_text}

Question: {query}

Please answer the question based on the context above. Cite sources using [ref:chunk_INDEX]."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        return await self.chat(messages)


deepseek_client = DeepSeekClient()
