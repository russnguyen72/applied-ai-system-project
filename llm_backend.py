"""Local-LLM backend wrapping the Ollama Python client."""
from __future__ import annotations

from typing import Callable


class LLMBackendError(Exception):
    """Raised when the LLM backend cannot complete a request."""


class OllamaBackend:
    def __init__(
        self,
        model: str = "qwen3.5:4b",
        host: str = "http://localhost:11434",
        keep_alive: str = "30m",
    ) -> None:
        self.model = model
        self.host = host
        self.keep_alive = keep_alive
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
            except ImportError as e:
                raise LLMBackendError(
                    "The 'ollama' Python package is not installed. "
                    "Run: pip install -r requirements.txt"
                ) from e
            self._client = ollama.Client(host=self.host)
        return self._client

    def _raise_friendly(self, e: Exception) -> LLMBackendError:
        msg = str(e).lower()
        if isinstance(e, ConnectionError) or "connection" in msg or "refused" in msg or "could not connect" in msg:
            return LLMBackendError(
                "Ollama daemon not reachable. Start it with `ollama serve` "
                f"and ensure the model is pulled: `ollama pull {self.model}`."
            )
        if "not found" in msg and self.model.split(":")[0] in msg:
            return LLMBackendError(
                f"Model '{self.model}' is not available locally. "
                f"Run: ollama pull {self.model}"
            )
        return LLMBackendError(f"Ollama request failed: {e}")

    def chat(self, messages: list[dict], tools: list[dict], think: bool = False) -> dict:
        """Sends a non-streaming chat turn to Ollama and returns the raw response dict."""
        client = self._get_client()
        try:
            response = client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                think=think,
                keep_alive=self.keep_alive,
            )
        except TypeError:
            # Older ollama client versions don't accept `think` — retry without it.
            try:
                response = client.chat(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    keep_alive=self.keep_alive,
                )
            except Exception as e:
                raise self._raise_friendly(e) from e
        except Exception as e:
            raise self._raise_friendly(e) from e

        if hasattr(response, "model_dump"):
            return response.model_dump()
        if isinstance(response, dict):
            return response
        return dict(response)

    def chat_stream(
        self,
        messages: list[dict],
        tools: list[dict],
        think: bool = False,
        on_chunk: Callable[[dict], None] | None = None,
    ) -> dict:
        """Streams a chat turn, invoking `on_chunk` for each thinking/content delta.

        Returns an accumulated response with the same shape as `chat()`. Falls back
        to non-streaming if the client doesn't support streaming with tools.
        """
        client = self._get_client()

        stream_iter = None
        try:
            stream_iter = client.chat(
                model=self.model,
                messages=messages,
                tools=tools,
                think=think,
                stream=True,
                keep_alive=self.keep_alive,
            )
        except TypeError:
            # Older client: retry without think.
            try:
                stream_iter = client.chat(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                    keep_alive=self.keep_alive,
                )
            except Exception:
                return self.chat(messages, tools, think=think)
        except Exception as e:
            raise self._raise_friendly(e) from e

        accumulated_content = ""
        accumulated_thinking = ""
        final_tool_calls: list = []
        final_role = "assistant"

        try:
            for chunk in stream_iter:
                chunk_dict = self._normalize_chunk(chunk)
                msg = chunk_dict.get("message") or {}
                if not isinstance(msg, dict):
                    msg = self._normalize_chunk(msg)

                thinking_delta = msg.get("thinking") or ""
                content_delta = msg.get("content") or ""
                if thinking_delta:
                    accumulated_thinking += thinking_delta
                    if on_chunk is not None:
                        try:
                            on_chunk({"type": "thinking", "text": thinking_delta})
                        except Exception:
                            pass
                if content_delta:
                    accumulated_content += content_delta
                    if on_chunk is not None:
                        try:
                            on_chunk({"type": "content", "text": content_delta})
                        except Exception:
                            pass

                tc = msg.get("tool_calls") or []
                if tc:
                    # Ollama streams tool calls non-cumulatively — each chunk carries
                    # only the call(s) emitted in that chunk, not the running total.
                    # Extend, don't replace, or all but the last chunk's calls are lost.
                    final_tool_calls.extend(tc)
                role = msg.get("role")
                if role:
                    final_role = role
        except Exception as e:
            raise self._raise_friendly(e) from e

        return {
            "message": {
                "role": final_role,
                "content": accumulated_content,
                "thinking": accumulated_thinking,
                "tool_calls": final_tool_calls,
            }
        }

    @staticmethod
    def _normalize_chunk(chunk) -> dict:
        if hasattr(chunk, "model_dump"):
            return chunk.model_dump()
        if isinstance(chunk, dict):
            return chunk
        try:
            return dict(chunk)
        except Exception:
            return {}

    def health_check(self) -> tuple[bool, str]:
        """Returns (ok, message) describing whether the backend is usable."""
        try:
            client = self._get_client()
        except LLMBackendError as e:
            return False, str(e)

        try:
            models = client.list()
        except Exception as e:
            return False, (
                "Ollama daemon is not responding at "
                f"{self.host}. Start it with `ollama serve`. ({e})"
            )

        available = []
        models_list = models.get("models", []) if isinstance(models, dict) else getattr(models, "models", [])
        for m in models_list:
            if isinstance(m, dict):
                available.append(m.get("name") or m.get("model", ""))
            else:
                available.append(getattr(m, "name", None) or getattr(m, "model", ""))

        if not any(name.startswith(self.model.split(":")[0]) for name in available if name):
            return False, (
                f"Model '{self.model}' not found locally. "
                f"Run: ollama pull {self.model}"
            )

        return True, f"Ollama ready (model: {self.model})"
