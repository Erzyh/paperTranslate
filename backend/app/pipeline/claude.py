"""Claude translator (Anthropic Messages API, official ``anthropic`` SDK).

Reuses the whole OllamaTranslator guard chain (placeholder / echo / script /
identifier guards, sentence fallback, glossary, continuation, document
brief) and only replaces the transport layer, like OpenAITranslator.

Claude Opus 5.5 specifics handled here:
- Thinking is always on and cannot be disabled; ``output_config.effort`` is
  the control (its API default is "medium" — set explicitly). No sampling
  parameters (``temperature`` etc. are rejected).
- Safety classifiers may decline a request (HTTP 200, ``stop_reason:
  "refusal"``). Server-side fallbacks (``fallbacks="default"``) re-run a
  declined request on the model Anthropic recommends for that category; a
  refusal that survives the fallback raises TranslationRefusedError so the
  paragraph keeps its source text instead of failing the whole paper.
- The system prompt (rules, examples, paper brief, glossary) is identical
  for every paragraph of a paper, so it is prompt-cached.

Key handling contract (same as OpenAITranslator): the API key lives only
inside the SDK client for the lifetime of this object; it is never stored as
an attribute, logged, or included in error messages.
"""
from __future__ import annotations

from typing import Iterator

import anthropic

from .translate import (
    _EXPLAIN_SYSTEM_PROMPT,
    OllamaTranslator,
    TranslationRefusedError,
    TranslatorUnavailableError,
    strip_think_blocks,
)

DEFAULT_CLAUDE_MODEL = "claude-opus-5-5"
DEFAULT_CLAUDE_EFFORT = "medium"
# Thinking counts toward max_tokens even though its text is not returned, so
# leave room beyond the (short) paragraph translation itself.
_MAX_TOKENS = 16000
# Server-side refusal fallback, routed by refusal category.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeTranslator(OllamaTranslator):
    """Translator backed by Claude via the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_CLAUDE_MODEL,
        effort: str = DEFAULT_CLAUDE_EFFORT,
        timeout: float = 180.0,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout)
        # The parent's Ollama HTTP client is not used by this translator.
        self._client.close()
        self.effort = effort
        # Tests inject a fake ``client``; production builds the SDK client
        # (which retries connection errors, 429 and 5xx on its own).
        self._anthropic = client or anthropic.Anthropic(
            api_key=api_key, timeout=timeout, max_retries=2)

    def _create(self, system: str, conversation: list[dict]):
        try:
            return self._anthropic.beta.messages.create(
                model=self.model,
                max_tokens=_MAX_TOKENS,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                messages=conversation,
                output_config={"effort": self.effort},
                betas=[_FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.AuthenticationError as exc:
            raise TranslatorUnavailableError(
                "Claude API 키가 유효하지 않습니다. Anthropic API 키를 확인해 주세요."
            ) from exc
        except anthropic.PermissionDeniedError as exc:
            raise TranslatorUnavailableError(
                "이 Anthropic API 키로는 해당 모델을 사용할 수 없습니다."
            ) from exc
        except anthropic.NotFoundError as exc:
            raise TranslatorUnavailableError(
                f"Claude 모델 '{self.model}'을(를) 찾을 수 없습니다."
            ) from exc
        except anthropic.RateLimitError as exc:
            raise TranslatorUnavailableError(
                "Claude API 요청 한도 초과입니다. 잠시 후 다시 시도해 주세요."
            ) from exc
        except anthropic.APIStatusError as exc:
            # The API's own message is self-diagnosing and never contains the key.
            raise TranslatorUnavailableError(
                f"Claude 서버가 오류를 반환했습니다 (HTTP {exc.status_code})"
                f" — {str(exc.message)[:200]}"
            ) from exc
        except anthropic.APIConnectionError as exc:
            raise TranslatorUnavailableError(
                "Claude 서버에 연결할 수 없습니다. 네트워크 상태를 확인해 주세요."
            ) from exc

    def _chat(self, messages: list[dict]) -> str:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        conversation = [{"role": m["role"], "content": m["content"]}
                        for m in messages if m["role"] != "system"]
        response = self._create(system, conversation)
        if response.stop_reason == "refusal":
            # The fallback chain declined too (or the category is never
            # retried, e.g. reasoning_extraction).
            raise TranslationRefusedError(
                "Claude가 이 문단의 번역을 거절했습니다.")
        # Read by block type: thinking blocks (empty under the default
        # display) and fallback markers precede the text.
        text = "".join(block.text for block in response.content
                       if block.type == "text")
        return strip_think_blocks(text).strip()

    def close(self) -> None:
        self._anthropic.close()

    def explain_stream(self, prompt: str) -> Iterator[str]:
        """Single-chunk explanation (no server-side streaming needed here)."""
        yield self._chat([
            {"role": "system", "content": _EXPLAIN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
