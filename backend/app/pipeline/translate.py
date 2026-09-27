# Translators and mask/unmask helpers for non-translatable tokens.
from __future__ import annotations

import json
import re
import time
from typing import Iterator, Protocol

import httpx

# Automatic retries for transient Ollama transport failures (connection
# refused during a reload, timeout while the GPU is busy) before a document
# run is aborted with TranslatorUnavailableError.
_TRANSPORT_RETRIES = 2
_TRANSPORT_RETRY_DELAY = 5.0

# Order matters: equation placeholders first, then URLs, e-mail addresses,
# DOIs and arXiv ids (never translated, and models like to "fix" them), then
# numeric citations.
_MASKABLE_RE = re.compile(
    r"(?:⟦EQ\d+⟧)"
    r"|(?:(?:https?://|www\.)[^\s⟦⟧]+)"
    r"|(?:\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+)"
    r"|(?:\b10\.\d{4,9}/[^\s⟦⟧]+)"
    r"|(?:\barXiv:\d{4}\.\d{4,5}(?:v\d+)?)"
    r"|(?:\[\d+(?:\s*[,–-]\s*\d+)*\])"
)
MASK_TOKEN_RE = re.compile(r"⟦M\d+⟧")
# Any placeholder-looking token, including extraction-side ⟦EQn⟧ ones.
_ANY_TOKEN_RE = re.compile(r"⟦[^⟦⟧\s]{1,12}⟧")


class Translator(Protocol):
    def translate(
        self, text: str, context: str | None = None,
        continuation: bool = False,
    ) -> str: ...


def mask(text: str) -> tuple[str, dict[str, str]]:
    """Replace equations (⟦EQn⟧), citations ([12]) and URLs with ⟦Mn⟧ tokens."""
    mapping: dict[str, str] = {}

    def _replace(match: re.Match) -> str:
        token = f"⟦M{len(mapping)}⟧"
        mapping[token] = match.group(0)
        return token

    return _MASKABLE_RE.sub(_replace, text), mapping


def unmask(text: str, mapping: dict[str, str]) -> str:
    """Restore the originals for every ⟦Mn⟧ token produced by mask()."""
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text


_SENTENCE_POOL = [
    "이 문장은 파이프라인 검증을 위한 더미 번역입니다.",
    "본 연구는 레이아웃을 보존하는 문서 번역 기법을 다룹니다.",
    "제안한 방법은 단 구조와 그림 배치를 그대로 유지합니다.",
    "각 문단은 원문과 동일한 위치에 다시 조판됩니다.",
    "수식과 인용 표기는 자리표시자로 보호된 뒤 복원됩니다.",
    "실험 결과는 검증 기준을 만족하는지 확인하는 데 쓰입니다.",
    "하이픈으로 나뉜 단어는 병합 단계에서 복원됩니다.",
    "출력 문서에서는 한국어 텍스트가 추출 가능해야 합니다.",
]


class StubTranslator:
    """Offline dummy translator.

    Returns Korean filler sentences amounting to roughly 85% of the source
    length while preserving every ⟦Mn⟧ mask token exactly once. The
    ``continuation`` flag is accepted for interface parity and ignored.
    """

    def translate(self, text: str, context: str | None = None,
                  continuation: bool = False) -> str:
        tokens = MASK_TOKEN_RE.findall(text)
        target_len = max(1, int(len(text) * 0.85))
        parts: list[str] = []
        total = 0
        index = 0
        while total < target_len:
            sentence = _SENTENCE_POOL[index % len(_SENTENCE_POOL)]
            parts.append(sentence)
            total += len(sentence) + 1
            index += 1
        for offset, token in enumerate(tokens):
            parts.insert(min(offset + 1, len(parts)), token)
        return " ".join(parts)


_UNAVAILABLE_MESSAGE = (
    "Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해 주세요."
)


class TranslatorUnavailableError(RuntimeError):
    """Raised when the translation backend cannot be reached (down/timeout)."""

    def __init__(self, message: str = _UNAVAILABLE_MESSAGE) -> None:
        super().__init__(message)


class TranslationRefusedError(RuntimeError):
    """The model's safety system declined to translate this text.

    Raised by API translators (e.g. a Claude ``refusal`` stop reason that
    survived the server-side fallback); translate() keeps the source text.
    """


# Qwen3-style reasoning blocks; stripped defensively even with "think": false.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks (multiline) from model output."""
    return _THINK_BLOCK_RE.sub("", text)


# Output-script guard: scripts a Korean translation of an English paper must
# never contain. Small local models (qwen3:8b) sporadically mix Han ideographs
# (e.g. "활성화된但我们"), kana or Cyrillic into otherwise-Korean output, and
# prompt rules alone do not prevent it — so it is enforced with the same
# corrective-retry machinery as the placeholder guard.
_FOREIGN_SCRIPT_RE = re.compile(
    "[一-鿿"   # CJK Unified Ideographs (Han)
    "぀-ヿ"    # Hiragana + Katakana
    "Ѐ-ӿ]"   # Cyrillic
)


def find_foreign_script_chars(source: str, translated: str) -> list[str]:
    """Return foreign-script characters (Han/kana/Cyrillic) the model injected.

    Characters that already occur in ``source`` are legitimate (e.g. a CJK
    title cited inside a reference) and are not reported. An empty list means
    the translation is script-safe.
    """
    found = set(_FOREIGN_SCRIPT_RE.findall(translated))
    return sorted(found - set(source))


def find_placeholder_errors(source: str, translated: str) -> list[str]:
    """Return the mask tokens of ``source`` not present exactly once in ``translated``.

    Shared placeholder guard for real translators (the stub preserves tokens
    by construction). An empty list means the translation is token-safe.
    Tokens hallucinated by the model (present in ``translated`` but absent
    from ``source``) are also errors — they would print as garbage in the PDF.
    """
    source_tokens = set(MASK_TOKEN_RE.findall(source))
    errors = [t for t in source_tokens if translated.count(t) != 1]
    # Any other ⟦...⟧ token (e.g. an ⟦EQn⟧ copied from the context) would
    # print as garbage or be mistaken for an equation slot downstream.
    known = set(_ANY_TOKEN_RE.findall(source))
    errors += [t for t in set(_ANY_TOKEN_RE.findall(translated)) - known]
    return sorted(errors)


# --- verbatim identifiers ---------------------------------------------------
# Names of models, datasets and tools (GPT-4, ImageNet, PyTorch) and acronyms
# (LLM, BLEU) must survive translation letter for letter; small models like to
# transliterate them ("버트") or drop them. Detected heuristically: a token is
# an identifier when it mixes letters and digits, or when one of its parts has
# two or more capitals (acronym or CamelCase). Plain Title-Case words
# ("Multi-Task", "Persona") are ordinary vocabulary and may be translated.
_IDENT_TOKEN_RE = re.compile(
    r"(?<![\w⟦])[A-Za-z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*(?![\w⟧])")
_ROMAN_NUMERAL_RE = re.compile(r"[IVXLCDM]+")
_DOTTED_ABBREV_RE = re.compile(r"(?:[A-Za-z]\.)*[A-Za-z]")  # "U.S", "e.g"
# A source whose letters are mostly capitals is an ALL-CAPS heading; its
# words are ordinary vocabulary, not identifiers.
_CAPS_HEADING_RATIO = 0.6
_MAX_REPORTED_IDENTIFIERS = 8


def _identifier_form(token: str) -> str | None:
    """The form of ``token`` that must appear verbatim, or None."""
    if _ROMAN_NUMERAL_RE.fullmatch(token) or _DOTTED_ABBREV_RE.fullmatch(token):
        return None
    has_letter = any(c.isalpha() for c in token)
    has_digit = any(c.isdigit() for c in token)
    parts = re.split(r"[-.]", token)
    capped = any(sum(c.isupper() for c in part) >= 2 for part in parts)
    if not (capped or (has_digit and has_letter)):
        return None
    # Plural acronyms: Korean has no plural, "LLMs" becomes "LLM".
    if token.endswith("s") and len(token) > 2 and token[-2].isupper():
        return token[:-1]
    return token


def find_missing_identifiers(source: str, translated: str) -> list[str]:
    """Identifiers of ``source`` (see above) that ``translated`` lost."""
    plain = _ANY_TOKEN_RE.sub(" ", source)
    letters = [c for c in plain if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > _CAPS_HEADING_RATIO:
        return []
    missing: list[str] = []
    for token in _IDENT_TOKEN_RE.findall(plain):
        form = _identifier_form(token)
        if form and form not in translated and form not in missing:
            missing.append(form)
    return missing[:_MAX_REPORTED_IDENTIFIERS]


def sanitize_context(text: str) -> str:
    """Context shown to the model must not carry placeholder tokens it could copy."""
    return _ANY_TOKEN_RE.sub("(수식)", text)


# Exact prompt-scaffold phrases, all under our control. A model response
# containing any of them has echoed the instructions instead of translating.
# Each marker is a full distinctive phrase so ordinary translations that
# merely contain words like "번역" can never match.
_ECHO_MARKERS = (
    "다음 텍스트를 한국어로 번역하라",
    "이전 문단(문맥 참고용",
    "번역문만 출력",
    "플레이스홀더는 번역하지 말고",
    # Glossary scaffold header (see _GLOSSARY_PROMPT_HEADER): a response
    # repeating it has echoed the injected glossary block.
    "용어집(반드시 이 표기를 따르라",
    # Continuation scaffold (see _CONTINUATION_INSTRUCTION).
    "새 문장처럼 시작하지 말고",
    # Few-shot, document-brief and previous-translation scaffolds.
    "[번역 예시",
    "[논문 정보",
    "이전 문단의 번역(",
    # The few-shot answers themselves: reproducing one is not a translation.
    "우리는 ImageNet으로 GPT-4를 파인튜닝하고",
    "Kim 등이 보였듯이 제안한 손실",
)

# Minimum len(stripped)/len(source) ratio to accept an echo-stripped answer.
_ECHO_STRIP_MIN_RATIO = 0.3


def find_echo_markers(text: str) -> list[str]:
    """Return the prompt-scaffold phrases echoed in a model response."""
    return [marker for marker in _ECHO_MARKERS if marker in text]


def strip_echo_lines(text: str) -> str:
    """Drop every line containing an echo marker, keeping the other lines."""
    kept = [line for line in text.splitlines() if not find_echo_markers(line)]
    return "\n".join(kept).strip()


# Formatting safety is enforced in code (placeholder/echo/script/identifier
# guards with corrective retries), so the prompt stays short and spends its
# budget on what to keep verbatim and on how a good translation reads.
_OLLAMA_SYSTEM_PROMPT = (
    "당신은 영어 학술 논문을 한국어로 옮기는 전문 번역가다. 다음 원칙을 지켜라.\n"
    "1. 번역문만 출력한다. 출력 첫 글자부터 번역문이며, 설명·인사·따옴표나 "
    "어떤 라벨·머리말·요약도 붙이지 않는다.\n"
    "2. ⟦M0⟧처럼 ⟦...⟧ 형태의 플레이스홀더는 번역하지 말고 글자 그대로, "
    "원문에 있는 것만 정확히 한 번씩 유지한다.\n"
    "3. 다음은 번역·음차하지 않고 원문 표기 그대로 둔다: 인명(예: 'Alice Kim, Bob Lee'), "
    "모델·데이터셋·도구 이름(예: GPT-4, ImageNet, PyTorch), 약어(예: LLM, BLEU), "
    "변수와 기호, 수치와 단위.\n"
    "4. 단어 대 단어 직역을 피하고, 한국어 학술 논문의 자연스러운 문어체(-다 체)로 옮긴다. "
    "이전 문단의 번역과 어투·용어를 일관되게 맞춘다.\n"
    "5. 외래 학술용어는 무리하게 한자어로 옮기지 말고 관례적 음차나 원어를 쓴다. "
    "예: Persona→페르소나, pipeline→파이프라인, baseline→베이스라인.\n"
    "6. Figure는 '그림', Table은 '표', Equation은 '식', Section은 '절'로 옮긴다.\n"
    "7. 한글과 로마자 이외의 문자(한자·가나·키릴 등)는 절대 섞지 않는다."
)

# Two worked examples: small models follow a demonstration far better than
# more rules. They show placeholders, names and acronyms kept verbatim and
# the target register. Their answers are echo markers (see _ECHO_MARKERS).
_FEW_SHOT_BLOCK = (
    "[번역 예시 — 형식과 어투만 참고할 것]\n"
    "원문: We fine-tune GPT-4 on ImageNet and report BLEU scores in Table 2 ⟦M0⟧.\n"
    "번역: 우리는 ImageNet으로 GPT-4를 파인튜닝하고, 표 2에 BLEU 점수를 보고한다 ⟦M0⟧.\n"
    "원문: As shown by Kim et al., the proposed loss ⟦M1⟧ converges faster than "
    "the baseline, which suggests that LLMs benefit from curriculum learning.\n"
    "번역: Kim 등이 보였듯이 제안한 손실 ⟦M1⟧은 베이스라인보다 빠르게 수렴하며, "
    "이는 LLM이 커리큘럼 학습의 이점을 얻는다는 것을 시사한다."
)

# Document brief header (title + abstract excerpt, set by the runner).
_BRIEF_PROMPT_HEADER = "[논문 정보 — 분야와 맥락 파악용, 번역하거나 출력하지 말 것]\n"
_BRIEF_MAX_CHARS = 700

# Instruction injected into the user message when the runner detected that
# the segment text starts mid-sentence (a tail continuing the previous
# column/page, see runner.is_continuation_start): without it the model
# opens a fresh sentence and the broken start reads like a new paragraph.
_CONTINUATION_INSTRUCTION = (
    "이 텍스트는 직전 문단에서 이어지는 문장의 뒷부분이다. "
    "새 문장처럼 시작하지 말고 자연스럽게 이어지는 번역만 출력하라."
)

# Glossary block header appended to the system prompt when a document
# glossary is available (built once per document by build_glossary()).
_GLOSSARY_PROMPT_HEADER = "용어집(반드시 이 표기를 따르라): "

_GLOSSARY_SYSTEM_PROMPT = (
    "당신은 영어 학술 논문의 전문용어를 한국어 관례 표기로 정리하는 전문가다. "
    "외래 학술용어는 무리한 한자어 직역 대신 관례적 음차나 원어 표기를 쓴다. "
    "예: Persona = 페르소나, pipeline = 파이프라인, baseline = 베이스라인."
)
_GLOSSARY_USER_PROMPT = (
    "다음 논문 발췌에서 핵심 전문용어 15~30개를 뽑아 "
    "'영어 = 한국어 관례 표기' 형식으로 한 줄에 하나씩만 출력하라. "
    "설명이나 번호는 붙이지 마라.\n\n"
)
# Hard cap on accepted glossary entries regardless of what the model returns.
_GLOSSARY_MAX_TERMS = 40

# Curated seed glossary: canonical Korean renderings of common academic loan
# terms. Model extraction is stochastic and may miss these, so any seed term
# that literally occurs in the document sample is force-included (overriding
# a conflicting extracted rendering — the seed is the convention).
_GLOSSARY_SEED = {
    "persona": "페르소나",
    "pipeline": "파이프라인",
    "baseline": "베이스라인",
    "framework": "프레임워크",
    "benchmark": "벤치마크",
    "heuristic": "휴리스틱",
    "ensemble": "앙상블",
    "architecture": "아키텍처",
    "multi-task": "멀티태스크",
    "transformer": "트랜스포머",
    "attention": "어텐션",
    "embedding": "임베딩",
    "fine-tuning": "파인튜닝",
    "prompt": "프롬프트",
    "dataset": "데이터셋",
    "layout": "레이아웃",
    "parameter": "파라미터",
    "hyperparameter": "하이퍼파라미터",
    "cold-start": "콜드 스타트",
    "gold standard": "골드 스탠더드",
    "ablation": "애블레이션",
    "random seed": "랜덤 시드",
    "repository": "리포지토리",
    "retrieval": "검색",
    "entailment": "함의",
    "hallucination": "할루시네이션",
}


def apply_glossary_seed(glossary: dict[str, str], sample_text: str) -> dict[str, str]:
    """Force-include seed terms that occur in ``sample_text``.

    Extracted entries whose key equals a seed term (case-insensitive) are
    overridden by the canonical seed rendering; other entries are kept.
    """
    haystack = sample_text.lower()
    merged = {
        term: rendering
        for term, rendering in glossary.items()
        if term.lower() not in _GLOSSARY_SEED
    }
    for term, rendering in _GLOSSARY_SEED.items():
        if term in haystack:
            merged[term.capitalize() if term[0].isalpha() else term] = rendering
    return merged
# Tolerated list-marker prefixes on glossary lines ("- ", "* ", "1. ", "1) ").
_GLOSSARY_LINE_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def parse_glossary(answer: str, sample_text: str) -> dict[str, str]:
    """Parse "English = Korean" lines into a glossary dict.

    Lines that do not match the format are dropped; terms whose English side
    does not literally occur in ``sample_text`` (case-insensitive) are dropped
    too, so the model cannot inject vocabulary absent from the document.
    """
    glossary: dict[str, str] = {}
    haystack = sample_text.lower()
    for raw_line in answer.splitlines():
        line = _GLOSSARY_LINE_PREFIX_RE.sub("", raw_line).strip()
        if "=" not in line:
            continue
        term, _, rendering = line.partition("=")
        term, rendering = term.strip(), rendering.strip()
        if not term or not rendering:
            continue
        if term.lower() not in haystack:
            continue
        glossary[term] = rendering
        if len(glossary) >= _GLOSSARY_MAX_TERMS:
            break
    return glossary


# Paraphrase-echo guard: models sometimes prepend a short label line such as
# "초록:" or "요약:" before the actual translation. Only clearly model-added
# labels are stripped (see strip_leading_label).
_LABEL_MAX_CHARS = 20


def _source_opens_with_colon_label(source: str) -> bool:
    """True when the source's own first sentence carries a colon label."""
    head = source.lstrip()
    first_line = head.split("\n", 1)[0]
    first_sentence = _SENTENCE_SPLIT_RE.split(first_line, maxsplit=1)[0]
    return ":" in first_sentence or "：" in first_sentence


def strip_leading_label(translated: str, source: str) -> str:
    """Drop a model-added label line (e.g. "초록:") from a translation.

    Conservative by design — the first line is removed only when all hold:
    it ends with a colon, is at most 20 characters, contains no mask token
    and no echo marker (echoed scaffold has its own retry-based guard), more
    lines follow it, and the source itself does not open with a colon label
    (which would make the label a legitimate part of the translation).
    """
    lines = translated.splitlines()
    if len(lines) < 2:
        return translated
    first = lines[0].strip()
    if not first.endswith((":", "：")) or len(first) > _LABEL_MAX_CHARS:
        return translated
    if MASK_TOKEN_RE.search(first) or find_echo_markers(first):
        return translated
    if _source_opens_with_colon_label(source):
        return translated
    rest = "\n".join(lines[1:]).strip()
    return rest or translated

# ---------------------------------------------------------------------------
# Explanation helpers (POST /explain) — separate from the translation logic.

# Hard cap on the excerpt passed to the explanation prompt.
_EXPLAIN_TEXT_MAX_CHARS = 4000

_EXPLAIN_SYSTEM_PROMPT = (
    "당신은 학술 논문을 쉽게 풀어 설명하는 한국어 조교다. "
    "요청받은 내용만 한국어로 설명하고, 인사말이나 머리말을 붙이지 않는다."
)


def build_explain_prompt(text: str, kind: str, title: str | None = None) -> str:
    """Build the Korean explanation prompt for POST /explain.

    ``text`` is capped at 4000 characters. ``kind`` selects the scaffold
    ("selection" | "formula"); ``title`` is the document's first heading
    source and is omitted from the prompt when unknown.
    """
    excerpt = text[:_EXPLAIN_TEXT_MAX_CHARS]
    if kind == "formula":
        return "다음 수식을 항별로 한국어로 해설하라.\n\n" + excerpt
    paper = f"논문 '{title}'" if title else "논문"
    return (
        f"다음은 {paper}의 발췌다. 학부생에게 설명하듯 한국어로 "
        "간결히 설명하라(5문장 이내).\n\n" + excerpt
    )


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def _partial_tag_suffix(text: str, tag: str) -> str:
    """Longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for length in range(min(len(text), len(tag) - 1), 0, -1):
        if tag.startswith(text[-length:]):
            return text[-length:]
    return ""


class _ThinkStreamFilter:
    """Drop <think>...</think> spans from a stream of content chunks.

    Stateful on purpose: a tag (or the whole block) may be split across
    chunk boundaries, so a possible partial tag at the end of a chunk is
    held back until the next chunk decides what it was.
    """

    def __init__(self) -> None:
        self._in_think = False
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        emitted: list[str] = []
        while True:
            if self._in_think:
                idx = self._buffer.find(_THINK_CLOSE)
                if idx == -1:
                    self._buffer = _partial_tag_suffix(self._buffer, _THINK_CLOSE)
                    break
                self._buffer = self._buffer[idx + len(_THINK_CLOSE):]
                self._in_think = False
            else:
                idx = self._buffer.find(_THINK_OPEN)
                if idx == -1:
                    tail = _partial_tag_suffix(self._buffer, _THINK_OPEN)
                    emitted.append(self._buffer[:len(self._buffer) - len(tail)])
                    self._buffer = tail
                    break
                emitted.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + len(_THINK_OPEN):]
                self._in_think = True
        return "".join(emitted)

    def flush(self) -> str:
        """Return the held-back tail (unless it belongs to an open think block)."""
        rest = "" if self._in_think else self._buffer
        self._buffer = ""
        return rest


# Max characters of the previous-paragraph context (source and translation)
# passed to the model.
_CONTEXT_MAX_CHARS = 400
# A response that repeats this many leading characters of the previous
# paragraph's translation has echoed the context instead of translating.
_PREVIOUS_ECHO_PROBE_CHARS = 30
# Extra corrective attempts after the first response fails a guard
# (placeholder loss, prompt echo, or foreign-script contamination).
_MAX_PLACEHOLDER_RETRIES = 2
# Corrective-retry budget of the per-sentence fallback pass. Two retries
# (instead of the former one) so a residual failure costs fewer sentences
# left untranslated in the output.
_MAX_SENTENCE_RETRIES = 2

# Corrective instruction for the foreign-script guard.
_FOREIGN_SCRIPT_CORRECTION = (
    "한자·가나·키릴 문자를 쓰지 말고 한국어와 로마자만 사용해 다시 출력하라."
)

# Sentence boundary: terminal punctuation followed by a capital/token/bracket.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z⟦(\[])")


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into rough sentences for the per-sentence retry pass."""
    return [s for s in (p.strip() for p in _SENTENCE_SPLIT_RE.split(text)) if s]


class OllamaTranslator:
    """Translator backed by a local Ollama server (POST /api/chat).

    Safety contract: tests must NEVER let this class reach a real server —
    always inject an ``httpx.MockTransport`` via ``transport``. The default
    (``transport=None``) performs real HTTP and is for production only.

    After each ``translate()`` call, ``last_fallback`` is True iff the model
    kept losing mask tokens, echoing the prompt scaffold, or mixing foreign
    scripts (Han/kana/Cyrillic) and the source text was returned unchanged;
    the runner uses this flag to record the segment status as "fallback".
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        base_url: str = "http://localhost:11434",
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.last_fallback = False
        # Document-level glossary injected into the system prompt; set by the
        # runner via build_glossary(). Empty dict means "no glossary".
        self.glossary: dict[str, str] = {}
        # Title + abstract excerpt of the paper, set by the runner; tells the
        # model the field so ambiguous terms get the right sense.
        self.document_brief: str = ""
        # Injected (mock) transports are for tests: skip transient retries so
        # error-path tests stay fast and call counts stay deterministic.
        self._transport_retries = 0 if transport is not None else _TRANSPORT_RETRIES
        self._client = httpx.Client(
            base_url=self.base_url, timeout=timeout, transport=transport
        )

    def _chat(self, messages: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Disable Qwen3-style thinking (Ollama 0.9+); metadata-only flag.
            "think": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
        }
        # Transient transport failures (Ollama busy while another workload
        # holds the GPU, brief reloads) abort a whole document run, so retry
        # before surfacing the error.
        response = None
        last_exc: Exception | None = None
        for attempt in range(self._transport_retries + 1):
            try:
                response = self._client.post("/api/chat", json=payload)
                break
            except httpx.TimeoutException as exc:
                last_exc = exc
            except httpx.TransportError as exc:
                last_exc = exc
            if attempt < self._transport_retries:
                time.sleep(_TRANSPORT_RETRY_DELAY)
        if response is None:
            if isinstance(last_exc, httpx.TimeoutException):
                raise TranslatorUnavailableError(
                    f"Ollama 응답 시간이 초과되었습니다({int(self.timeout)}초). "
                    "GPU가 다른 작업으로 바쁘거나 모델이 재로딩 중일 수 있습니다. "
                    "잠시 후 다시 시도해 주세요."
                ) from last_exc
            raise TranslatorUnavailableError() from last_exc
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise TranslatorUnavailableError(
                f"Ollama 서버가 오류를 반환했습니다 (HTTP {exc.response.status_code}). "
                "모델 설정과 Ollama 상태를 확인해 주세요."
            ) from exc
        content = response.json()["message"]["content"]
        return strip_think_blocks(content).strip()

    def close(self) -> None:
        """Release the underlying HTTP connection pool."""
        self._client.close()

    def explain_stream(self, prompt: str) -> Iterator[str]:
        """Stream a Korean explanation for ``prompt`` (POST /api/chat stream=true).

        Yields only each streamed chunk's ``message.content``; empty chunks
        and <think> reasoning blocks (even split across chunks) are dropped.
        Raises TranslatorUnavailableError when the server cannot be reached
        or answers with an HTTP error before the stream starts.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": True,
            # Disable Qwen3-style thinking (Ollama 0.9+); metadata-only flag.
            "think": False,
            "options": {"temperature": 0.2, "num_ctx": 8192},
        }
        think_filter = _ThinkStreamFilter()
        try:
            with self._client.stream("POST", "/api/chat", json=payload) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    raise TranslatorUnavailableError(
                        f"Ollama 서버가 오류를 반환했습니다 "
                        f"(HTTP {exc.response.status_code}). "
                        "모델 설정과 Ollama 상태를 확인해 주세요."
                    ) from exc
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except ValueError:
                        continue  # skip malformed NDJSON lines defensively
                    content = (data.get("message") or {}).get("content") or ""
                    if content:
                        filtered = think_filter.feed(content)
                        if filtered:
                            yield filtered
                    if data.get("done"):
                        break
                tail = think_filter.flush()
                if tail:
                    yield tail
        except httpx.TransportError as exc:
            raise TranslatorUnavailableError() from exc

    @staticmethod
    def _build_user_message(
        text: str, context: str | None, continuation: bool = False,
        previous_translation: str | None = None,
    ) -> str:
        parts = []
        if context:
            parts.append(
                "이전 문단(문맥 참고용, 번역하지 말 것):\n"
                + sanitize_context(context)[:_CONTEXT_MAX_CHARS]
            )
        if previous_translation:
            parts.append(
                "이전 문단의 번역(어투·용어를 이어서 맞출 것, 다시 출력하지 말 것):\n"
                + sanitize_context(previous_translation)[:_CONTEXT_MAX_CHARS]
            )
        if continuation:
            parts.append(_CONTINUATION_INSTRUCTION)
        parts.append("다음 텍스트를 한국어로 번역하라:\n" + text)
        return "\n\n".join(parts)

    def _system_prompt(self) -> str:
        """Rules + worked examples, plus the paper brief and glossary when set."""
        prompt = _OLLAMA_SYSTEM_PROMPT + "\n\n" + _FEW_SHOT_BLOCK
        if self.document_brief:
            prompt += ("\n\n" + _BRIEF_PROMPT_HEADER
                       + sanitize_context(self.document_brief)[:_BRIEF_MAX_CHARS])
        if self.glossary:
            pairs = ", ".join(f"{en}={ko}" for en, ko in self.glossary.items())
            prompt += "\n" + _GLOSSARY_PROMPT_HEADER + pairs
        return prompt

    def build_glossary(self, sample_text: str) -> dict[str, str]:
        """Extract a document glossary ("English = Korean") in one chat call.

        Best-effort: any failure (server error, malformed answer, empty
        sample) yields an empty dict so translation proceeds without a
        glossary instead of aborting the pipeline.
        """
        if not sample_text.strip():
            return {}
        messages = [
            {"role": "system", "content": _GLOSSARY_SYSTEM_PROMPT},
            {"role": "user", "content": _GLOSSARY_USER_PROMPT + sample_text},
        ]
        try:
            answer = self._chat(messages)
        except Exception:
            # Extraction failed — the curated seed still applies.
            return apply_glossary_seed({}, sample_text)
        return apply_glossary_seed(parse_glossary(answer, sample_text), sample_text)

    def _attempt(self, text: str, context: str | None, retries: int,
                 continuation: bool = False,
                 previous_translation: str | None = None) -> str | None:
        """Translate ``text`` with the placeholder/echo/script/identifier guards.

        Returns None when every guarded attempt fails. Lost identifiers are a
        soft guard: they trigger corrective retries but never reject the
        final answer (a slightly off name beats an English paragraph).
        """
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user",
             "content": self._build_user_message(
                 text, context, continuation, previous_translation)},
        ]
        probe = (previous_translation or "").strip()[:_PREVIOUS_ECHO_PROBE_CHARS]

        def echoed(answer: str) -> list[str]:
            found = find_echo_markers(answer)
            if len(probe) == _PREVIOUS_ECHO_PROBE_CHARS and probe in answer:
                found.append("이전 문단의 번역 반복")
            return found

        result = strip_leading_label(self._chat(messages), text)
        for _ in range(retries):
            errors = find_placeholder_errors(text, result)
            echoes = echoed(result)
            foreign = find_foreign_script_chars(text, result)
            missing = find_missing_identifiers(text, result)
            if not errors and not echoes and not foreign and not missing:
                return result
            # Corrective retry: keep the failed answer in the conversation and
            # state exactly what was wrong with it.
            corrections = []
            if errors:
                corrections.append(
                    "플레이스홀더 오류가 있다. 원문에 있는 플레이스홀더는 정확히 "
                    "한 번씩 그대로 포함하고, 원문에 없는 플레이스홀더를 만들어내지 마라. "
                    "문제가 된 토큰: " + ", ".join(errors)
                )
            if echoes:
                corrections.append(
                    "지시문이나 문맥 라벨을 출력하지 말고 번역문만 출력하라."
                )
            if foreign:
                corrections.append(
                    _FOREIGN_SCRIPT_CORRECTION
                    + " 문제가 된 문자: " + ", ".join(foreign)
                )
            if missing:
                corrections.append(
                    "다음 이름·약어는 번역하거나 음차하지 말고 원문 표기 그대로 써라: "
                    + ", ".join(missing)
                )
            corrections.append("설명 없이 번역문만 다시 출력하라.")
            messages = messages + [
                {"role": "assistant", "content": result},
                {"role": "user", "content": "\n".join(corrections)},
            ]
            result = strip_leading_label(self._chat(messages), text)
        if echoed(result):
            # Last resort before giving up: drop the echoed instruction lines
            # and accept the remainder only when enough translation survives.
            result = strip_echo_lines(result)
            if len(result) < len(text) * _ECHO_STRIP_MIN_RATIO:
                return None
        if find_placeholder_errors(text, result):
            return None
        return None if find_foreign_script_chars(text, result) else result

    def _plain_sentence_retry(self, text: str, context: str | None,
                              continuation: bool = False,
                              previous_translation: str | None = None) -> str | None:
        """Last-chance single retranslation of a mask-free sentence.

        Runs only after _attempt() exhausted its guarded retries in the
        per-sentence pass. One fresh chat call without the corrective loop;
        the only rejection guard is the foreign-script check. Echoed scaffold
        lines are stripped instead of rejected, and an answer carrying a mask
        token is refused — the sentence has none, so a hallucinated token
        would collide with the enclosing paragraph's mapping at unmask time.
        """
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user",
             "content": self._build_user_message(
                 text, context, continuation, previous_translation)},
        ]
        result = strip_echo_lines(strip_leading_label(self._chat(messages), text))
        if not result or MASK_TOKEN_RE.search(result):
            return None
        return None if find_foreign_script_chars(text, result) else result

    def translate(self, text: str, context: str | None = None,
                  continuation: bool = False,
                  previous_translation: str | None = None) -> str:
        """Translate ``text``; ``continuation=True`` marks a mid-sentence
        start (the tail of a sentence broken across a column/page boundary)
        and injects _CONTINUATION_INSTRUCTION into the prompt.
        ``previous_translation`` is how the preceding paragraph was rendered,
        so tone and terminology carry over between paragraphs.

        A paragraph the model's safety system declines to translate
        (TranslationRefusedError) keeps its source text like any other
        fallback, so one declined paragraph never fails the whole paper."""
        self.last_fallback = False
        try:
            return self._translate_guarded(text, context, continuation,
                                           previous_translation)
        except TranslationRefusedError:
            self.last_fallback = True
            return text

    def _translate_guarded(self, text: str, context: str | None,
                           continuation: bool,
                           previous_translation: str | None) -> str:
        result = self._attempt(text, context, _MAX_PLACEHOLDER_RETRIES,
                               continuation, previous_translation)
        if result is not None:
            return result
        # Paragraph-level guard failure. Token-dense paragraphs (many inline
        # equations) overwhelm small models, so retry sentence by sentence: a
        # residual failure then costs one English sentence, not the paragraph.
        # Only the first sentence can be a continuation tail.
        sentences = _split_sentences(text)
        if len(sentences) >= 2:
            pieces: list[str] = []
            any_ok = False
            for index, sentence in enumerate(sentences):
                is_tail = continuation and index == 0
                out = self._attempt(sentence, context, _MAX_SENTENCE_RETRIES,
                                    is_tail, previous_translation)
                if out is None and not MASK_TOKEN_RE.search(sentence):
                    # Mask-free sentences get one plain last-chance call
                    # before being left in English (see _plain_sentence_retry).
                    out = self._plain_sentence_retry(
                        sentence, context, is_tail, previous_translation)
                if out is None:
                    pieces.append(sentence)
                else:
                    pieces.append(out)
                    any_ok = True
            if any_ok:
                return " ".join(pieces)
        # Full guard failure: fall back to the source text (tokens intact by
        # definition) and let the runner record the segment status.
        self.last_fallback = True
        return text


_OPENAI_INVALID_KEY_MESSAGE = "API 키가 유효하지 않습니다"
_OPENAI_RATE_LIMIT_MESSAGE = "요청 한도 초과"
_OPENAI_UNREACHABLE_MESSAGE = (
    "{provider} 서버에 연결할 수 없습니다. 네트워크 상태를 확인해 주세요."
)


class OpenAITranslator(OllamaTranslator):
    """One-off translator backed by an OpenAI-style Chat Completions API.

    Serves OpenAI itself and OpenAI-compatible endpoints such as Gemini's
    (``chat_path`` and ``provider`` differ). Reuses the entire
    OllamaTranslator guard chain (placeholder/echo/script guards, sentence
    fallback, glossary, continuation) by inheritance and only replaces the
    transport layer (_chat / explain_stream).

    Key handling contract: the API key lives ONLY in the httpx client's
    Authorization header for the lifetime of this object. It is never stored
    as an attribute, never logged, and never included in exception messages.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.6-luna",
        base_url: str = "https://api.openai.com",
        timeout: float = 180.0,
        transport: httpx.BaseTransport | None = None,
        chat_path: str = "/v1/chat/completions",
        provider: str = "OpenAI",
    ) -> None:
        super().__init__(model=model, base_url=base_url, timeout=timeout,
                         transport=transport)
        self.chat_path = chat_path
        self.provider = provider
        # Replace the header-less client built by the parent constructor.
        self._client.close()
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def _chat(self, messages: list[dict]) -> str:
        # GPT-5-family reasoning models reject sampling overrides such as
        # temperature (HTTP 400: only the default of 1 is supported), so no
        # sampling parameters are sent — the model default applies.
        payload = {
            "model": self.model,
            "messages": messages,
        }
        response = None
        last_exc: Exception | None = None
        for attempt in range(self._transport_retries + 1):
            try:
                response = self._client.post(self.chat_path, json=payload)
                break
            except httpx.TransportError as exc:
                last_exc = exc
            if attempt < self._transport_retries:
                time.sleep(_TRANSPORT_RETRY_DELAY)
        if response is None:
            raise TranslatorUnavailableError(
                _OPENAI_UNREACHABLE_MESSAGE.format(provider=self.provider)
            ) from last_exc
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 401:
                raise TranslatorUnavailableError(
                    _OPENAI_INVALID_KEY_MESSAGE
                    + f". {self.provider} API 키를 확인해 주세요."
                ) from exc
            if status == 429:
                raise TranslatorUnavailableError(
                    _OPENAI_RATE_LIMIT_MESSAGE
                    + "입니다. 잠시 후 다시 시도해 주세요."
                ) from exc
            # Surface OpenAI's own error message (param mismatches, unknown
            # model, ...) so failures are self-diagnosing. Never include the
            # API key — it only exists in the request header, not the body.
            detail = ""
            try:
                body = exc.response.json()
                # Gemini's compatibility layer wraps the error in a list.
                if isinstance(body, list) and body:
                    body = body[0]
                message = body.get("error", {}).get("message", "")
                if message:
                    detail = f" — {message[:200]}"
            except Exception:
                pass
            raise TranslatorUnavailableError(
                f"{self.provider} 서버가 오류를 반환했습니다 (HTTP {status}){detail}"
            ) from exc
        content = response.json()["choices"][0]["message"]["content"]
        return strip_think_blocks(content).strip()

    def explain_stream(self, prompt: str) -> Iterator[str]:
        """Single-chunk explanation (no server-side streaming needed here)."""
        yield self._chat([
            {"role": "system", "content": _EXPLAIN_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])


def get_translator(name: str, ollama_model: str | None = None) -> Translator:
    """Build a translator by config name ("stub" | "ollama").

    ``ollama_model`` picks one of the installed local models for this job;
    None uses the configured default (PAPERTRANSLATE_OLLAMA_MODEL).
    """
    normalized = (name or "").strip().lower()
    if normalized == "stub":
        return StubTranslator()
    if normalized == "ollama":
        from app import config

        return OllamaTranslator(
            model=ollama_model or config.get_ollama_model(),
            base_url=config.get_ollama_url(),
            timeout=config.get_ollama_timeout(),
        )
    raise ValueError(f"알 수 없는 번역기 이름입니다: {name}")
