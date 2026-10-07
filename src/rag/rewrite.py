from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter

import httpx

from rag.config import GenerationConfig, InteractiveConfig
from rag.grounding import strip_markers
from rag.normalize import normalize_for_search
from rag.sessions import Turn

_EN_OPENER = re.compile(r"^(and|also|then|so|what about|how about|what else)\b")
_EN_REFERENCE = re.compile(r"\b(it|its|it's|itself|they|them|their|theirs|that|this|those|these|there|one|ones|same|he|she|him|her|his)\b")
_AR_OPENER = re.compile(r"^(و(كم|ما|ماذا|هل|اين|متي|كيف|بكم|ايش|لو)|طيب|طب|ماذا عن|وماذا عن|وش عن|وبالنسبه|بالنسبه ل|كذلك|ايضا|بعد)(\s|$|ال)")
_AR_REFERENCE_WORDS = {
    "هم", "هذا", "هذه", "هذي", "ذلك", "تلك", "ذا", "عنه", "عنها", "عنهم", "فيه", "فيها", "له", "لها", "لهم",
    "منه", "منها", "نفسه", "نفسها", "حقه", "حقها", "حقته", "حقهم", "اياه", "اياها",
}
_AR_ATTACHED = re.compile(
    r"(سعر|اسعار|ابعاد|مقاس|مقاسات|حجم|رقم|هاتف|جوال|موقع|عنوان|مكان|مدت|ضمان|خدمت|خدمات|تركيب|وقت|عدد|سقف|بدل|ساعات|دوام"
    r"|يقدم|تقدم|يقع|تقع|يكلف|تكلف|يحتاج|تحتاج|يستغرق|تستغرق|يغطي|تغطي|يشمل|تشمل)(ه|ها|هم)$"
)
_PRODUCT_CODE = re.compile(r"[A-Za-z]{2,}-[A-Za-z0-9]+-?[0-9]+")
_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_NOT_NAMES = {"I", "I'm", "SAR"}
SHORT_QUERY_TOKENS = 3
REFERENCE_MAX_TOKENS = 8
SELF_CONTAINED_MIN_TOKENS = 5


def names_entity(query: str) -> bool:
    if _PRODUCT_CODE.search(query):
        return True
    stripped = query.strip()
    return any(
        word[0].isupper() and word not in _NOT_NAMES and not stripped.startswith(word)
        for word in _LATIN_WORD.findall(stripped)
    )


@dataclass(frozen=True)
class GateDecision:
    rewrite: bool
    reason: str


def gate(query: str, history: list[Turn]) -> GateDecision:
    if not history:
        return GateDecision(False, "no_history")
    text = normalize_for_search(query).strip(" ؟?!.،,")
    tokens = re.findall(r"\w+", text)
    if _EN_OPENER.search(text) or _AR_OPENER.search(text):
        return GateDecision(True, "follow_up_opener")
    if len(tokens) >= SELF_CONTAINED_MIN_TOKENS and names_entity(query):
        return GateDecision(False, "names_its_entity")
    if len(tokens) <= REFERENCE_MAX_TOKENS:
        if _EN_REFERENCE.search(text) or any(token in _AR_REFERENCE_WORDS for token in tokens):
            return GateDecision(True, "reference_word")
        if any(_AR_ATTACHED.search(token) for token in tokens):
            return GateDecision(True, "attached_pronoun")
    if len(tokens) <= SHORT_QUERY_TOKENS:
        return GateDecision(True, "short_query")
    return GateDecision(False, "standalone")


_RULES_V1 = """You rewrite follow-up questions for a search engine. Given the previous exchange and a follow-up question, write one standalone question that can be understood without the conversation.
Rules:
1. Replace pronouns and vague references with the exact product, place, service or policy they refer to, copied from the previous exchange. Keep product names and codes exactly as written.
2. If the follow-up names a new item ("what about X"), ask the previous question about X.
3. Keep the language of the follow-up question: Arabic stays Arabic, English stays English, and a mixed Arabic/English question may stay mixed.
4. Do not answer the question and do not add details that were not asked.
5. If the follow-up question is already standalone, return it unchanged.
Output only the rewritten question."""

_RULES_V2 = """You rewrite follow-up questions for a search engine. Given the previous exchange and a follow-up question, write one standalone question that can be understood without the conversation.
Rules:
1. Replace pronouns and vague references with the exact product, place, service or policy they refer to, copied from the previous exchange. Keep product names and codes exactly as written.
2. If the follow-up names a new item ("what about X"), ask about X exactly what the previous question asked (the same attribute: price, dimensions, warranty, availability, location, phone, duration, and so on). Translate that attribute if the follow-up is in another language.
3. Keep the language of the follow-up question: Arabic stays Arabic, English stays English, and a mixed Arabic/English question may stay mixed.
4. Do not answer the question and do not add details that were not asked.
5. If the follow-up question is already standalone, return it unchanged.
Output only the rewritten question."""

_EXAMPLES_V1 = """Examples:
Previous question: How much is the Orbit floor lamp?
Follow-up question: And its height?
Standalone question: What is the height of the Orbit floor lamp?

Previous question: ما سعر طاولة القهوة Luna؟
Follow-up question: وماذا عن طاولة Sol؟
Standalone question: ما سعر طاولة Sol؟

Previous question: Where is the Tabuk branch?
Follow-up question: وكم رقم جواله؟
Standalone question: كم رقم جوال فرع تبوك؟"""

_EXAMPLES_BALANCED = """Examples:
Previous question: How much is the Orbit floor lamp?
Follow-up question: And its height?
Standalone question: What is the height of the Orbit floor lamp?

Previous question: What are the dimensions of the Luna coffee table?
Follow-up question: وماذا عن طاولة Nova؟
Standalone question: ما أبعاد طاولة Nova؟

Previous question: كم مدة ضمان كنبة Cedar؟
Follow-up question: وماذا عن كنبة Zahra؟
Standalone question: كم مدة ضمان كنبة Zahra؟

Previous question: Is the Yara armchair available in grey?
Follow-up question: What about the Dunes armchair?
Standalone question: Is the Dunes armchair available in grey?

Previous question: وين يقع فرع ينبع؟
Follow-up question: وش رقم الـ phone حقه؟
Standalone question: ما رقم هاتف فرع ينبع؟

Previous question: ما سعر مكيف Falcon؟
Follow-up question: How long does installation take?
Standalone question: How long does it take to install the Falcon air conditioner?"""

REWRITE_PROMPTS = {
    "v1": f"{_RULES_V1}\n\n{_EXAMPLES_V1}",
    "v2": f"{_RULES_V2}\n\n{_EXAMPLES_BALANCED}",
    "v3": f"{_RULES_V1}\n\n{_EXAMPLES_BALANCED}",
}


def build_rewrite_messages(query: str, previous: Turn, answer_chars: int, prompt: str = "v1") -> list[dict[str, str]]:
    lines = [f"Previous question: {previous.standalone_query}"]
    if previous.status == "answered":
        lines.append(f"Previous answer: {strip_markers(previous.answer)[:answer_chars]}")
    lines.append(f"Follow-up question: {query}")
    lines.append("Standalone question:")
    return [{"role": "system", "content": REWRITE_PROMPTS[prompt]}, {"role": "user", "content": "\n".join(lines)}]


def clean_rewrite(text: str) -> str:
    line = next((line for line in text.strip().splitlines() if line.strip()), "")
    line = re.sub(r"^(standalone question|rewritten question)\s*:\s*", "", line.strip(), flags=re.IGNORECASE)
    return line.strip().strip("\"'«»“”`").strip()


@dataclass(frozen=True)
class Rewrite:
    query: str
    applied: bool
    reason: str
    latency_ms: float
    fallback: bool = False


class QueryRewriter:
    def __init__(self, generation: GenerationConfig, settings: InteractiveConfig) -> None:
        self.generation = generation
        self.settings = settings
        self.client = httpx.Client(base_url=generation.url, timeout=settings.rewrite_timeout_s)

    def complete(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.generation.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.settings.rewrite_max_tokens,
            "seed": 0,
        }
        response = self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def rewrite(self, query: str, history: list[Turn]) -> Rewrite:
        decision = gate(query, history)
        if not decision.rewrite:
            return Rewrite(query=query, applied=False, reason=decision.reason, latency_ms=0.0)
        previous = history[-1]
        started = perf_counter()
        try:
            rewritten = clean_rewrite(self.complete(build_rewrite_messages(query, previous, self.settings.answer_context_chars, self.settings.rewrite_prompt)))
            fallback = not rewritten or len(rewritten) > 4 * max(len(query), len(previous.standalone_query))
        except httpx.HTTPError:
            rewritten, fallback = "", True
        if fallback:
            rewritten = f"{previous.standalone_query} {query}"
        latency = round((perf_counter() - started) * 1000, 2)
        return Rewrite(query=rewritten, applied=True, reason=decision.reason, latency_ms=latency, fallback=fallback)
