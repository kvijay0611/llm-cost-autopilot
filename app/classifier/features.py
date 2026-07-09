"""
Phase 2 — Feature extraction for the complexity classifier.

Kept deliberately simple and interpretable: this is the routing skeleton,
not a place to over-engineer. Every feature here is something you could
explain to an interviewer in one sentence.
"""
import re
import numpy as np

FEATURE_NAMES = [
    "token_count",
    "has_analysis_verb",
    "has_creative_verb",
    "num_constraints",
    "has_context",
    "wants_structured_output",
    "num_sentences",
    "avg_word_len",
    "question_count",
]

_ANALYSIS_VERBS = re.compile(
    r"\b(analyze|compare|evaluate|assess|reason|explain why|justify|critique|"
    r"weigh|synthesize|derive|prove)\b",
    re.IGNORECASE,
)
_CREATIVE_VERBS = re.compile(
    r"\b(write a story|compose|imagine|brainstorm|invent|design a|come up with|"
    r"draft a creative|generate ideas)\b",
    re.IGNORECASE,
)
_STRUCTURED_HINTS = re.compile(
    r"\b(json|table|csv|bullet|markdown|schema|format as|return as)\b",
    re.IGNORECASE,
)
_CONSTRAINT_HINTS = re.compile(
    r"\b(must|should|only|exactly|no more than|at least|within|excluding|"
    r"and also|additionally|furthermore)\b",
    re.IGNORECASE,
)


def extract_features(prompt: str, context: str | None = None) -> np.ndarray:
    full_text = prompt if not context else f"{prompt}\n{context}"
    tokens = full_text.split()
    sentences = re.split(r"[.!?]+", prompt.strip())
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r"[A-Za-z']+", prompt)

    token_count = len(tokens)
    has_analysis_verb = 1.0 if _ANALYSIS_VERBS.search(prompt) else 0.0
    has_creative_verb = 1.0 if _CREATIVE_VERBS.search(prompt) else 0.0
    num_constraints = float(len(_CONSTRAINT_HINTS.findall(prompt)))
    has_context = 1.0 if context and context.strip() else 0.0
    wants_structured_output = 1.0 if _STRUCTURED_HINTS.search(prompt) else 0.0
    num_sentences = float(len(sentences))
    avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
    question_count = float(prompt.count("?"))

    return np.array(
        [
            token_count,
            has_analysis_verb,
            has_creative_verb,
            num_constraints,
            has_context,
            wants_structured_output,
            num_sentences,
            avg_word_len,
            question_count,
        ],
        dtype=float,
    )
