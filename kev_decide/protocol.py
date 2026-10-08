# -*- coding: utf-8 -*-
"""The Kev request protocol, without transformers.

Ported from open-jev (``src/encoding.ts``, ``src/questions.ts``,
``src/answers.ts``) and the ``kev`` section of the checkpoint's config.json.

One state and any number of typed questions pack into a single sequence::

    [STATE] state...
      [Q] instructions... [OPT] option_1 [END] [OPT] option_2 [END] [DECIDE]
      [Q] instructions... [OPT] ...      [END] [DECIDE]

The model derives the block-causal mask from the delimiter ids in-graph, so
every question sees the state and itself only. A pointer head scores each
option's ``[END]`` token against its question's ``[DECIDE]`` token: read the
logit at each ``[END]`` position and softmax within the question.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Question", "choice", "score", "noul",
    "NOUL_OPTIONS", "Delimiters", "escape", "option_texts", "labels_of",
    "encode_sequence", "decode_answer", "EncodedSequence",
]

#: What the models were trained with for noul questions; index 1 is "yes".
NOUL_OPTIONS = ("no", "yes")

#: Caller text can never forge a delimiter: <|name|> becomes <¦name¦>.
_DELIMITER_LIKE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def escape(text: str) -> str:
    """open-jev :: kevFamily.escape."""
    return _DELIMITER_LIKE.sub(r"<¦\1¦>", text)


@dataclass(frozen=True)
class Delimiters:
    """The five marker token ids, read from config.json's `kev` section."""

    state: int
    question: int
    option_start: int
    option_end: int
    decide: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any], tokenizer=None) -> "Delimiters":
        section = config.get("kev") or {}
        ids = section.get("delimiter_ids") or {}
        names = section.get("delimiters") or {}
        fallback = {"state": "<|fim_prefix|>", "question": "<|fim_middle|>",
                    "option_start": "<|box_start|>", "option_end": "<|box_end|>",
                    "decide": "<|fim_suffix|>"}
        resolved = {}
        for key, default_name in fallback.items():
            if isinstance(ids.get(key), int):
                resolved[key] = ids[key]
                continue
            if tokenizer is None:
                raise ValueError(
                    "config.json has no delimiter_ids[%r] and no tokenizer was given" % key
                )
            token_ids = tokenizer.encode(names.get(key, default_name))
            if len(token_ids) != 1:
                raise ValueError("delimiter %r is not a single token" % key)
            resolved[key] = token_ids[0]
        return cls(**resolved)


@dataclass
class Question:
    type: str
    instructions: str
    options: Tuple[str, ...] = ()
    descriptions: Dict[str, str] = field(default_factory=dict)


def choice(instructions: str, options: Sequence[str],
           descriptions: Optional[Mapping[str, str]] = None) -> Question:
    """Pick one of the given options."""
    return Question("choice", instructions, tuple(options), dict(descriptions or {}))


def score(instructions: str, levels: Sequence[str]) -> Question:
    """Rate on an ordered scale; the first level is the lowest."""
    return Question("score", instructions, tuple(levels))


def noul(statement: str) -> Question:
    """Does the statement hold for the state?"""
    return Question("noul", statement)


def labels_of(question: Question) -> Tuple[str, ...]:
    """The keys of the answer distribution."""
    return NOUL_OPTIONS if question.type == "noul" else question.options


def option_texts(question: Question) -> List[str]:
    """The option strings as the model sees them (``name: description``)."""
    if question.type == "noul":
        return list(NOUL_OPTIONS)
    if question.type == "choice" and question.descriptions:
        return [
            "%s: %s" % (option, question.descriptions[option])
            if question.descriptions.get(option) else option
            for option in question.options
        ]
    return list(question.options)


@dataclass
class EncodedSequence:
    input_ids: List[int]
    #: For each question, the index of every option's [END] token.
    groups: List[List[int]]
    state_tokens: int
    state_truncated: bool


def encode_sequence(
    state: Sequence[int],
    questions: Sequence[Tuple[Sequence[int], Sequence[Sequence[int]]]],
    delimiters: Delimiters,
    max_state_tokens: int = 8192,
    max_length: int = 8192,
) -> EncodedSequence:
    """open-jev :: encodeKevSequence.

    ``questions`` is a list of (instruction_ids, [option_ids, ...]); the caller
    tokenises those, because tokenisation is the runtime's job.
    """
    branches = []
    for instructions, options in questions:
        branch: List[int] = [delimiters.question, *instructions]
        ends: List[int] = []
        for option in options:
            branch.extend([delimiters.option_start, *option, delimiters.option_end])
            ends.append(len(branch) - 1)
        branch.append(delimiters.decide)
        branches.append((branch, ends))

    longest = max(len(branch) for branch, _ in branches)
    budget = max_length - 1 - longest          # the [STATE] token counts too
    if budget < 0:
        raise ValueError(
            "a question needs %d tokens, which exceeds the %d token context; "
            "shorten the instructions or options" % (longest + 1, max_length)
        )

    limit = min(max_state_tokens, budget)
    kept = list(state[:limit])

    input_ids: List[int] = [delimiters.state, *kept]
    groups: List[List[int]] = []
    for branch, ends in branches:
        base = len(input_ids)
        input_ids.extend(branch)
        groups.append([base + end for end in ends])

    return EncodedSequence(input_ids, groups, len(kept), len(kept) < len(state))


def _softmax(values: Sequence[float], temperature: float = 1.0) -> List[float]:
    scaled = [v / temperature for v in values]
    top = max(scaled)
    weights = [math.exp(v - top) for v in scaled]
    total = sum(weights)
    return [w / total for w in weights]


def decode_answer(question: Question, logits: Sequence[float],
                  temperature: float = 1.0) -> Dict[str, Any]:
    """open-jev :: decodeAnswer."""
    probabilities = _softmax(logits, temperature)
    labels = labels_of(question)
    best = max(range(len(probabilities)), key=lambda i: probabilities[i])
    table = {label: probabilities[i] for i, label in enumerate(labels)}

    if question.type == "choice":
        return {"type": "choice", "choice": labels[best],
                "confidence": probabilities[best], "probabilities": table}

    if question.type == "score":
        expected = sum(p * i for i, p in enumerate(probabilities))
        last = len(labels) - 1
        nearest = min(last, max(0, round(expected)))
        return {"type": "score", "score": expected,
                "normalized": expected / last if last > 0 else 0.0,
                "level": labels[nearest], "confidence": probabilities[best],
                "probabilities": table}

    probability = table["yes"]
    return {"type": "noul", "answer": probability >= 0.5, "probability": probability,
            "confidence": max(probability, 1 - probability), "probabilities": table}
