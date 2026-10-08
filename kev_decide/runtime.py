# -*- coding: utf-8 -*-
"""Kev on onnxruntime + numpy.

One forward pass answers every question in the request. The graph takes only
``input_ids`` and ``attention_mask``; segments, positions and the block-causal
mask are all derived in-graph from the delimiter ids, so there is nothing else
to compute here.

Runtime dependencies: onnxruntime and numpy.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

from .protocol import (
    Delimiters,
    Question,
    decode_answer,
    encode_sequence,
    escape,
    option_texts,
)
from .tokenizer import QwenBPE

__all__ = ["Decider", "MODEL_NAME", "VARIANTS", "gpu_providers"]

MODEL_NAME = "kev"
VARIANTS = ("q4f16", "q4")


GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")


def gpu_providers() -> List[str]:
    """GPU execution providers available in this onnxruntime build, CPU last."""
    available = ort.get_available_providers()
    return [p for p in GPU_PROVIDERS if p in available] + ["CPUExecutionProvider"]


def _session(path: pathlib.Path, threads: Optional[int],
             providers: Optional[Sequence[str]] = None) -> ort.InferenceSession:
    if not path.exists():
        raise FileNotFoundError("%s not found (download it with `uv run download_model.py`)" % path)
    options = ort.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    return ort.InferenceSession(
        str(path), sess_options=options, providers=list(providers) if providers else ["CPUExecutionProvider"]
    )


class Decider:
    """Load once, then answer typed questions about a state."""

    def __init__(
        self,
        model_dir: Union[str, pathlib.Path],
        *,
        variant: str = "q4f16",
        graph: Optional[str] = None,
        intra_op_threads: Optional[int] = None,
        temperature: Optional[float] = None,
        providers: Optional[Sequence[str]] = None,
    ):
        if variant not in VARIANTS:
            raise ValueError("unknown variant %r; pick one of %s" % (variant, list(VARIANTS)))
        model_dir = pathlib.Path(model_dir)
        self.model_dir = model_dir
        self.variant = variant

        self.session = _session(
            model_dir / "onnx" / (graph or "model_%s.onnx" % variant), intra_op_threads,
            providers,
        )
        self.tokenizer = QwenBPE(str(model_dir / "tokenizer.json"))

        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        section = config.get("kev") or {}
        if not section:
            raise ValueError("config.json has no 'kev' section; is this a Kev checkpoint?")
        self.delimiters = Delimiters.from_config(config, self.tokenizer)
        self.temperature = float(
            temperature if temperature is not None else section.get("temperature", 1.0)
        )
        self.max_state_tokens = int(section.get("max_state_tokens", 8192))
        self.max_length = int(section.get("max_branch_tokens", 8192))
        self.max_options = int(section.get("max_options", 255))

    # ------------------------------------------------------------- encoding

    def encode_text(self, text: str) -> List[int]:
        """Tokenise caller text: escaped, no special tokens."""
        return self.tokenizer.encode(escape(text))

    def encode(self, state: str, questions: Sequence[Question]):
        for index, question in enumerate(questions):
            count = len(option_texts(question))
            if not 1 <= count <= self.max_options:
                raise ValueError(
                    "question %d has %d options; this model allows 1-%d"
                    % (index, count, self.max_options)
                )
        tokenised = [
            (self.encode_text(question.instructions),
             [self.encode_text(option) for option in option_texts(question)])
            for question in questions
        ]
        return encode_sequence(
            self.encode_text(state), tokenised, self.delimiters,
            max_state_tokens=self.max_state_tokens, max_length=self.max_length,
        )

    # -------------------------------------------------------------- forward

    def logits(self, input_ids: Sequence[int]) -> np.ndarray:
        """One logit per token, from the pointer head."""
        ids = np.asarray([list(input_ids)], dtype=np.int64)
        outputs = self.session.run(
            None, {"input_ids": ids, "attention_mask": np.ones_like(ids)}
        )
        return np.asarray(outputs[0]).reshape(-1).astype(np.float64)

    # --------------------------------------------------------------- decide

    def decide(self, state: str, questions: Sequence[Question]) -> List[Dict[str, Any]]:
        """Answer every question in one forward pass."""
        encoded = self.encode(state, questions)
        scores = self.logits(encoded.input_ids)

        answers = []
        for question, group in zip(questions, encoded.groups):
            answer = decode_answer(question, [scores[i] for i in group], self.temperature)
            answer["instructions"] = question.instructions
            answers.append(answer)
        if encoded.state_truncated:
            for answer in answers:
                answer["state_truncated"] = True
        return answers
