# -*- coding: utf-8 -*-
"""Typed decisions with Kev, on onnxruntime + numpy.

    uv run demo_inference_text.py
    uv run demo_inference_text.py --state "..." --choice "team=billing,technical"

One state and every question pack into a single sequence, and one forward pass
answers all of them: a pointer head scores each option's [END] token against its
question's [DECIDE] token. Nothing is generated, so the answer is always one of
the options supplied.

Kev is text only; there is no demo_inference_image.py.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from kev_decide import Decider, choice, gpu_providers, noul, score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent

EXAMPLE_STATE = (
    "Hi, I was charged twice for my Pro subscription this month. "
    "Please refund the duplicate payment."
)
EXAMPLE_QUESTIONS = [
    choice("Which team should handle this request?",
           ["billing", "technical", "sales"],
           {"billing": "charges, refunds, invoices",
            "technical": "bugs and outages",
            "sales": "pricing and upgrades"}),
    noul("The customer is angry."),
    score("How urgent is this?", ["low", "medium", "high", "critical"]),
]


def parse_questions(args):
    questions = []
    for spec in args.choice or []:
        name, _, rest = spec.partition("=")
        options = [v.strip() for v in rest.split(",") if v.strip()]
        questions.append(choice(args.instructions or name.strip(), options))
    for spec in args.score or []:
        name, _, rest = spec.partition("=")
        levels = [v.strip() for v in rest.split(",") if v.strip()]
        questions.append(score(args.instructions or name.strip(), levels))
    for statement in args.noul or []:
        questions.append(noul(statement))
    return questions


def show(answer: dict) -> None:
    spread = "  ".join(
        "%s=%.3f" % kv
        for kv in sorted(answer["probabilities"].items(), key=lambda kv: -kv[1])
    )
    if answer["type"] == "choice":
        head, extra = answer["choice"], ""
    elif answer["type"] == "score":
        head = answer["level"]
        extra = "  (expected %.2f, normalised %.2f)" % (answer["score"], answer["normalized"])
    else:
        head = "yes" if answer["answer"] else "no"
        extra = "  (p(yes)=%.3f)" % answer["probability"]
    print("%-7s %-22s %.4f%s\n        %s"
          % (answer["type"], head, answer["confidence"], extra, spread))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--state", help="omit to run the built-in example")
    parser.add_argument("--choice", action="append", metavar="NAME=a,b,c")
    parser.add_argument("--score", action="append", metavar="NAME=low,...,high")
    parser.add_argument("--noul", action="append", metavar="STATEMENT")
    parser.add_argument("--instructions", default="")
    parser.add_argument("--model", default="models/onnx")
    parser.add_argument("--variant", default="q4f16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gpu", action="store_true",
                        help="use CUDA / DirectML if this onnxruntime build has it "
                             "(needs onnxruntime-gpu or onnxruntime-directml); falls back to CPU")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.state is None:
        state, questions = EXAMPLE_STATE, EXAMPLE_QUESTIONS
    else:
        state = args.state
        questions = parse_questions(args)
        if not questions:
            parser.error("give at least one --choice, --score or --noul")

    started = time.perf_counter()
    providers = gpu_providers() if args.gpu else None
    if args.gpu and providers[0] == "CPUExecutionProvider":
        print("warning: no GPU provider in this onnxruntime build; running on CPU "
              "(install onnxruntime-gpu or onnxruntime-directml instead of onnxruntime)",
              file=sys.stderr)
    decider = Decider(ROOT / args.model, variant=args.variant,
                      intra_op_threads=args.threads, providers=providers)
    load = time.perf_counter() - started
    if args.gpu and providers[0] != "CPUExecutionProvider" \
            and decider.session.get_providers()[0] == "CPUExecutionProvider":
        print("warning: %s failed to load; fell back to CPU (check CUDA / cuDNN DLLs on PATH)"
              % providers[0], file=sys.stderr)

    encoded = decider.encode(state, questions)
    started = time.perf_counter()
    answers = decider.decide(state, questions)
    elapsed = time.perf_counter() - started

    if args.json:
        print(json.dumps(answers, ensure_ascii=False, indent=2))
        return 0

    print("state:\n  %s\n" % state[:160])
    for answer in answers:
        show(answer)
    print("\n%d questions packed into %d tokens, answered in one forward pass"
          % (len(answers), len(encoded.input_ids)))
    print("load %.1f s, inference %.2f s (%s, %s)"
          % (load, elapsed, args.variant, decider.session.get_providers()[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
