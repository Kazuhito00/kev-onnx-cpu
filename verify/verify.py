# -*- coding: utf-8 -*-
"""Gate: our answers vs open-jev's own, on the same ONNX weights.

artifacts/golden.json was produced by running the reference implementation
(the `open-jev` npm package, which is what this protocol was ported from)
against onnx-community/kev-0.6b-ONNX. Any difference here is a porting bug,
not a model difference: both sides run the identical graph.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from kev_decide import Decider, choice, noul, score  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def rebuild(spec):
    """Turn a recorded question back into one of ours."""
    if spec["type"] == "choice":
        return choice(spec["instructions"], spec["options"], spec.get("descriptions"))
    if spec["type"] == "score":
        return score(spec["instructions"], spec["options"])
    return noul(spec["instructions"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/onnx")
    parser.add_argument("--variant", default="q4f16")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    golden = json.loads((ROOT / "artifacts/golden.json").read_text(encoding="utf-8"))
    decider = Decider(ROOT / args.model, variant=args.variant, intra_op_threads=args.threads)

    worst = 0.0
    agreed = total = failures = 0
    for case in golden["cases"]:
        questions = [rebuild(q) for q in case["questions"]]
        ours = decider.decide(case["state"], questions)
        bad = False
        lines = []
        for got, expected in zip(ours, case["answers"]):
            total += 1
            # open-jev's noul answer carries only p(yes); choice and score carry
            # the full table. Ours always carries the table, which is a superset.
            if expected["type"] == "noul":
                delta = abs(got["probability"] - expected["probability"])
            else:
                keys = expected["probabilities"]
                delta = max(abs(got["probabilities"][k] - keys[k]) for k in keys)
            worst = max(worst, delta)
            ours_pick = got.get("choice") or got.get("level") or (
                "yes" if got.get("answer") else "no")
            ref_pick = expected.get("choice") or expected.get("level") or (
                "yes" if expected.get("answer") else "no")
            same = ours_pick == ref_pick and delta < 1e-6
            agreed += same
            bad |= not same
            lines.append("      %-7s %s ours=%-16s ref=%-16s max|dp|=%.2e"
                         % (got["type"], "ok " if same else "XX ",
                            ours_pick, ref_pick, delta))
        failures += bad
        print("%s %-16s %d questions" % ("ok " if not bad else "XX ", case["id"], len(ours)))
        for line in lines:
            print(line)

    print("\nanswers %d/%d identical   worst |dp| %.3e" % (agreed, total, worst))
    print("failures: %d / %d" % (failures, len(golden["cases"])))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
