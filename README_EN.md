[[Japanese](README.md)/[English](README_EN.md)]

# kev-onnx-cpu

A minimal-dependency inference implementation that runs [onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX) (Kev, a typed-decision model) on CPU with only onnxruntime and numpy (GPU inference is also supported).<br>
It does not use torch, transformers or tokenizers.

# Features
- Minimal dependencies: only two packages are needed at run time, onnxruntime and numpy
- Two scripts are all you need to run it: `download_model.py` (fetch the model) and `demo_inference_text.py` (inference)
- Answers are bit-for-bit identical to the reference implementation ([open-jev](https://www.npmjs.com/package/open-jev) on npm): 12/12, max probability difference 1.11e-16 (float64 rounding error)
- All questions are answered in a single forward pass, so the cost per decision drops as you add questions (121 ms for 1 question → 56 ms/decision for 16, 8 threads)
- Peak RAM is about 553 MB, and the ONNX is 323 MB (q4f16)
- The ONNX is identical to the q4f16 of [onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX) and is distributed in [Releases](https://github.com/Kazuhito00/kev-onnx-cpu/releases/tag/v0.0.0)
- The preprocessing (Qwen byte-level BPE tokenizer, sequence packing, delimiter escaping) is ported to pure Python, with checks against the reference implementation included

# Purpose of This Repository
This repository is for verifying the following.
- How far Kev's typed decisions can be run with just two run-time dependencies
- Whether open-jev's preprocessing can be ported to pure Python and produce the same inputs
- How speed and memory look on CPU

See [artifacts/report.md](artifacts/report.md) (Japanese) for the measurements and the reasoning behind the port.

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
```
These two are all that is needed to run. Downloading the model uses only the standard library.<br>
Environments are managed with [uv](https://docs.astral.sh/uv/).<br>
Only the tokenizer cross-check (`check_tokenizer_fuzz`) needs `tokenizers` (the `verify` group).

# Installation

```bash
git clone https://github.com/Kazuhito00/kev-onnx-cpu
cd kev-onnx-cpu

# install only the run-time dependencies (onnxruntime and numpy)
uv sync
```

# Download Model
The ONNX files are in [Releases](https://github.com/Kazuhito00/kev-onnx-cpu/releases/tag/v0.0.0) (they are not in the repository).<br>
The script below places them in `models/onnx/onnx/`. It uses only standard-library HTTPS; neither git nor git-lfs is needed.
```bash
uv run download_model.py
```
- Fetches `model_q4f16.onnx` and `model_q4f16.onnx_data` (about 330 MB in total) and checks them against the SHA-256 pinned in the script (on a mismatch it deletes the file and stops)
- Does nothing if they are already present, and an interrupted download resumes with the same command
- The tokenizer and config files are included in the repository
- To fetch manually, put the two files from Releases into `models/onnx/onnx/`

Other ways to fetch:
```bash
uv run download_model.py --list                  # list profiles / variants
uv run download_model.py --source hf             # the same q4f16 from Hugging Face
uv run download_model.py --variant q4            # 4-bit MatMulNBits, fp32 elsewhere (Hugging Face)
uv run download_model.py --profile kev-4b        # Qwen3-4B base (Hugging Face, fetched to models/onnx-4b/)
```

| profile | base | size |
|---|---|---:|
| `kev-0.6b` | Qwen3-0.6B-Base | 323 MB |
| `kev-4b` | Qwen3-4B-Base | larger |

**Only 4-bit builds are published** (`q4f16` / `q4`). There is no fp32 baseline, so the effect of quantization cannot be measured.<br>
With `kev-4b`, its tokenizer and config files are also fetched to `models/onnx-4b/`, and you select it with `--model models/onnx-4b` (accuracy was verified for kev-0.6b only).

# Usage

### Example (Python)
```bash
uv run demo_inference_text.py
```

```python
from kev_decide import Decider, choice, score, noul

d = Decider("models/onnx", variant="q4f16")

d.decide("I was charged twice for my Pro subscription. Please refund it.", [
    choice("Which team should handle this?", ["billing", "technical", "sales"],
           {"billing": "charges, refunds, invoices"}),
    noul("The customer is angry."),
    score("How urgent is this?", ["low", "medium", "high", "critical"]),
])
```
All questions are answered in one forward pass. Nothing is generated, so the answer is always one of the options you supplied.<br>
This model is text only, so there is no image demo.

You can also give questions on the command line.
```bash
uv run demo_inference_text.py --state "The export button crashes in Safari." \
    --choice "severity=cosmetic,degraded,blocking" --noul "This blocks the release."
```

### Question types
| function | answer |
|---|---|
| `choice(instructions, options, descriptions=None)` | one of the options (`choice`, `confidence`, `probabilities`) |
| `score(instructions, levels)` | expected value on an ordered scale (`score`, `normalized`, `level`); the first level is the lowest |
| `noul(statement)` | whether the statement holds (`answer`, `probability`); options are fixed to `["no", "yes"]` |

### Input layout
```text
[STATE] state...
  [Q] instructions... [OPT] option_1 [END] [OPT] option_2 [END] [DECIDE]
  [Q] instructions... [OPT] ...      [END] [DECIDE]
```
- The graph takes only `input_ids` and `attention_mask`. Segments, positions and the block-causal mask are all derived in-graph from the delimiter tokens
- The readout is the logit at each option's `[END]` position, softmaxed within each question (the pointer head's score against that question's `[DECIDE]`)
- Caller text has `<|name|>` rewritten to `<¦name¦>`, so delimiters cannot be forged
- Delimiter token ids: `[STATE]`=151659 `[Q]`=151660 `[OPT]`=151648 `[END]`=151649 `[DECIDE]`=151661

### Running on GPU (optional)
With `--gpu`, inference runs on CUDA (or DirectML). If this onnxruntime build has no GPU provider, it prints a warning and runs on CPU.
```bash
uv run --no-sync demo_inference_text.py --gpu
uv run --no-sync verify/bench.py --gpu
```
```python
from kev_decide import Decider, gpu_providers

d = Decider("models/onnx", variant="q4f16", providers=gpu_providers())
```

Notes:
- Install `onnxruntime-gpu` (or `onnxruntime-directml` on Windows) instead of `onnxruntime`. They write into the same `onnxruntime/` directory, so do not install them together. After swapping, reinstall with `uv pip install --force-reinstall --no-deps onnxruntime-gpu`
- `uv run` reinstalls the CPU `onnxruntime` to match `pyproject.toml`, so use `uv run --no-sync` (or `UV_NO_SYNC=1`)
- The provider actually used is shown on the last line of the demo and in the `provider` field of `verify/bench.py`
- See the Performance section for measurements. Probabilities on GPU do not match CPU exactly (fp16 graph arithmetic; the check against the reference implementation was done on CPU only)

# Verification
```bash
uv run verify/run_all.py                                 # all gates
uv run --with tokenizers verify/run_all.py --full-fuzz   # full cross-check against Rust tokenizers
```

| gate | what it checks | needs |
|---|---|---|
| check_prompt | sequence structure and delimiter escaping (6 items) | run-time deps only |
| verify | answers vs the reference implementation (12/12, max diff 1.1e-16) | run-time deps only |
| check_limits | state truncation, 255 options, context overflow, empty input, forged delimiters, etc. | run-time deps only |
| check_tokenizer_fuzz | cross-check with Rust `tokenizers` (about 25k cases by default with 0 mismatches; 187,658 cases with `--full-fuzz`) | tokenizers |

`artifacts/golden.json` was produced by actually running npm's `open-jev`. Both sides run the same weights, so any difference is a porting bug.

### Performance (Core i7-12800H, CPU only, q4f16)
3 questions, about 64 tokens. Session load is about 3.9 s and peak RAM is about 553 MB.

| threads | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| median | 656 ms | 376 ms | 252 ms | 224 ms |

Varying the number of questions (8 threads). All questions share one forward pass, so the cost per decision falls as you add more.

| questions | tokens | median | per decision |
|---:|---:|---:|---:|
| 1 | 34 | 121 ms | 121 ms |
| 3 | 64 | 224 ms | 75 ms |
| 8 | 135 | 451 ms | 56 ms |
| 16 | 254 | 888 ms | 56 ms |

The same 3 questions on GPU (`--gpu`, NVIDIA GeForce RTX 3050 Ti Laptop GPU, onnxruntime-gpu):

| | median | session load | peak RAM |
|---|---:|---:|---:|
| CPU (8 threads) | 224 ms | 3.9 s | 553 MB |
| GPU (CUDA) | 21.7 ms | 2.3 s | 905 MB |

| | packages |
|---|---:|
| reference (`open-jev` + `@huggingface/transformers`, Node) | 46 |
| this implementation | 2 |

# Limitations
- Only 4-bit builds are published. There is no fp32 baseline, so the effect of quantization cannot be measured
- Text only; images are not supported
- `max_options` is 255, and state and branch are each capped at 8192 tokens (trained with state 384 / branch 1024). A state over the cap is truncated at the end and the answers carry `state_truncated`
- A single question that cannot fit in the context raises an error instead of silently dropping options
- `tokenizer.py` is specific to this checkpoint and rejects an unexpected `tokenizer.json` with an exception

# Project Structure

```text
README.md                # README (Japanese)
README_EN.md             # README (English)
LICENSE                  # Apache-2.0
pyproject.toml           # dependencies (two at run time; the verify group is the oracle for the cross-check)
uv.lock                  # uv lock file
download_model.py        # fetch the model (Releases / Hugging Face)
demo_inference_text.py   # inference demo
kev_decide/              # inference implementation (light dependencies only)
  runtime.py             #   ORT session, readout, Decider
  tokenizer.py           #   pure-Python Qwen byte-level BPE
  protocol.py            #   sequence packing and answer decoding
  _regex_translate.py    #   translate tokenizer.json's regex for Python
  _uniprops.py           #   generated: Unicode range tables
export/
  make_golden.mjs        # make the golden data with the reference implementation (open-jev, Node)
verify/                  # verification gates (run all with run_all.py)
artifacts/               # golden.json (ground truth for verification), report.md
models/                  # ONNX (weights are gitignored), tokenizer and config files
```

# Reference
- [onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX)
- [open-jev](https://www.npmjs.com/package/open-jev) (reference implementation)

# Author
Kazuhito Takahashi (https://x.com/KzhtTkhs)

# License
kev-onnx-cpu is under [Apache-2.0 license](LICENSE).<br>
The model weights follow the license of the original model ([onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX), Apache-2.0).
