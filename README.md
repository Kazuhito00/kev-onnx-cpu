[[Japanese](README.md)/[English](README_EN.md)]

# kev-onnx-cpu

[onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX)（Kev：型付き判定モデル）を、onnxruntime と numpy だけでCPU推論する最小依存の推論実装です。 ※GPU推論も可<br>
torch・transformers・tokenizers は使いません。

# Features
以下の特徴があります。
- 最小依存：実行時に必要なパッケージは onnxruntime と numpy の2つだけ
- 動かすのに必要なスクリプトは `download_model.py`（モデル取得）と `demo_inference_text.py`（推論）の2つ
- 判定は参照実装（npmの[open-jev](https://www.npmjs.com/package/open-jev)）とビット単位で一致（12/12、最大確率差 1.11e-16 = float64の丸め誤差）
- 全質問が1回の順伝播で決まるため、質問を増やすほど1判定あたりのコストが下がる（1問 121 ms → 16問で 56 ms/判定、8スレッド）
- ピークRAMは約 553 MB、ONNXは 323 MB（q4f16）
- ONNXは[onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX)のq4f16と同一のものを[Releases](https://github.com/Kazuhito00/kev-onnx-cpu/releases/tag/v0.0.0)で配布
- 公式の前処理（Qwen byte-level BPEトークナイザ、系列の組み立て、区切りの退避）を純Pythonで移植し、参照実装との突き合わせ検証を同梱

# Purpose of This Repository
以下の検証を目的としています。
- Kevの型付き判定を、実行時依存2つだけでどこまで動かせるか
- 参照実装（open-jev）の前処理を純Pythonに移植して、同じ入力が作れるか
- CPU推論で速度・メモリがどうなるか

実測値と移植の根拠は[artifacts/report.md](artifacts/report.md)を参照してください。

# Requirements
```
Python 3.10 or later

numpy        1.26 or later
onnxruntime  1.20 or later
```
実行に必要なのはこの2つだけです。モデルの取得も標準ライブラリのみで動きます。<br>
環境構築は[uv](https://docs.astral.sh/uv/)を使います。<br>
トークナイザの突き合わせ検証（`check_tokenizer_fuzz`）にだけ `tokenizers` が必要です（`verify` グループ）。

# Installation

```bash
git clone https://github.com/Kazuhito00/kev-onnx-cpu
cd kev-onnx-cpu

# 実行時の依存（onnxruntime と numpy）のみインストール
uv sync
```

# Download Model
ONNXファイルは[Releases](https://github.com/Kazuhito00/kev-onnx-cpu/releases/tag/v0.0.0)に置いています（リポジトリには含めていません）。<br>
以下のスクリプトで `models/onnx/onnx/` に取得します。標準ライブラリのHTTPSのみで動き、gitもgit-lfsも不要です。
```bash
uv run download_model.py
```
- `model_q4f16.onnx` と `model_q4f16.onnx_data`（合計 約330 MB）を取得し、スクリプトに固定したSHA-256と照合します（不一致ならファイルを削除して停止）
- 取得済みなら何もせず、中断した場合は同じコマンドで続きから再開します
- トークナイザと設定ファイルはリポジトリに含まれています
- 手動で取得する場合は、Releasesの2ファイルを `models/onnx/onnx/` に置いてください

他の取得方法です。
```bash
uv run download_model.py --list                  # profile / variant一覧
uv run download_model.py --source hf             # 同じq4f16をHugging Faceから取得
uv run download_model.py --variant q4            # 4bit MatMulNBits、他はfp32（Hugging Face）
uv run download_model.py --profile kev-4b        # Qwen3-4Bベース（Hugging Face、models/onnx-4b/ に取得）
```

| profile | ベース | サイズ |
|---|---|---:|
| `kev-0.6b` | Qwen3-0.6B-Base | 323 MB |
| `kev-4b` | Qwen3-4B-Base | より大きい |

**4bitビルドしか公開されていません**（`q4f16` / `q4`）。fp32の基準が無いため、量子化の影響は測定できません。<br>
`kev-4b` を使う場合は、`kev-4b` のトークナイザ・設定ファイルも `models/onnx-4b/` に取得され、`--model models/onnx-4b` で指定できます（精度検証はkev-0.6bのみ実施）。

# Usage

### 実行例(Python)
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
全質問が1回の順伝播で決まります。生成は行わないため、答えは必ず与えた選択肢のどれかです。<br>
このモデルはテキスト専用のため、画像のデモはありません。

コマンドラインから質問を指定することもできます。
```bash
uv run demo_inference_text.py --state "The export button crashes in Safari." \
    --choice "severity=cosmetic,degraded,blocking" --noul "This blocks the release."
```

### 質問の種類
| 関数 | 答え |
|---|---|
| `choice(instructions, options, descriptions=None)` | 選択肢から1つ（`choice`、`confidence`、`probabilities`） |
| `score(instructions, levels)` | 順序尺度の期待値（`score`、`normalized`、`level`）。先頭が最低 |
| `noul(statement)` | 文が成り立つか（`answer`、`probability`）。選択肢は `["no", "yes"]` 固定 |

### 入力レイアウト
```text
[STATE] state...
  [Q] instructions... [OPT] option_1 [END] [OPT] option_2 [END] [DECIDE]
  [Q] instructions... [OPT] ...      [END] [DECIDE]
```
- グラフの入力は `input_ids` と `attention_mask` だけ。セグメント・位置・ブロックcausalマスクはグラフ内で区切りトークンから導出される
- 読み出しは各選択肢の `[END]` 位置のlogitを、質問ごとにsoftmax（pointer headがその質問の `[DECIDE]` に対して採点した値）
- 呼び出し側のテキストは `<|name|>` → `<¦name¦>` に退避され、区切りを偽造できない
- 区切りトークンID: `[STATE]`=151659 `[Q]`=151660 `[OPT]`=151648 `[END]`=151649 `[DECIDE]`=151661

### GPUで実行する（任意）
`--gpu` を付けると、CUDA（または DirectML）で推論します。このonnxruntimeビルドにGPUプロバイダが無い場合は、警告を出してCPUで動きます。
```bash
uv run --no-sync demo_inference_text.py --gpu
uv run --no-sync verify/bench.py --gpu
```
```python
from kev_decide import Decider, gpu_providers

d = Decider("models/onnx", variant="q4f16", providers=gpu_providers())
```

注意点です。
- `onnxruntime` の代わりに `onnxruntime-gpu`（Windowsなら `onnxruntime-directml` も可）を入れます。両者は同じ `onnxruntime/` ディレクトリにファイルを書くため、併存させないでください。入れ替えたときは `uv pip install --force-reinstall --no-deps onnxruntime-gpu` で入れ直します
- `uv run` は `pyproject.toml` に合わせて `onnxruntime`（CPU版）を再インストールするため、`uv run --no-sync` を使ってください（`UV_NO_SYNC=1` でも同じです）
- 実際に使われたプロバイダは、デモの最終行と `verify/bench.py` の出力（`provider`）で確認できます
- 測定結果は「性能」の節を参照してください。GPUでの確率は、CPUと完全には一致しません（fp16グラフの演算差。参照実装との一致検証はCPUのみ）

# Verification
```bash
uv run verify/run_all.py                                 # 全ゲート一括
uv run --with tokenizers verify/run_all.py --full-fuzz   # Rust tokenizersとの全件突き合わせ
```

| ゲート | 内容 | 必要なもの |
|---|---|---|
| check_prompt | 系列の構造と区切りの退避（6項目） | 実行時依存のみ |
| verify | 参照実装との回答一致（12/12、最大差 1.1e-16） | 実行時依存のみ |
| check_limits | 状態の切り捨て・255選択肢・コンテキスト超過・空文字・区切りの偽造など | 実行時依存のみ |
| check_tokenizer_fuzz | Rust `tokenizers` との突合（既定は約2.5万件で不一致0、`--full-fuzz` で187,658ケース） | tokenizers |

`artifacts/golden.json` は、npmの `open-jev` を実際に動かして作った正解データです。同じ重みを両者が走らせるので、差が出たら移植のバグと断定できます。

### 性能（Core i7-12800H、CPUのみ、q4f16）
3問・約64トークンのケースです。セッションロードは約3.9秒、ピークRAMは約553 MBです。

| threads | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| median | 656 ms | 376 ms | 252 ms | 224 ms |

質問数を変えたときです（8スレッド）。全質問が1回の順伝播で決まるため、1判定あたりのコストは質問を増やすほど下がります。

| 質問数 | トークン | median | 1判定あたり |
|---:|---:|---:|---:|
| 1 | 34 | 121 ms | 121 ms |
| 3 | 64 | 224 ms | 75 ms |
| 8 | 135 | 451 ms | 56 ms |
| 16 | 254 | 888 ms | 56 ms |

GPU（`--gpu`、NVIDIA GeForce RTX 3050 Ti Laptop GPU、onnxruntime-gpu）で同じ3問を測った結果です。

| | median | セッション生成 | ピークRAM |
|---|---:|---:|---:|
| CPU（8スレッド） | 224 ms | 3.9 s | 553 MB |
| GPU（CUDA） | 21.7 ms | 2.3 s | 905 MB |

| | パッケージ数 |
|---|---:|
| 参照実装（`open-jev` + `@huggingface/transformers`、Node） | 46 |
| この実装 | 2 |

# Limitations
- 4bitビルドのみ公開されています。fp32の基準が無いため、量子化の影響は測れません
- テキストのみです。画像は非対応です
- `max_options` は255、stateとbranchはそれぞれ8192トークンが上限です（学習時はstate 384 / branch 1024）。stateが上限を超えると末尾を切り捨て、回答に `state_truncated` が付きます
- 1問が単独でコンテキストに収まらない場合は、選択肢を落とさず例外を出します
- `tokenizer.py` はこのチェックポイント専用で、想定外の `tokenizer.json` は例外で弾きます

# Project Structure

```text
README.md                # README（日本語）
README_EN.md             # README（英語）
LICENSE                  # Apache-2.0
pyproject.toml           # 依存定義（実行時は2つ。verifyグループは突合検証のoracle用）
uv.lock                  # uvのロックファイル
download_model.py        # モデル取得（Releases / Hugging Face）
demo_inference_text.py   # 推論デモ
kev_decide/              # 推論実装（軽い依存のみ）
  runtime.py             #   ORTセッション、読み出し、Decider
  tokenizer.py           #   Qwen byte-level BPEの純Python実装
  protocol.py            #   系列の組み立てと回答のデコード
  _regex_translate.py    #   tokenizer.jsonの正規表現をPython用に翻訳
  _uniprops.py           #   生成物: Unicode範囲表
export/
  make_golden.mjs        # 参照実装（open-jev）でゴールデンを作る（Node）
verify/                  # 検証ゲート（run_all.pyで一括）
artifacts/               # golden.json（検証の正解データ）、report.md
models/                  # ONNX（重みはgitignore）、トークナイザ・設定ファイル
```

# Reference
- [onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX)
- [open-jev](https://www.npmjs.com/package/open-jev)（参照実装）

# Author
高橋かずひと(https://x.com/KzhtTkhs)

# License
kev-onnx-cpu is under [Apache-2.0 license](LICENSE).<br>
モデルの重みは、元のモデル（[onnx-community/kev-0.6b-ONNX](https://huggingface.co/onnx-community/kev-0.6b-ONNX)、Apache-2.0）のライセンスに従います。
