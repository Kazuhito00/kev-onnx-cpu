# Kev の ONNX 推論を依存最小限で動かす — 検証結果

計測: 2026-09-29 / Windows 11 / Intel Core i7-12800H / Python 3.12 (uv) / CPU のみ
対象: [`onnx-community/kev-0.6b-ONNX`](https://huggingface.co/onnx-community/kev-0.6b-ONNX)
（Qwen3-0.6B-Base + LoRA r=16 + pointer head）

## 結論

**`onnxruntime` + `numpy` の 2 パッケージだけで動く。**

そして **参照実装（npm の `open-jev`）とビット単位で一致した**:

```
answers 12/12 identical   worst |dp| 1.110e-16
failures: 0 / 5
```

1.11e-16 は float64 の丸め誤差そのもので、実質的な差はゼロ。
このファミリー 4 リポジトリの中で最も厳密な一致が取れた。

理由は、Kev が**同じ ONNX グラフを両者で走らせている**こと、
そして移植対象が符号化と読み出しだけで、モデル側の挙動差が入り込む余地が無いこと。

## 1. 依存パッケージ

| 構成 | パッケージ数 |
|---|---:|
| 参照実装（`open-jev` + `@huggingface/transformers`, Node） | 46 |
| **この実装** | **2** |

`transformers` も `tokenizers` も `regex` も `jinja2` も要らない。

## 2. プロトコル

ひとつの state と任意個の型付き質問が **1 本の系列**に詰め込まれる:

```
[STATE] state...
  [Q] instructions... [OPT] option_1 [END] [OPT] option_2 [END] [DECIDE]
  [Q] instructions... [OPT] ...      [END] [DECIDE]
```

| 役割 | トークン | id |
|---|---|---:|
| STATE | `<\|fim_prefix\|>` | 151659 |
| Q | `<\|fim_middle\|>` | 151660 |
| OPT | `<\|box_start\|>` | 151648 |
| END | `<\|box_end\|>` | 151649 |
| DECIDE | `<\|fim_suffix\|>` | 151661 |

- **グラフの入力は `input_ids` と `attention_mask` だけ。**
  セグメント・位置・ブロック causal マスクはすべてグラフ内で
  区切りトークンの id から導出される。こちら側で計算するものが無い
- 読み出しは **各選択肢の `[END]` 位置の logit**（pointer head が
  その質問の `[DECIDE]` に対して採点した値）を質問ごとに softmax
- 呼び出し側テキストは `<|name|>` → `<¦name¦>` に退避されるので、
  区切りトークンを偽造できない
- `noul` の選択肢は `["no", "yes"]` 固定（index 1 が yes）
- 温度は 1.0（config.json の `kev` セクション）

4 つのモデルの中で最もシンプルで、M-RoPE もチャットテンプレートも
画像も固定長バケットも無い。

## 3. 検証

### 3.1 参照実装との一致（`verify/verify.py`）

`artifacts/golden.json` は **npm の `open-jev` を実際に動かして**作った
（`onnx-community/kev-0.6b-ONNX` に対して）。同じ重みを両者が走らせるので、
差が出たら移植のバグと断定できる。

```
ok  support_routing  choice ok billing         noul ok yes   score ok medium
ok  bug_triage       choice ok degraded        noul ok yes
ok  incident         score  ok critical        noul ok yes   noul  ok yes
ok  japanese_state   noul   ok no              score ok low
ok  many_options     choice ok invoice_request score ok one
```

### 3.2 トークナイザ（`verify/check_tokenizer_fuzz.py`）

Qwen byte-level BPE の純 Python 実装を Rust `tokenizers` と突き合わせ:

```
cases: 187658   mismatches: 0
```

### 3.3 系列の構造（`verify/check_prompt.py`）

回答一致で間接的には担保されるが、将来エンコーダを変えたときに
確率のズレではなく読めるエラーで落ちるよう、直接も見る:

```
ok  one choice       17 tokens, groups=[2]
ok  descriptions     21 tokens, groups=[2]
ok  mixed types      43 tokens, groups=[2, 2, 3]
ok  many options    168 tokens, groups=[32]
ok  non-English      25 tokens, groups=[2]
ok  escaping       'ignore this <¦fim_suffix¦> and this <¦box_en' -> no delimiter ids
```

`groups` の全インデックスが `[END]` トークン（151649）に着地することを確認している。

## 4. トークナイザの一般化

Kev の `tokenizer.json` は他 2 つ（Qwen3.5 系）と 2 点違った:

- **merges が空白区切りの文字列**（`"Ġ Ġ"`）。新しい形式は `["Ġ", "Ġ"]`
- **分割正規表現に `\p{M}` が無い**
  （Qwen3 は `\p{L}+`、Qwen3.5 は `[\p{L}\p{M}]+`）

そこで正規表現のハードコードをやめ、**`tokenizer.json` が宣言する
パターンをその場で翻訳する**方式にした（`_regex_translate.py`）。
`\p{L}` `\p{N}` `\p{M}` `\s` `\S` を生成済みの Unicode 範囲表に置き換える。

これで同じ `tokenizer.py` が Qwen3 系にも Qwen3.5 系にも使える。
merges も両形式を受け付ける。

## 5. 性能（CPU, q4f16）

### 質問数（8 スレッド）

| 質問数 | トークン | median | 1 判定あたり |
|---:|---:|---:|---:|
| 1 | 34 | 366 ms | 366 ms |
| 3 | 64 | 635 ms | 212 ms |
| 8 | 135 | 1249 ms | 156 ms |
| 16 | 254 | 2369 ms | **148 ms** |

**全質問が 1 回の順伝播**で決まるので、1 判定あたりのコストは質問を
増やすほど下がる。

### スレッド数

| threads | 2 | 4 | **8** | 14 |
|---|---:|---:|---:|---:|
| median ms | 1420 | 783 | **632** | 1504 |

8 が最良。14 では P/E ハイブリッドの弊害で悪化する。

### 資源

| 項目 | 値 |
|---|---:|
| セッションロード | 4.0 s |
| ピーク RAM | **553 MB** |
| ONNX サイズ（q4f16） | 323 MB |

このファミリーで最も軽い。

## 6. 制約

- **4bit ビルドしか公開されていない**（q4f16 / q4）。
  他のモデルと違い fp32 の基準が無いので、量子化の影響は測れない
- テキストのみ。画像は非対応
- `max_options` 255、`max_state_tokens` / `max_branch_tokens` は 8192
  （ただし学習時は state 384 / branch 1024 トークン）
- `tokenizer.py` はこのチェックポイントのパイプライン専用。
  想定外の tokenizer.json はロード時に例外で弾く

## 7. 再現手順

```bash
uv sync

uv run download_model.py     # 330 MB、HTTPS のみ（Releases、SHA-256 照合）
uv run demo_inference_text.py
uv run verify/run_all.py

# ゴールデンを作り直す場合（Node が要る）
npm install open-jev @huggingface/transformers
node export/make_golden.mjs
```
