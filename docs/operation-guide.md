# 操作方法ガイド

このガイドでは、準備から結果の確認、設定の調整までを手順どおりに説明します。
ツールの概要は [README](../README.md) を参照してください。

---

## 1. 準備する

### 1-1. Python環境を作る

必要なもの:

- Python 3.13以上
- [uv](https://docs.astral.sh/uv/)

プロジェクトのフォルダで次のコマンドを実行します。

```bash
uv sync
```

依存関係が `.venv/` に入ります。以降のコマンドは `uv run` を付けて実行します。

### 1-2. Ollamaを用意する(LLM要約を使う場合)

LLM要約を使わないなら、この手順は省略できます。

1. Ollamaをインストールします。
   - macOS / Linux: `curl -fsSL https://ollama.com/install.sh | sh`
   - Windows: 公式サイト（https://ollama.com/download）からインストーラーを実行します。
2. Ollamaが起動していることを確認します。

   ```bash
   ollama list
   ```

   エラーが出る場合は、`ollama serve` で起動してください。
3. 使うモデルをダウンロードします。

   ```bash
   ollama pull gemma2:2b
   ```

   モデルが無い場合、ツールは実行時に自動でダウンロードしますが、時間がかかるため先に取得することを推奨します。

### 1-3. 埋め込みモデルについて

初回の実行時に、埋め込みモデル（既定は `cl-nagoya/ruri-v3-310m`、約1.2GB）が自動でダウンロードされます。
ダウンロードが終わるまで、待ってください。

---

## 2. 文書を用意する

分析する文書は、Pythonのリスト（文字列の一覧）として渡します。
1件が1つの文字列です。

### 方法A: リストを直接書く

```python
documents = [
    "機械学習の基礎についての解説記事です。",
    "深層学習とニューラルネットワークの比較",
    "自然言語処理の歴史と応用",
]
```

### 方法B: CSVから読み込む

1. 文書を1列にまとめたCSVを用意します。文字コードはUTF-8にしてください。
2. 空欄の行は取り除いておくと安心です。
3. 次のように読み込みます。

   ```python
   import pandas as pd

   df = pd.read_csv("documents.csv")
   documents = df["text_column"].dropna().tolist()  # text_column は列名に置き換える
   ```

### 注意点

- 文書数は、クラスタ数（`n_clusters`）以上にしてください。文書数のほうが少ないと、ツールは `n_clusters` を文書数に合わせて自動で減らします。
- 文書数が少ないと、TF-IDFの警告が出ることがあります。文書数が少ない場合は正常です。

---

## 3. 実行する

### 3-1. サンプルで試す

リポジトリに入っている `main.py` を実行すると、組み込みのサンプル（20件）で動作を確認できます。

```bash
uv run python main.py
```

サンプルは `qwen2.5:7b` を使う設定です。Ollamaを使わない場合は、`main.py` の `use_llm_summary=True` を `False` に変えてから実行してください。

### 3-2. 自分の文書で実行する

プロジェクトのフォルダに `run_mece.py` を作ります。

```python
from pathlib import Path

import pandas as pd

from main import MECEAnalyzer, MECEConfig

documents = pd.read_csv("documents.csv")["text_column"].dropna().tolist()

config = MECEConfig(
    n_clusters=5,
    use_llm_summary=False,  # LLM要約を使うなら True にして llm_model_name を指定する
    output_dir=Path("mece_output"),
)

analyzer = MECEAnalyzer(config)
analyzer.load_documents(documents).run()
```

実行します。

```bash
uv run python run_mece.py
```

`MECEAnalyzer(config)` の時点で埋め込みモデルが読み込まれるため、初回は起動に時間がかかります。

### 3-3. 処理の流れ

`run()` は次の順に処理します。各段階の開始時に `▶` で始まる行が表示されるので、止まった位置から失敗した段階を特定できます。

1. 文書のベクトル化
2. 次元削減（PCA・UMAP）
3. クラスタリング（KMeans・HDBSCAN）
4. クラスタ要約（キーワード・代表文・LLM要約）
5. 重複度とカバレッジの計算
6. 可視化
7. ファイルへの保存

---

## 4. 結果を確認する

結果は `output_dir` で指定したフォルダ（既定は `mece_output/`）に保存されます。
最初に見るべきファイルは `cluster_summary.md` です。

| ファイル | 中身 | 見る目的 |
|---|---|---|
| `cluster_summary.md` | MECE指標とクラスタごとの要約 | 全体把握。最初に開く |
| `cluster_summary.csv` | クラスタごとの要約（表形式） | 表計算ソフトで並べ替えたいとき |
| `documents_with_clusters.csv` | 文書ごとのクラスタ割り当て | どの文書がどのクラスタに入ったか確認する |
| `cluster_overlaps.csv` | クラスタ間の重複度 | 重複候補の確認。重複が無いと作られない |
| `mece_metadata.csv` | 全体の統計（文書数、カバレッジなど） | 分析条件と結果の記録 |
| `mece_visualization.png` | PCA・UMAPの散布図とSilhouette分布 | クラスタの形と重なりを画像で見る |
| `mece_interactive.html` | 操作できる散布図 | ブラウザで開き、点にカーソルを合わせて文書を読む |

### 指標の読み方

- **凝集度（cohesion）**: 同じクラスタ内の文書がどれだけ近いかを示す値（Silhouette係数）。値が低いクラスタは、境界が曖昧です。
- **カバレッジ**: HDBSCANでどのクラスタにも入らなかった文書（ノイズ）以外の割合。ノイズが多いと、カバレッジは下がります。
- **重複度（overlap）**: 2つのクラスタの中心どうしのコサイン類似度（0〜1）。`overlap_threshold`（既定 0.3）以上のペアだけが `cluster_overlaps.csv` に出力されます。

値はMECE判断の材料です。値が良くてもMECEとは限らないため、最終的な判断は内容を読んで行ってください。

---

## 5. 結果をもとに設定を調整する

結果が思わしくない場合は、次の表を参考に設定を変えて、もう一度実行します。

| 状況 | 変える設定 | 例 |
|---|---|---|
| クラスタが大まかすぎる | `n_clusters` を増やす | `n_clusters=6` |
| クラスタが細かすぎる | `n_clusters` を減らす | `n_clusters=3` |
| 重複候補が多すぎる | `n_clusters` を減らす | `n_clusters=4` |
| 重複候補をもっと厳しく見たい | `overlap_threshold` を下げる | `overlap_threshold=0.2` |
| 日本語の表現が合わない | 埋め込みモデルを変える | `embedding_model="intfloat/multilingual-e5-base"` |
| 要約が遅い、または長すぎる | 生成トークン上限を下げる | `llm_max_tokens=64` |
| メモリ不足で止まる | 軽量モデル、またはLLM要約をオフにする | `llm_model_name="gemma2:2b"` / `use_llm_summary=False` |

### 結果を残しながら試す

同じフォルダに保存すると、ファイルは上書きされます。
条件を変えて比べたいときは、`output_dir` を変えてください。

```python
config = MECEConfig(
    n_clusters=6,
    output_dir=Path("mece_output_k6"),
)
```

---

## 6. よく使う設定の一覧

`MECEConfig` の主な設定と既定値です。全項目は [README](../README.md) の「詳細設定」を参照してください。

| 設定 | 既定値 | 意味 |
|---|---|---|
| `n_clusters` | 3 | クラスタ数（KMeans） |
| `embedding_model` | `cl-nagoya/ruri-v3-310m` | 文書をベクトル化するモデル |
| `use_llm_summary` | `True` | LLMでクラスタを要約するか |
| `llm_backend` | `"ollama"` | `"ollama"` または `"transformers"` |
| `llm_model_name` | `"gemma2:2b"` | 要約に使うモデル名 |
| `llm_max_tokens` | 128 | 要約の生成トークン上限 |
| `overlap_threshold` | 0.3 | 重複候補とみなす類似度の下限 |
| `output_dir` | `Path("mece_output")` | 出力先フォルダ |
| `random_state` | 42 | 乱数シード（同じ条件で同じ結果を得るため） |

---

## 7. 動作確認（開発者向け）

補助関数のテストを実行します。

```bash
uv run python test_main_helpers.py
```

`ok` と表示されれば成功です。

---

## 8. 困ったとき

| 症状 | 対処 |
|---|---|
| `Ollama接続エラー` | Ollamaが起動しているか確認する（`ollama serve`）。 |
| `モデル 'xxx' が見つかりません` | `ollama pull <モデル名>` で取得する。 |
| `Warning: LLM初期化失敗` | LLM要約はスキップされ、処理は続行します。要約が不要なら問題ありません。 |
| メモリ不足 | 上の「設定を調整する」表を参照する。 |
| 日本語が図で文字化けする | `main.py` の `plt.rcParams` のフォント設定を、自分の環境にあるフォントに変える。 |

そのほかの問題は、[README](../README.md) の「トラブルシューティング」も参照してください。
