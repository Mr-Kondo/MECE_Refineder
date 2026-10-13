# MECEクラスタリング補助ツール

思考内容をMECE（Mutually Exclusive, Collectively Exhaustive）に分割するための判断材料を、
機械学習で自動生成するツールです。

## 📖 概要

このツールは「MECEを自動生成する」ものではありません。
「人間がMECE判断を下すための材料を、機械で最大限用意する」ことを目的に作られています。

### 設計思想

- MECEは視点や目的に依存するため、最終判断は人間が行う
- 機械は「分散の大きい切り口」「意味的に似た塊」「各塊の傾向」を提示する
- クラスタ同士の重複や曖昧さ（MECE違反の候補）を可視化する

### なぜこのツールが必要か

- 大量の文書・アイデアを手作業で分類するのは時間がかかる
- 主観的な分類では、見落としや重複が起きやすい
- ローカルLLMの要約により、各クラスタの「視点」を自然言語で理解できる

---

## 📘 ドキュメント

- **[操作方法ガイド](docs/operation-guide.md)**: 準備、実行、結果の確認、設定の調整を手順どおりに説明しています。

---

## ✨ 主な機能

### 1. 意味ベクトル化

- Sentence-BERTで文書を意味空間にマッピングします。
- 既定は日本語に強い `ruri-v3-310m`（クラスタリング用の接頭辞 `トピック: ` を自動で付与）
- `multilingual-e5-base` にも対応（接頭辞 `query: ` を自動で付与）

### 2. 次元削減・可視化

- PCA: 線形な主成分分析
- UMAP: 非線形な意味空間の可視化

### 3. クラスタリング

- KMeans: クラスタ数を指定して、安定した分割を行います。
- HDBSCAN: 密度ベースで、クラスタ数を自動的に決めます。正規化済みの埋め込みに対して `euclidean` 距離を使います。

### 4. クラスタ要約

- TF-IDF: 代表キーワードを抽出します（SudachiPyで名詞に分割）。
- Medoid: 最も典型的な文書を選びます。
- **ローカルLLM要約**: 各クラスタの「視点・観点」を自然言語で要約します。
  - LLMは要約生成時に初期化されます（起動時間を短縮するため）。
  - 初期化に失敗した場合は、LLM要約を自動でスキップし、`use_llm_summary=False` として処理を続行します。

### 5. MECE判断支援

- クラスタ間の重複度（クラスタ中心のコサイン類似度。閾値は `overlap_threshold`）
- カバレッジ分析（ノイズ文書の割合）
- クラスタ内の凝集度（Silhouette係数）

### 6. tqdmによる進捗表示

- 全処理フェーズで進捗バー表示
- 処理時間の予測が可能

---

## 🚀 インストール

### 必須要件

- Python 3.13以上
- 8GB以上のRAM（推奨: 16GB以上）
- Ollama（ローカルLLM要約を使う場合）

### ライブラリのインストール

依存関係は `pyproject.toml` と `uv.lock` で管理しています。

```bash
uv sync
```

初回の実行時に、埋め込みモデル（ruri-v3-310m、約1.2GB）がダウンロードされます。

### Ollamaのインストール（LLM要約を使う場合）

#### macOS / Linux

```bash
# 公式サイトからインストールする
# https://ollama.com/download

# または、次のコマンドでインストールする
curl -fsSL https://ollama.com/install.sh | sh
```

#### Windows

公式サイトからインストーラーをダウンロードしてください。
https://ollama.com/download

---

## 📦 推奨モデル

### Ollama用（簡単・推奨）

| モデル | コマンド | サイズ | 特徴 |
|--------|---------|--------|------|
| **Gemma2 2B** | `ollama pull gemma2:2b` | 1.6GB | 軽量・高速 |
| **Qwen2.5 7B** | `ollama pull qwen2.5:7b` | 4.7GB | 日本語◎ |
| **Llama3.1 8B** | `ollama pull llama3.1:8b` | 4.7GB | バランス型 |
| **DeepSeek-R1** | `ollama pull deepseek-r1:8b` | 4.9GB | 推論特化 |

### 日本語特化モデル（手動セットアップが必要）

ELYZA、Swallow、LLM-jpなどの日本語特化モデルは、
HuggingFaceからggufファイルをダウンロードし、Modelfileで設定する必要があります。

詳細は「[日本語特化モデルのセットアップ](#-日本語特化モデルのセットアップ)」を参照してください。

---

## 💻 基本的な使い方

具体的な手順は [操作方法ガイド](docs/operation-guide.md) にもまとめています。

### 1. 最小構成（LLM要約なし）

```python
from pathlib import Path
from main import MECEAnalyzer, MECEConfig

# 分析対象の文書リスト
documents = [
    "機械学習の基礎についての解説記事です。",
    "深層学習とニューラルネットワークの比較",
    "自然言語処理の歴史と応用",
    # ...
]

# 設定（LLM要約なし・高速）
config = MECEConfig(
    n_clusters=5,
    use_llm_summary=False
)

# 実行
analyzer = MECEAnalyzer(config)
analyzer.load_documents(documents).run()
```

### 2. Ollama + LLM要約（推奨）

```python
config = MECEConfig(
    n_clusters=5,
    use_llm_summary=True,
    llm_backend="ollama",
    llm_model_name="gemma2:2b"  # 軽量・高速
)

analyzer = MECEAnalyzer(config)
analyzer.load_documents(documents).run()
```

### 3. 日本語性能を重視する

```python
config = MECEConfig(
    n_clusters=5,
    embedding_model="intfloat/multilingual-e5-base",
    use_llm_summary=True,
    llm_backend="ollama",
    llm_model_name="qwen2.5:7b"  # 日本語性能◎
)

analyzer = MECEAnalyzer(config)
analyzer.load_documents(documents).run()
```

### 4. CSVファイルから文書を読み込む

```python
import pandas as pd

# CSVから読み込む
df = pd.read_csv("documents.csv")
documents = df["text_column"].tolist()

config = MECEConfig(n_clusters=5)
analyzer = MECEAnalyzer(config)
analyzer.load_documents(documents).run()
```

---

## 🎯 出力ファイル

すべて `mece_output/` ディレクトリに保存されます（`output_dir` で変更できます）。

### 1. cluster_summary.csv

各クラスタの要約情報です。

| 列名 | 説明 |
|-----|------|
| cluster | クラスタID |
| size | 文書数 |
| cohesion | 凝集度（Silhouette係数） |
| keywords | 代表キーワード（TF-IDF） |
| representative_sentence | 代表文（Medoid） |
| llm_summary | LLMによる要約 |

### 2. cluster_summary.md

各クラスタ要約のMarkdown版です。
人が読みやすい構造になっています。MECE指標のサマリと、クラスタごとの詳細（キーワード、代表文、LLM要約）を含みます。

### 3. documents_with_clusters.csv

各文書のクラスタ割り当てです。

| 列名 | 説明 |
|-----|------|
| document | 文書の内容 |
| cluster_kmeans | KMeansクラスタID |
| cluster_hdbscan | HDBSCANクラスタID |
| silhouette | 凝集度（個別） |

### 4. cluster_overlaps.csv

クラスタ間の重複度（MECE違反の候補）です。

| 列名 | 説明 |
|-----|------|
| cluster_A | クラスタAのID |
| cluster_B | クラスタBのID |
| overlap | 重複度（クラスタ中心のコサイン類似度、0〜1） |

### 5. mece_metadata.csv

全体の統計情報です。

### 6. mece_visualization.png

4種類の可視化（PCA/UMAP × KMeans/HDBSCAN）です。

### 7. mece_interactive.html

インタラクティブな散布図（凝集度を点サイズで表現）。

---

## ⚙️ 詳細設定

### MECEConfigのパラメータ

```python
config = MECEConfig(
    # 埋め込みモデル
    embedding_model="cl-nagoya/ruri-v3-310m",
    embedding_prefix=None,  # None ならモデル名から自動判定（ruri-v3: "トピック: "、E5: "query: "）

    # クラスタ数
    n_clusters=5,

    # HDBSCAN最小クラスタサイズ
    hdbscan_min_size=2,

    # UMAP設定
    umap_n_neighbors=15,
    umap_min_dist=0.1,

    # TF-IDF設定
    tfidf_max_features=100,
    tfidf_ngram_range=(1, 2),
    top_keywords=10,

    # 重複判定の閾値
    overlap_threshold=0.3,

    # LLM設定
    use_llm_summary=True,
    llm_backend="ollama",
    llm_model_name="gemma2:2b",
    llm_max_tokens=128,  # 生成トークン上限（速度・出力量を制御）

    # 出力ディレクトリ
    output_dir=Path("mece_output"),

    # ランダムシード
    random_state=42
)
```

---

## 🔧 日本語特化モデルのセットアップ

### ELYZA-JP-8Bの例

```bash
# 1. 作業ディレクトリを作成する
mkdir -p ~/ollama_models
cd ~/ollama_models

# 2. HuggingFaceからggufファイルをダウンロードする
curl -L -o llama-3-elyza-jp-8b.gguf \
  "https://huggingface.co/mmnga/Llama-3-ELYZA-JP-8B-gguf/resolve/main/Llama-3-ELYZA-JP-8B-q4_k_m.gguf"

# 3. Modelfileを作成する
cat > Modelfile << 'EOF'
FROM llama-3-elyza-jp-8b.gguf

TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.9
EOF

# 4. Ollamaのモデルとして登録する
ollama create elyza-jp8b -f Modelfile

# 5. 登録を確認する
ollama list
```

### コード側の設定

```python
config = MECEConfig(
    llm_backend="ollama",
    llm_model_name="elyza-jp8b",  # 作成したモデル名
)
```

---

## 🐛 トラブルシューティング

### Q1. `ModuleNotFoundError: No module named 'ollama'`

```bash
pip install ollama
```

### Q2. `Ollama接続エラー`

Ollamaが起動していることを確認してください。

```bash
# macOS / Linux
ollama serve

# Windows
# タスクトレイのOllamaアイコンを確認する
```

### Q3. `モデル 'xxx' が見つかりません`

モデルを手動でダウンロードしてください。

```bash
ollama pull gemma2:2b
```

### Q4. メモリ不足エラー

- クラスタ数を減らす: `n_clusters=3`
- 軽量なモデルを使う: `llm_model_name="gemma2:2b"`
- LLM要約をオフにする: `use_llm_summary=False`

### Q5. TF-IDFの警告が出る

文書数が少ない場合は正常です。警告は無視して構いません。

### Q6. 日本語が文字化けする

matplotlibのフォント設定を確認してください。

```python
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Hiragino Sans', 'Yu Gothic', 'Meiryo']
```

---

## 🧰 macOS（MPS）環境の注意

Apple Silicon環境では、PyTorchのMPS（Metal Performance Shaders）が利用できます。
次のコマンドで動作を確認できます。

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('MPS built:', hasattr(torch.backends,'mps') and torch.backends.mps.is_built())
print('MPS available:', hasattr(torch.backends,'mps') and torch.backends.mps.is_available())
PY
```

- `available=True` の場合、GPU（MPS）が使われます。`False` の場合は、CPUに自動で切り替わります。
- 本ツールは、MPSとCPUの両方で動作するように実装しています。Tokenizerやモデルの `.to(device)` を適切に処理しています。

### Tokenizersのフォーク警告の抑制

HuggingFace `tokenizers` が、プロセスのフォーク後に並列化の警告を出す場合があります。
次の環境変数で抑制できます。

```bash
export TOKENIZERS_PARALLELISM=false
```

---

## 🧩 VS Code設定のヒント（Problemsの低減）

IDEの型チェックがグローバルのPythonを見ていると、未インストールの依存関係で「Problems」が多数表示されることがあります。
ワークスペースの仮想環境を指定しましょう。

1. VS Codeのワークスペース設定に次を追加します。

   ```json
   {
      "python.defaultInterpreterPath": ".venv/bin/python"
   }
   ```

2. 型チェックとリントを実行します。

   ```bash
   uvx pyright
   uvx ruff check
   ```

---

## 📊 使用例

### ビジネスアイデアの分類

```python
ideas = [
    "顧客満足度向上のための新サービス企画",
    "業務効率化を実現する自動化ツール開発",
    "新市場開拓のためのマーケティング戦略",
    # ...
]

config = MECEConfig(
    n_clusters=4,
    llm_model_name="qwen2.5:7b"
)

analyzer = MECEAnalyzer(config)
analyzer.load_documents(ideas).run()
```

### 研究論文の分類

```python
abstracts = [
    "深層学習を用いた画像認識手法の提案",
    "自然言語処理における転移学習の応用",
    # ...
]

config = MECEConfig(
    n_clusters=6,
    embedding_model="intfloat/multilingual-e5-base"
)

analyzer = MECEAnalyzer(config)
analyzer.load_documents(abstracts).run()
```

### 顧客フィードバックの分類

```python
import pandas as pd

feedbacks = pd.read_csv("customer_feedback.csv")
texts = feedbacks["comment"].tolist()

config = MECEConfig(
    n_clusters=5,
    llm_model_name="gemma2:2b"
)

analyzer = MECEAnalyzer(config)
analyzer.load_documents(texts).run()
```

---

## 🔄 ワークフロー

```
1. 文書リストを準備する
   ↓
2. MECEConfigを設定する
   ↓
3. MECEAnalyzerを実行する
   ↓
4. 出力CSVを確認する
   ↓
5. クラスタに「視点」を付与する
   ↓
6. 重複・粒度を検討する
   ↓
7. 必要に応じて再実行する
   （n_clustersの変更など）
   ↓
8. MECEを完成させる
```

---

## 📚 技術詳細

### 処理フロー

```
文章集合
   ↓
意味ベクトル化（Sentence-BERT）
   ↓
次元削減（PCA / UMAP）
   ↓
クラスタリング（KMeans / HDBSCAN）
   ↓
クラスタ要約（TF-IDF + Medoid + LLM）
   ↓
MECE判断支援指標の計算
   ↓
可視化・保存
```

### 使用技術

- **Sentence-BERT**: 文書の意味ベクトル化
- **PCA**: 線形次元削減
- **UMAP**: 非線形次元削減
- **KMeans**: 分割最適化クラスタリング
- **HDBSCAN**: 密度ベースのクラスタリング
- **TF-IDF**: キーワード抽出
- **SudachiPy**: 日本語の形態素解析（TF-IDFの語の分割）
- **Cosine Similarity**: 類似度の計算
- **Silhouette係数**: クラスタ品質の評価
- **コサイン類似度（クラスタ中心間）**: クラスタ間の重複度の推定

---

## 🤝 貢献

バグ報告や機能提案は大歓迎です。

---

## 📝 ライセンス

MIT License

---

## 🙏 謝辞

このツールは、次のオープンソースプロジェクトを使用しています。

- [Sentence Transformers](https://www.sbert.net/)
- [scikit-learn](https://scikit-learn.org/)
- [UMAP](https://umap-learn.readthedocs.io/)
- [HDBSCAN](https://hdbscan.readthedocs.io/)
- [Ollama](https://ollama.com/)
- [Ruri](https://huggingface.co/cl-nagoya/ruri-v3-310m)
- [SudachiPy](https://github.com/WorksApplications/SudachiPy)

---

## 📧 お問い合わせ

質問やフィードバックは Issue からお願いします。

---

**最終更新**: 2026年10月12日
