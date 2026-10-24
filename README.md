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

- [操作方法ガイド](docs/operation-guide.md)では、環境の準備から実行、結果の確認、設定の調整、トラブル対応までを説明しています。

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

- Python 3.13または3.14（各系列の最新パッチ版）
- 8GB以上のRAM（推奨: 16GB以上）
- Ollama（ローカルLLM要約を使う場合）

環境の準備と依存関係のインストールは、[操作方法ガイド](docs/operation-guide.md#1-準備する)を参照してください。

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

実行方法、入力データの準備、出力ファイル、設定項目は[操作方法ガイド](docs/operation-guide.md)を参照してください。

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

VS CodeのPython拡張機能がグローバル環境を使っていると、このプロジェクトの依存関係を認識できず、Problemsに診断が表示されることがあります。プロジェクトの仮想環境を使うよう設定します。

`uv` が未導入の場合は、[操作ガイドの環境構築](docs/operation-guide.md)にある公式インストール手順から導入します。

1. 初めて使う場合はリポジトリを取得します。すでにクローン済みの場合は、リポジトリのルートで `git pull` を実行します。

   ```bash
   git clone https://github.com/Mr-Kondo/MECE_Refineder.git
   cd MECE_Refineder
   ```

   初回・更新時のどちらも、プロジェクトのルートで依存関係を同期します。

   ```bash
   uv sync
   ```

2. VS Codeで `Python: Select Interpreter` を実行し、このプロジェクトの `.venv` を選びます。新しい環境での既定値も設定する場合は、ワークスペース設定（`.vscode/settings.json`）に次を追加します。

   ```json
   {
     "python.defaultInterpreterPath": "${workspaceFolder}/.venv"
   }
   ```

   `python.defaultInterpreterPath` は初期設定に使われる値です。すでに別のインタープリターを選択済みの場合は、`Python: Select Interpreter` で `.venv` に切り替えてください。

3. PyrightとRuffによる型チェックとリントをターミナルで実行する場合は、プロジェクトのルートで次を実行します。どちらも開発依存関係としてプロジェクト環境に同期されています。

   ```bash
   uv run pyright
   uv run ruff check
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

**最終更新**: 2026年10月24日
