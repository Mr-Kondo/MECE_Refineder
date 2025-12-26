from typing import Any, cast

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Literal
from dataclasses import dataclass, field
import time

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import umap
import hdbscan

import matplotlib.pyplot as plt
import plotly.express as px

from tqdm.auto import tqdm

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Hiragino Sans", "Yu Gothic", "Meirio"]


# =========================================================
# LLMバックエンド抽象化
# =========================================================
class LLMBackend:
    """ローカルLLMのバックエンド抽象化クラス"""

    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    """Ollamaバックエンド"""

    def __init__(self, model_name: str = "qwen2.5:7b", max_tokens: int = 128):
        self.model_name = model_name
        self.max_tokens = max_tokens
        try:
            import ollama

            self.client = ollama

            # 接続テスト
            self.client.list()

            # モデルの存在確認
            models = self.client.list()
            model_names = [m["name"] for m in models.get("models", [])]

            if self.model_name not in model_names:
                print(f"  Warning: モデル '{self.model_name}' が見つかりません")
                print("  モデルをダウンロード中...")
                self.client.pull(self.model_name)
                print(f"  ダウンロード完了: {self.model_name}")

            print(f"  Ollama接続成功: {model_name}")

        except Exception as e:
            raise RuntimeError(
                f"Ollama接続エラー: {e}\n"
                "Ollamaが起動していることを確認してください。\n"
                "インストール: https://ollama.com/download"
            )

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": self.max_tokens, "top_p": 0.9},
            )
            return response["response"].strip()
        except Exception as e:
            return f"[LLM生成エラー: {e}]"


class TransformersBackend(LLMBackend):
    """Transformersバックエンド(GPU推奨)"""

    def __init__(self, model_name: str = "rinna/japanese-gpt-neox-3.6b", max_tokens: int = 128):
        self.model_name = model_name
        self.max_tokens = max_tokens
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            print(f"  モデル読み込み中: {model_name}")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            )
            # Move model to target device (MPS/CPU)
            from torch.nn import Module

            _ = cast(Module, self.model).to(torch.device(self.device))

            print(f"  モデル読み込み完了({self.device})")

        except Exception as e:
            raise RuntimeError(f"Transformersモデル読み込みエラー: {e}")

    def generate(self, prompt: str) -> str:
        try:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(torch.device(self.device)) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # プロンプト部分を除去
            response = response.replace(prompt, "").strip()

            return response

        except Exception as e:
            return f"[LLM生成エラー: {e}]"


# =========================================================
# 設定クラス
# =========================================================
@dataclass
class MECEConfig:
    """MECE分析の設定パラメータ"""

    # 埋め込みモデル
    embedding_model: str = "intfloat/multilingual-e5-base"
    # 英語のみの場合: "all-MiniLM-L6-v2"

    # クラスタ数
    n_clusters: int = 3

    # HDBSCAN最小クラスタサイズ
    hdbscan_min_size: int = 2

    # 次元削減
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    # TF-IDF設定
    tfidf_max_features: int = 100
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    top_keywords: int = 10

    # 重複判定閾値
    overlap_threshold: float = 0.3

    # LLM設定
    use_llm_summary: bool = True
    llm_backend: Literal["ollama", "transformers"] = "ollama"

    # === Ollama用モデル(実在するもの) ===
    # 推奨: gemma2:2b (軽量・高速)
    llm_model_name: str = "gemma2:2b"
    # 生成トークン上限（LLMの速度・出力量制御）
    llm_max_tokens: int = 128

    # その他の選択肢:
    # llm_model_name: str = "qwen2.5:7b"     # 日本語性能◎
    # llm_model_name: str = "llama3.1:8b"    # バランス型
    # llm_model_name: str = "deepseek-r1:8b" # 推論特化
    # llm_model_name: str = "gemma2:9b"      # 高性能

    # === 日本語特化モデル(手動セットアップ必要) ===
    # llm_model_name: str = "elyza-jp8b"     # 要: 事前にModelfileで作成
    # llm_model_name: str = "swallow-8b"     # 要: 事前にModelfileで作成

    # === Transformers用モデル ===
    # llm_model_name: str = "rinna/japanese-gpt-neox-3.6b"
    # llm_model_name: str = "cyberagent/calm2-7b-chat"
    # llm_model_name: str = "line-corporation/japanese-large-lm-3.6b"

    # 出力ディレクトリ
    output_dir: Path = field(default_factory=lambda: Path("mece_output"))

    # ランダムシード
    random_state: int = 42


# =========================================================
# MECEクラスタリング分析クラス
# =========================================================
class MECEAnalyzer:
    """
    MECE分割補助ツール(LLM要約 + tqdm対応)

    設計思想:
    - MECEを自動生成するのではない
    - 人間の判断材料を機械的に提供する
    - ローカルLLMで各クラスタの「視点」を言語化
    """

    def __init__(self, config: MECEConfig):
        self.config = config
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.labels_kmeans: Optional[np.ndarray] = None
        self.labels_hdb: Optional[np.ndarray] = None
        self.df_summary: Optional[pd.DataFrame] = None
        self.llm_backend: Optional[LLMBackend] = None

        # 出力ディレクトリ作成
        self.config.output_dir.mkdir(exist_ok=True)

        # 埋め込みモデル読み込み
        print(f"▶ 埋め込みモデル読み込み: {self.config.embedding_model}")
        self.model = SentenceTransformer(self.config.embedding_model)

        # LLMバックエンド初期化は遅延実行（要約生成時）
        if self.config.use_llm_summary:
            print(f"▶ LLMバックエンド準備: {self.config.llm_backend} (遅延初期化)")

    def _init_llm_backend(self):
        """LLMバックエンドを初期化"""
        try:
            if self.config.llm_backend == "ollama":
                self.llm_backend = OllamaBackend(self.config.llm_model_name, max_tokens=self.config.llm_max_tokens)
            elif self.config.llm_backend == "transformers":
                self.llm_backend = TransformersBackend(self.config.llm_model_name, max_tokens=self.config.llm_max_tokens)
            else:
                raise ValueError(f"未対応のLLMバックエンド: {self.config.llm_backend}")
        except Exception as e:
            print(f"  Warning: LLM初期化失敗 - {e}")
            print("  LLM要約をスキップします")
            self.config.use_llm_summary = False
            self.llm_backend = None

    def load_documents(self, documents: List[str]) -> "MECEAnalyzer":
        """文書リストを読み込む"""
        self.documents = documents
        print(f"▶ 文書読み込み: {len(documents)}件")
        return self

    def embed(self) -> "MECEAnalyzer":
        """文書を意味ベクトル化"""
        print("▶ 文書ベクトル化中...")

        if len(self.documents) == 0:
            raise ValueError("文書が空です。'load_documents'で文書を設定してください。")

        self.embeddings = self.model.encode(self.documents, show_progress_bar=True, normalize_embeddings=True, batch_size=32)

        print(f"  ベクトル形状: {self.embeddings.shape}")
        return self

    def reduce_dimensions(self) -> Tuple[np.ndarray, np.ndarray]:
        """次元削減(PCA + UMAP)"""
        print("▶ 次元削減中...")

        # Safety: embeddings 必須
        assert self.embeddings is not None, "embeddings が未計算です。embed() を先に呼んでください。"

        # PCA
        pca = PCA(n_components=2, random_state=self.config.random_state)
        emb_pca = pca.fit_transform(self.embeddings)
        emb_pca = np.asarray(emb_pca)
        print(f"  PCA説明分散: {pca.explained_variance_ratio_.sum():.3f}")

        # UMAP(tqdm対応)
        umap_reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.config.umap_n_neighbors,
            min_dist=self.config.umap_min_dist,
            metric="cosine",
            random_state=self.config.random_state,
            verbose=False,
        )

        with tqdm(total=1, desc="  UMAP実行") as pbar:
            emb_umap = umap_reducer.fit_transform(self.embeddings)
            emb_umap = np.asarray(emb_umap)
            pbar.update(1)

        self.emb_pca = emb_pca
        self.emb_umap = emb_umap

        return emb_pca, emb_umap

    def cluster(self) -> "MECEAnalyzer":
        """クラスタリング(KMeans + HDBSCAN)"""
        print("▶ クラスタリング実行中...")

        # Safety: embeddings 必須
        assert self.embeddings is not None, "embeddings が未計算です。embed() を先に呼んでください。"

        # クラスタ数の妥当性チェック
        if self.config.n_clusters > len(self.documents):
            print(
                f"  Warning: n_clusters({self.config.n_clusters}) > 文書数({len(self.documents)}), 文書数に合わせて調整します"
            )
            self.config.n_clusters = max(1, len(self.documents))

        # KMeans
        with tqdm(total=1, desc="  KMeans") as pbar:
            kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=self.config.random_state, n_init=10)
            self.labels_kmeans = kmeans.fit_predict(self.embeddings)
            self.kmeans = kmeans
            pbar.update(1)

        # HDBSCAN
        with tqdm(total=1, desc="  HDBSCAN") as pbar:
            # Option A: Use euclidean on normalized embeddings to avoid BallTree cosine error
            hdb = hdbscan.HDBSCAN(min_cluster_size=self.config.hdbscan_min_size, metric="euclidean")
            self.labels_hdb = hdb.fit_predict(self.embeddings)
            pbar.update(1)

        # Silhouette Score
        try:
            sil_kmeans = silhouette_score(self.embeddings, self.labels_kmeans)
            print(f"  Silhouette(KMeans): {sil_kmeans:.3f}")
        except Exception as e:
            print(f"  Warning: Silhouette計算失敗 - {e}")
            sil_kmeans = float("nan")

        # 文書ごとのSilhouette
        with tqdm(total=1, desc="  凝集度計算") as pbar:
            try:
                self.silhouette_samples = silhouette_samples(self.embeddings, self.labels_kmeans)
            except Exception:
                self.silhouette_samples = np.zeros(len(self.documents), dtype=float)
            pbar.update(1)

        return self

    def _generate_llm_summary(
        self, cluster_id: int, keywords: str, representative_sentence: str, cluster_docs: List[str]
    ) -> str:
        """LLMでクラスタ要約を生成"""

        # サンプル文書(最大3件)
        sample_docs = "\n".join([f"- {doc[:100]}" for doc in cluster_docs[:3]])

        prompt = f"""以下のクラスタ情報から、このクラスタが表す「視点・観点」を1-2文で要約してください。

【キーワード】
{keywords}

【代表文】
{representative_sentence}

【サンプル文書】
{sample_docs}

【要約】(1-2文で簡潔に):
"""

        assert self.llm_backend is not None, "LLMバックエンドが初期化されていません"
        return self.llm_backend.generate(prompt)

    def summarize_clusters(self) -> pd.DataFrame:
        """クラスタ要約生成"""
        print("▶ クラスタ要約生成中...")
        # 必要ならLLMバックエンドを初期化（遅延）
        if self.config.use_llm_summary and self.llm_backend is None:
            self._init_llm_backend()

        # TF-IDF計算
        vectorizer = TfidfVectorizer(max_features=self.config.tfidf_max_features, ngram_range=self.config.tfidf_ngram_range)

        try:
            tfidf = vectorizer.fit_transform(self.documents)
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            print("  Warning: TF-IDF失敗、文字n-gramカウントに切替")
            try:
                from sklearn.feature_extraction.text import CountVectorizer

                count_vec = CountVectorizer(
                    analyzer="char", ngram_range=(2, 4), min_df=1, max_features=self.config.tfidf_max_features
                )
                tfidf = count_vec.fit_transform(self.documents)
                feature_names = np.array(count_vec.get_feature_names_out())
            except Exception:
                tfidf = None
                feature_names = None

        cluster_summaries = []

        # クラスタごとに処理(tqdm対応)
        assert self.labels_kmeans is not None, "KMeansラベルが未計算です。cluster() を先に呼んでください。"
        assert self.embeddings is not None, "embeddings が未計算です。embed() を先に呼んでください。"
        cluster_ids = sorted(set(self.labels_kmeans))

        for cluster_id in tqdm(cluster_ids, desc="  各クラスタ処理"):
            idxs = np.where(self.labels_kmeans == cluster_id)[0]

            # 代表キーワード
            if tfidf is not None and feature_names is not None and len(idxs) > 0:
                idxs_list = idxs.tolist()
                mean_tfidf = cast(Any, tfidf)[idxs_list].mean(axis=0).A1
                if mean_tfidf.size > 0:
                    top_idx = mean_tfidf.argsort()[-self.config.top_keywords :][::-1]
                    keywords = ", ".join(feature_names[top_idx])
                else:
                    keywords = "N/A"
            else:
                keywords = "N/A"

            # 代表文(Medoid)
            vecs = self.embeddings[idxs]
            centroid = vecs.mean(axis=0) if len(idxs) > 0 else None
            if centroid is not None and len(idxs) > 0:
                sims = cosine_similarity(vecs, centroid.reshape(1, -1)).flatten()
                medoid_idx = idxs[np.argmax(sims)]
                representative_sentence = self.documents[medoid_idx]
            else:
                representative_sentence = self.documents[idxs[0]] if len(idxs) > 0 else "N/A"

            # クラスタ内凝集度
            avg_silhouette = float(np.take(self.silhouette_samples, idxs).mean()) if len(idxs) > 0 else float("nan")

            # LLM要約生成
            llm_summary = ""
            if self.config.use_llm_summary and self.llm_backend:
                cluster_docs = [self.documents[i] for i in idxs]
                llm_summary = self._generate_llm_summary(cluster_id, keywords, representative_sentence, cluster_docs)

            cluster_summaries.append(
                {
                    "cluster": cluster_id,
                    "size": len(idxs),
                    "cohesion": avg_silhouette,
                    "keywords": keywords,
                    "representative_sentence": representative_sentence,
                    "llm_summary": llm_summary if llm_summary else "N/A",
                }
            )

        self.df_summary = pd.DataFrame(cluster_summaries)
        return self.df_summary

    def calculate_mece_metrics(self) -> Dict[str, Any]:
        """MECE判断支援指標を計算"""
        print("▶ MECE指標計算中...")

        overlaps = []

        # クラスタ間重複度計算(tqdm対応)
        assert self.labels_kmeans is not None, "KMeansラベルが未計算です。cluster() を先に呼んでください。"
        assert self.embeddings is not None, "embeddings が未計算です。embed() を先に呼んでください。"
        cluster_pairs = [(i, j) for i in range(self.config.n_clusters) for j in range(i + 1, self.config.n_clusters)]

        for i, j in tqdm(cluster_pairs, desc="  重複度計算"):
            idxs_i = np.where(self.labels_kmeans == i)[0]
            idxs_j = np.where(self.labels_kmeans == j)[0]

            if len(idxs_i) == 0 or len(idxs_j) == 0:
                continue

            centroid_i = self.embeddings[idxs_i].mean(axis=0)
            centroid_j = self.embeddings[idxs_j].mean(axis=0)
            sim = float(cosine_similarity(centroid_i.reshape(1, -1), centroid_j.reshape(1, -1)).item())

            if sim >= self.config.overlap_threshold:
                overlaps.append({"cluster_A": i, "cluster_B": j, "overlap": sim})

        df_overlaps = pd.DataFrame(overlaps)

        # カバレッジ分析
        total_documents = len(self.documents)
        noise_count = int(np.sum(self.labels_hdb == -1)) if total_documents > 0 else 0
        coverage = 1 - (noise_count / total_documents) if total_documents > 0 else 0.0

        metrics = {
            "total_documents": total_documents,
            "n_clusters": self.config.n_clusters,
            "coverage": coverage,
            "noise_documents": noise_count,
            "overlaps": df_overlaps,
        }

        print(f"  カバレッジ: {coverage:.3f}")
        print(f"  ノイズ文書: {noise_count}件")

        return metrics

    def visualize(self) -> "MECEAnalyzer":
        """可視化"""
        print("▶ 可視化生成中...")

        # Safety: 可視化に必要な前提データ
        assert hasattr(self, "emb_pca"), "PCA結果がありません。reduce_dimensions() を先に呼んでください。"
        assert hasattr(self, "emb_umap"), "UMAP結果がありません。reduce_dimensions() を先に呼んでください。"
        assert self.labels_kmeans is not None, "KMeansラベルが未計算です。cluster() を先に呼んでください。"
        assert self.labels_hdb is not None, "HDBSCANラベルが未計算です。cluster() を先に呼んでください。"
        assert isinstance(self.silhouette_samples, np.ndarray), (
            "Silhouetteサンプルが未計算です。cluster() を先に呼んでください。"
        )

        with tqdm(total=2, desc="  グラフ作成") as pbar:
            # --- Matplotlib静的可視化 ---
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))

            # PCA + KMeans
            ax = axes[0, 0]
            ax.scatter(self.emb_pca[:, 0], self.emb_pca[:, 1], c=self.labels_kmeans, cmap="tab10", s=100, alpha=0.6)
            ax.set_title("PCA + KMeans", fontsize=14)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")

            # UMAP + KMeans
            ax = axes[0, 1]
            ax.scatter(self.emb_umap[:, 0], self.emb_umap[:, 1], c=self.labels_kmeans, cmap="tab10", s=100, alpha=0.6)
            ax.set_title("UMAP + KMeans", fontsize=14)
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")

            # UMAP + HDBSCAN
            ax = axes[1, 0]
            ax.scatter(self.emb_umap[:, 0], self.emb_umap[:, 1], c=self.labels_hdb, cmap="tab10", s=100, alpha=0.6)
            ax.set_title("UMAP + HDBSCAN", fontsize=14)
            ax.set_xlabel("UMAP1")
            ax.set_ylabel("UMAP2")

            # Silhouette分布
            ax = axes[1, 1]
            for cluster_id in sorted(set(self.labels_kmeans)):
                idxs = np.where(self.labels_kmeans == cluster_id)[0]
                sil_values = self.silhouette_samples[idxs]
                ax.hist(sil_values, bins=20, alpha=0.6, label=f"Cluster {cluster_id}")
            ax.set_title("Silhouette分布(凝集度)", fontsize=14)
            ax.set_xlabel("Silhouette Coefficient")
            ax.set_ylabel("Frequency")
            ax.legend()
            ax.axvline(0, color="red", linestyle="--", linewidth=1)

            plt.tight_layout()
            plt.savefig(self.config.output_dir / "mece_visualization.png", dpi=150, bbox_inches="tight")
            plt.close()

            pbar.update(1)

            # --- Plotlyインタラクティブ可視化 ---
            # Silhouetteを正規化してPlotlyのサイズに使用（常に正）
            sil = self.silhouette_samples
            sil_min = float(np.min(sil)) if sil.size > 0 else 0.0
            sil_max = float(np.max(sil)) if sil.size > 0 else 1.0
            eps = 1e-6
            size_norm = ((sil - sil_min) / (sil_max - sil_min + eps)) * 20.0 + 5.0

            df_plot = pd.DataFrame(
                {
                    "x": self.emb_umap[:, 0],
                    "y": self.emb_umap[:, 1],
                    "cluster": [str(c) for c in self.labels_kmeans],
                    "size": size_norm,
                    "silhouette": self.silhouette_samples,
                    "text": [d[:100] for d in self.documents],
                }
            )

            fig = px.scatter(
                df_plot,
                x="x",
                y="y",
                color="cluster",
                size="size",
                hover_data=["text", "silhouette"],
                title="UMAP Projection (サイズ=凝集度)",
            )

            fig.write_html(self.config.output_dir / "mece_interactive.html")

            pbar.update(1)

        return self

    def save_results(self, metrics: Dict) -> "MECEAnalyzer":
        """結果を保存"""
        print("▶ 結果保存中...")

        with tqdm(total=4, desc="  ファイル書き出し") as pbar:
            # 文書×クラスタ対応表
            df_result = pd.DataFrame(
                {
                    "document": self.documents,
                    "cluster_kmeans": self.labels_kmeans,
                    "cluster_hdbscan": self.labels_hdb,
                    "silhouette": self.silhouette_samples,
                }
            )
            df_result.to_csv(self.config.output_dir / "documents_with_clusters.csv", index=False, encoding="utf-8-sig")
            pbar.update(1)

            # クラスタ要約
            assert self.df_summary is not None, "クラスタ要約が未生成です。summarize_clusters() を先に呼んでください。"
            self.df_summary.to_csv(self.config.output_dir / "cluster_summary.csv", index=False, encoding="utf-8-sig")
            pbar.update(1)

            # MECE指標
            if not metrics["overlaps"].empty:
                metrics["overlaps"].to_csv(self.config.output_dir / "cluster_overlaps.csv", index=False, encoding="utf-8-sig")
            pbar.update(1)

            # メタ情報
            meta_df = pd.DataFrame(
                [
                    {
                        "total_documents": metrics["total_documents"],
                        "n_clusters": metrics["n_clusters"],
                        "coverage": metrics["coverage"],
                        "noise_documents": metrics["noise_documents"],
                        "llm_backend": self.config.llm_backend if self.config.use_llm_summary else "N/A",
                        "llm_model": self.config.llm_model_name if self.config.use_llm_summary else "N/A",
                    }
                ]
            )
            meta_df.to_csv(self.config.output_dir / "mece_metadata.csv", index=False, encoding="utf-8-sig")
            pbar.update(1)

        print(f"  保存先: {self.config.output_dir}")

        return self

    def run(self) -> "MECEAnalyzer":
        """全処理を実行"""
        start_time = time.time()

        self.embed()
        self.reduce_dimensions()
        self.cluster()
        self.summarize_clusters()
        metrics = self.calculate_mece_metrics()
        self.visualize()
        self.save_results(metrics)

        elapsed = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"処理完了 (経過時間: {elapsed:.1f}秒)")
        print(f"{'=' * 60}")

        print("\n=== クラスタ要約 ===")
        assert self.df_summary is not None, "クラスタ要約が未生成です。summarize_clusters() を先に呼んでください。"
        for _, row in self.df_summary.iterrows():
            print(f"\n【クラスタ {row['cluster']}】({row['size']}件, 凝集度={row['cohesion']})")
            print(f"キーワード: {row['keywords'][:80]}...")
            if row["llm_summary"] != "N/A":
                print(f"LLM要約: {row['llm_summary']}")
            print(f"代表文: {row['representative_sentence'][:80]}...")

        print("\n=== MECE指標 ===")
        print(f"カバレッジ: {metrics['coverage']}")
        print(f"ノイズ文書: {metrics['noise_documents']}件")

        if not metrics["overlaps"].empty:
            print("\nクラスタ間重複:")
            print(metrics["overlaps"].to_string(index=False))
        else:
            print("\nクラスタ間重複: なし")

        return self


# =========================================================
# 実行例
# =========================================================
if __name__ == "__main__":
    # サンプルデータ(ここを差し替える)
    documents = [
        "機械学習の基礎についての解説記事です。",
        "深層学習とニューラルネットワークの比較",
        "自然言語処理の歴史と応用",
        "画像分類のアルゴリズム解説",
        "ビジネスにおけるAIの活用事例",
        "統計解析と確率モデル",
        "Pythonでのデータ分析入門",
        "クラスタリング手法の比較と実装",
        "文章ベクトルの生成方法",
        "PCAと次元削減の理論",
        "転移学習とファインチューニング",
        "GANによる画像生成技術",
        "強化学習の基本概念",
        "時系列データ分析の手法",
        "異常検知アルゴリズムの比較",
        "推薦システムの構築方法",
        "グラフニューラルネットワーク入門",
        "因果推論とその応用",
        "メタ学習とfew-shot学習",
        "説明可能AIの重要性",
    ]

    # 設定
    config = MECEConfig(
        n_clusters=4,
        embedding_model="intfloat/multilingual-e5-base",
        use_llm_summary=True,
        llm_backend="ollama",
        llm_model_name="qwen2.5:7b",  # 実在するモデル
        output_dir=Path("mece_output"),
    )

    # 実行
    analyzer = MECEAnalyzer(config)
    analyzer.load_documents(documents).run()
