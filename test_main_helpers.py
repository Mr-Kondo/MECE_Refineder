"""main.py の補助関数の確認。`uv run python test_main_helpers.py` または pytest で実行する。"""

from main import _embedding_prefix, _make_tokenizer


def test_embedding_prefix():
    assert _embedding_prefix("cl-nagoya/ruri-v3-310m", None) == "トピック: "
    assert _embedding_prefix("intfloat/multilingual-e5-base", None) == "query: "
    assert _embedding_prefix("all-MiniLM-L6-v2", None) == ""
    assert _embedding_prefix("cl-nagoya/ruri-v3-310m", "") == ""  # 明示指定が優先


def test_tokenizer_keeps_nouns_only():
    words = _make_tokenizer()("転移学習とファインチューニング")
    assert "学習" in words and "チューニング" in words
    assert "と" not in words


if __name__ == "__main__":
    test_embedding_prefix()
    test_tokenizer_keeps_nouns_only()
    print("ok")
