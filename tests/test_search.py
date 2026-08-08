"""Phase 1 のスモークテスト。stdlib のみで動作する。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchengine import chunker, hybrid, query, tokenizer  # noqa: E402
from searchengine.embedder import Embedder  # noqa: E402
from searchengine.index import Index  # noqa: E402


def test_tokenize_nonempty():
    toks = tokenizer.tokenize("機械学習のモデル")
    assert toks, "トークンが空"
    assert all(isinstance(t, str) for t in toks)


def test_chunk_overlap_guard():
    chunks = chunker.chunk_text("a b c d e f", size=3, overlap=1)
    assert chunks
    assert all(c.text for c in chunks)


def test_query_builds_fts():
    q = query.to_fts_query('"機械学習" AND モデル')
    assert q and q != '""'


def test_index_and_search(tmp_path=None):
    db = ":memory:"
    idx = Index(db)
    idx.add_document("/x/ml.md", "機械学習はデータからモデルを学習する分野である。")
    idx.add_document("/x/cook.md", "今日の夕食はカレーライスを作りました。")
    hits = idx.search(query.to_fts_query("機械学習 モデル"))
    assert hits, "検索結果が空"
    assert hits[0].path.endswith("ml.md")
    idx.close()


def test_incremental_skip():
    idx = Index(":memory:")
    idx.add_document("/x/a.md", "同じ内容")
    before = idx.stats()["chunks"]
    idx.add_document("/x/a.md", "同じ内容")  # 変更なし → スキップ
    after = idx.stats()["chunks"]
    assert before == after
    idx.close()


# --- Phase 2 ---


def test_embed_stable_across_instances():
    # 別インスタンス（=別プロセス相当）でも同一ベクトルになること
    a = Embedder().encode(["機械学習のモデル"])[0]
    b = Embedder().encode(["機械学習のモデル"])[0]
    assert (a == b).all()


def test_vector_search():
    idx = Index(":memory:", embedder=Embedder())
    idx.add_document("/x/ml.md", "機械学習はデータからモデルを学習する。")
    idx.add_document("/x/cook.md", "カレーライスの作り方を説明する。")
    hits = idx.vector_search("機械学習 モデル")
    assert hits and hits[0].path.endswith("ml.md")
    idx.close()


def test_field_filter():
    p = query.parse("機械学習 type:code")
    assert p.filters.get("type") == "code"
    assert "機械学習" in p.raw and "type" not in p.raw


def test_hybrid_rrf():
    idx = Index(":memory:", embedder=Embedder())
    idx.add_document("/x/ml.md", "機械学習はデータからモデルを学習する分野である。")
    idx.add_document("/x/cook.md", "今日はカレーを作った。")
    fused = hybrid.search(idx, query.parse("機械学習 モデル"))
    assert fused and fused[0].hit.path.endswith("ml.md")
    assert fused[0].rrf > 0
    idx.close()


# --- Phase 3 / #69: 疑問文語尾 FTS 除去 ---


def test_question_suffix_stripped_from_fts():
    # "とは何ですか" を除去してキーワードのみ FTS に渡す
    p = query.parse("BM25とは何ですか")
    assert '"bm25"' in p.fts.lower() or "bm25" in p.fts.lower()
    # 疑問文語尾が FTS クエリに混入しないこと
    assert "とは" not in p.fts
    assert "何ですか" not in p.fts


def test_question_raw_preserved():
    # raw はベクトル検索用なので語尾を保持する
    p = query.parse("BM25とは何ですか")
    assert "bm25" in p.raw.lower()
    assert "とは" in p.raw or "何" in p.raw


def test_non_question_query_unchanged():
    # 疑問文でないクエリはそのまま処理される
    p1 = query.parse("機械学習 モデル")
    p2 = query.parse("FastAPI")
    assert "機械学習" in p1.fts or "機" in p1.fts  # トークン化されてもヒット
    assert "fastapi" in p2.fts.lower()


def test_various_question_suffixes():
    # 疑問文語尾は FTS から除去され、ノイズトークンが含まれないこと
    cases = [
        ("FastAPIについて教えてください", "fastapi"),
        ("機械学習とは何か", "機械"),  # bigram tokenizer: "機械学習" → "機械"/"械学"/"学習"
        ("Rustでしょうか", "rust"),
    ]
    for q_str, expected_keyword in cases:
        p = query.parse(q_str)
        assert expected_keyword.lower() in p.fts.lower(), f"failed for: {q_str!r}, fts={p.fts!r}"
        assert "ください" not in p.fts
        assert "でしょうか" not in p.fts


if __name__ == "__main__":
    test_tokenize_nonempty()
    test_chunk_overlap_guard()
    test_query_builds_fts()
    test_index_and_search()
    test_incremental_skip()
    test_embed_stable_across_instances()
    test_vector_search()
    test_field_filter()
    test_hybrid_rrf()
    test_question_suffix_stripped_from_fts()
    test_question_raw_preserved()
    test_non_question_query_unchanged()
    test_various_question_suffixes()
    print("all tests passed")
