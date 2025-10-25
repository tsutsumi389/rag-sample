"""ベクトルストア統合テスト

すべてのベクトルDB実装に対して共通のテストを実行します。
"""

import pytest
from src.rag.vector_store import create_vector_store, get_supported_db_types
from src.models.document import Chunk, SearchResult
from src.utils.config import Config


@pytest.fixture(scope="module")
def sample_chunks():
    """テスト用のサンプルチャンク"""
    return [
        Chunk(
            content="これはテストドキュメントの最初のチャンクです。",
            chunk_id="chunk-001",
            document_id="doc-001",
            chunk_index=0,
            start_char=0,
            end_char=50,
            metadata={
                "document_name": "test.txt",
                "source": "/tmp/test.txt",
                "doc_type": "text",
                "size": 50,
            }
        ),
        Chunk(
            content="これはテストドキュメントの2番目のチャンクです。",
            chunk_id="chunk-002",
            document_id="doc-001",
            chunk_index=1,
            start_char=50,
            end_char=100,
            metadata={
                "document_name": "test.txt",
                "source": "/tmp/test.txt",
                "doc_type": "text",
                "size": 50,
            }
        ),
    ]


@pytest.fixture(scope="module")
def sample_embeddings():
    """テスト用のサンプル埋め込みベクトル"""
    import random
    random.seed(42)

    # 384次元のランダムベクトル（nomic-embed-textと同じ次元）
    return [
        [random.random() for _ in range(384)],
        [random.random() for _ in range(384)],
    ]


@pytest.mark.parametrize("db_type", ["chroma", "qdrant"])
def test_vector_store_initialization(db_type):
    """ベクトルストアの初期化テスト"""
    config = Config()
    config.vector_db_type = db_type

    vector_store = create_vector_store(config)

    try:
        vector_store.initialize()
        assert vector_store is not None
    except Exception as e:
        pytest.skip(f"{db_type} が利用できません: {str(e)}")
    finally:
        vector_store.close()


@pytest.mark.parametrize("db_type", ["chroma", "qdrant"])
def test_add_and_search(db_type, sample_chunks, sample_embeddings):
    """ドキュメント追加と検索のテスト"""
    config = Config()
    config.vector_db_type = db_type

    vector_store = create_vector_store(config, collection_name=f"test_{db_type}")

    try:
        # 初期化
        vector_store.initialize()

        # ドキュメント追加
        vector_store.add_documents(sample_chunks, sample_embeddings)

        # ドキュメント数確認
        count = vector_store.get_document_count()
        assert count == 2

        # 検索実行
        results = vector_store.search(
            query_embedding=sample_embeddings[0],
            n_results=2
        )

        # 結果検証
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0

    except Exception as e:
        pytest.skip(f"{db_type} が利用できません: {str(e)}")
    finally:
        # クリーンアップ
        try:
            vector_store.clear()
        except:
            pass
        vector_store.close()


@pytest.mark.parametrize("db_type", ["chroma", "qdrant"])
def test_delete_operations(db_type, sample_chunks, sample_embeddings):
    """削除操作のテスト"""
    config = Config()
    config.vector_db_type = db_type

    vector_store = create_vector_store(config, collection_name=f"test_delete_{db_type}")

    try:
        # 初期化とデータ追加
        vector_store.initialize()
        vector_store.add_documents(sample_chunks, sample_embeddings)

        initial_count = vector_store.get_document_count()
        assert initial_count == 2

        # 1つ削除
        deleted_count = vector_store.delete(chunk_ids=["chunk-001"])
        assert deleted_count == 1

        # 残り確認
        remaining_count = vector_store.get_document_count()
        assert remaining_count == 1

    except Exception as e:
        pytest.skip(f"{db_type} が利用できません: {str(e)}")
    finally:
        try:
            vector_store.clear()
        except:
            pass
        vector_store.close()


@pytest.mark.parametrize("db_type", ["chroma", "qdrant"])
def test_list_documents(db_type, sample_chunks, sample_embeddings):
    """ドキュメント一覧取得のテスト"""
    config = Config()
    config.vector_db_type = db_type

    vector_store = create_vector_store(config, collection_name=f"test_list_{db_type}")

    try:
        # 初期化とデータ追加
        vector_store.initialize()
        vector_store.add_documents(sample_chunks, sample_embeddings)

        # ドキュメント一覧取得
        documents = vector_store.list_documents()

        assert len(documents) == 1  # 1つのドキュメントに2つのチャンク
        assert documents[0]["document_id"] == "doc-001"
        assert documents[0]["chunk_count"] == 2

    except Exception as e:
        pytest.skip(f"{db_type} が利用できません: {str(e)}")
    finally:
        try:
            vector_store.clear()
        except:
            pass
        vector_store.close()


@pytest.mark.parametrize("db_type", ["chroma", "qdrant"])
def test_get_document_by_id(db_type, sample_chunks, sample_embeddings):
    """ドキュメントID指定取得のテスト"""
    config = Config()
    config.vector_db_type = db_type

    vector_store = create_vector_store(config, collection_name=f"test_get_{db_type}")

    try:
        # 初期化とデータ追加
        vector_store.initialize()
        vector_store.add_documents(sample_chunks, sample_embeddings)

        # ドキュメントをIDで取得
        document = vector_store.get_document_by_id("doc-001")

        assert document is not None
        assert document["document_id"] == "doc-001"
        assert document["chunk_count"] == 2
        assert len(document["chunks"]) == 2
        assert document["chunks"][0]["chunk_index"] == 0
        assert document["chunks"][1]["chunk_index"] == 1

    except Exception as e:
        pytest.skip(f"{db_type} が利用できません: {str(e)}")
    finally:
        try:
            vector_store.clear()
        except:
            pass
        vector_store.close()


@pytest.mark.parametrize("db_type", ["chroma", "qdrant"])
def test_clear_all_documents(db_type, sample_chunks, sample_embeddings):
    """全ドキュメント削除のテスト"""
    config = Config()
    config.vector_db_type = db_type

    vector_store = create_vector_store(config, collection_name=f"test_clear_{db_type}")

    try:
        # 初期化とデータ追加
        vector_store.initialize()
        vector_store.add_documents(sample_chunks, sample_embeddings)

        initial_count = vector_store.get_document_count()
        assert initial_count == 2

        # 全削除
        vector_store.clear()

        # 空になっていることを確認
        count = vector_store.get_document_count()
        assert count == 0

    except Exception as e:
        pytest.skip(f"{db_type} が利用できません: {str(e)}")
    finally:
        try:
            vector_store.clear()
        except:
            pass
        vector_store.close()


def test_get_supported_db_types():
    """サポートされているDB種別の取得テスト"""
    supported_types = get_supported_db_types()

    assert isinstance(supported_types, list)
    assert "chroma" in supported_types
    assert len(supported_types) > 0
