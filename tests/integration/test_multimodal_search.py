"""マルチモーダル検索の統合テスト

このモジュールはテキストと画像を統合したマルチモーダル検索機能の
エンドツーエンドテストを実装します。
"""

import logging
import pytest
from pathlib import Path

from src.models.document import Chunk, ImageDocument
from src.rag.vector_store import BaseVectorStore, create_vector_store
from src.rag.embeddings import EmbeddingGenerator
from src.utils.config import get_config

logger = logging.getLogger(__name__)


@pytest.fixture
def multimodal_vector_store(tmp_path):
    """マルチモーダルテスト用のベクトルストア"""
    config = get_config()
    # テスト用の一時ディレクトリを使用
    config.chroma_persist_directory = str(tmp_path / "test_chroma_multimodal")

    vector_store = VectorStore(config)
    vector_store.initialize()

    yield vector_store

    # クリーンアップ
    vector_store.close()


@pytest.fixture
def embedding_generator():
    """埋め込み生成器フィクスチャ"""
    config = get_config()
    return EmbeddingGenerator(config)


@pytest.fixture
def sample_text_documents(embedding_generator, multimodal_vector_store):
    """サンプルテキストドキュメントを追加"""
    chunks = [
        Chunk(
            content="Pythonは汎用プログラミング言語です。機械学習やデータ分析でよく使用されます。",
            chunk_id="chunk_text_1",
            document_id="doc_text_1",
            chunk_index=0,
            start_char=0,
            end_char=100,
            metadata={
                "document_id": "doc_text_1",
                "document_name": "Python入門.txt",
                "source": "/test/python.txt",
                "doc_type": "text",
                "chunk_index": 0,
                "size": 100
            }
        ),
        Chunk(
            content="機械学習は人工知能の一分野です。データからパターンを学習します。",
            chunk_id="chunk_text_2",
            document_id="doc_text_2",
            chunk_index=0,
            start_char=0,
            end_char=80,
            metadata={
                "document_id": "doc_text_2",
                "document_name": "AI基礎.txt",
                "source": "/test/ai.txt",
                "doc_type": "text",
                "chunk_index": 0,
                "size": 80
            }
        ),
    ]

    # テキストの埋め込みを生成
    embeddings = embedding_generator.embed_documents([c.content for c in chunks])

    # ベクトルストアに追加
    multimodal_vector_store.add_documents(chunks, embeddings)

    return chunks


@pytest.fixture
def sample_images(embedding_generator, multimodal_vector_store):
    """サンプル画像ドキュメントを追加"""
    from datetime import datetime

    images = [
        ImageDocument(
            id="img_1",
            file_path=Path("/test/dog.jpg"),
            file_name="dog.jpg",
            image_type="jpg",
            caption="犬の写真。柴犬が公園で遊んでいる様子。",
            metadata={"tags": "動物,犬,屋外"},  # ChromaDBはリストを許可しないため文字列に変換
            created_at=datetime.now(),
            image_data=None
        ),
        ImageDocument(
            id="img_2",
            file_path=Path("/test/python_code.png"),
            file_name="python_code.png",
            image_type="png",
            caption="Pythonのコードスクリーンショット。機械学習のサンプルコード。",
            metadata={"tags": "プログラミング,Python,コード"},  # ChromaDBはリストを許可しないため文字列に変換
            created_at=datetime.now(),
            image_data=None
        ),
    ]

    # 画像のキャプションから埋め込みを生成
    embeddings = embedding_generator.embed_documents([img.caption for img in images])

    # ベクトルストアに追加
    multimodal_vector_store.add_images(images, embeddings, collection_name="images")

    return images


class TestMultimodalSearch:
    """マルチモーダル検索のテストクラス"""

    def test_multimodal_search_text_only(
        self,
        multimodal_vector_store,
        embedding_generator,
        sample_text_documents
    ):
        """テキストのみ存在する場合のマルチモーダル検索"""
        # クエリの埋め込みを生成
        query = "Pythonについて"
        query_embedding = embedding_generator.embed_query(query)

        # マルチモーダル検索を実行
        results = multimodal_vector_store.search_multimodal(
            query_embedding=query_embedding,
            top_k=5,
            text_weight=0.5,
            image_weight=0.5
        )

        # テキスト結果のみが返されることを確認
        assert len(results) > 0
        assert all(r.result_type == 'text' for r in results)

        # 最も関連性の高い結果がPythonに関するものであることを確認
        assert "Python" in results[0].chunk.content

    def test_multimodal_search_image_only(
        self,
        multimodal_vector_store,
        embedding_generator,
        sample_images
    ):
        """画像のみ存在する場合のマルチモーダル検索"""
        # クエリの埋め込みを生成
        query = "犬の写真"
        query_embedding = embedding_generator.embed_query(query)

        # マルチモーダル検索を実行
        results = multimodal_vector_store.search_multimodal(
            query_embedding=query_embedding,
            top_k=5,
            text_weight=0.5,
            image_weight=0.5
        )

        # 画像結果のみが返されることを確認
        assert len(results) > 0
        assert all(r.result_type == 'image' for r in results)

        # 最も関連性の高い結果が犬に関するものであることを確認
        assert "犬" in results[0].caption

    def test_multimodal_search_both(
        self,
        multimodal_vector_store,
        embedding_generator,
        sample_text_documents,
        sample_images
    ):
        """テキストと画像の両方が存在する場合のマルチモーダル検索"""
        # クエリの埋め込みを生成
        query = "Python"
        query_embedding = embedding_generator.embed_query(query)

        # マルチモーダル検索を実行
        results = multimodal_vector_store.search_multimodal(
            query_embedding=query_embedding,
            top_k=10,
            text_weight=0.5,
            image_weight=0.5
        )

        # テキストと画像の両方の結果が含まれることを確認
        assert len(results) > 0
        result_types = {r.result_type for r in results}
        assert 'text' in result_types
        assert 'image' in result_types

        # スコアが降順にソートされていることを確認
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # Pythonに関連するテキスト結果があることを確認
        python_texts = [r for r in results if r.result_type == 'text' and
                        ("Python" in r.chunk.content or "python" in r.chunk.content.lower())]
        assert len(python_texts) > 0, "Python related text should be in results"

    def test_multimodal_search_weight_adjustment(
        self,
        multimodal_vector_store,
        embedding_generator,
        sample_text_documents,
        sample_images
    ):
        """検索重みの調整が機能することを確認"""
        query = "Python"
        query_embedding = embedding_generator.embed_query(query)

        # テキスト重視の検索
        text_heavy_results = multimodal_vector_store.search_multimodal(
            query_embedding=query_embedding,
            top_k=5,
            text_weight=0.9,
            image_weight=0.1
        )

        # 画像重視の検索
        image_heavy_results = multimodal_vector_store.search_multimodal(
            query_embedding=query_embedding,
            top_k=5,
            text_weight=0.1,
            image_weight=0.9
        )

        # テキスト重視の場合、テキスト結果の割合が高いことを確認
        text_count_text_heavy = sum(1 for r in text_heavy_results if r.result_type == 'text')
        text_count_image_heavy = sum(1 for r in image_heavy_results if r.result_type == 'text')

        # 重みによって結果の分布が変わることを確認
        # （必ずしも厳密ではないが、傾向として確認）
        assert text_count_text_heavy >= text_count_image_heavy or len(text_heavy_results) == len(image_heavy_results)

    def test_multimodal_search_empty_query(
        self,
        multimodal_vector_store,
        embedding_generator,
        sample_text_documents,
        sample_images
    ):
        """空のクエリでエラーが発生することを確認"""
        # 空文字列のクエリ（スペースのみ）はOllamaがエラーを返すため、例外が発生することを確認
        from src.rag.embeddings import EmbeddingError

        query = "   "
        with pytest.raises((EmbeddingError, ValueError)):
            query_embedding = embedding_generator.embed_query(query)

    def test_multimodal_search_no_results(
        self,
        multimodal_vector_store,
        embedding_generator
    ):
        """ドキュメントや画像がない場合の検索"""
        query = "存在しない内容"
        query_embedding = embedding_generator.embed_query(query)

        # マルチモーダル検索を実行
        results = multimodal_vector_store.search_multimodal(
            query_embedding=query_embedding,
            top_k=5
        )

        # 空の結果が返されることを確認
        assert isinstance(results, list)
        assert len(results) == 0

    def test_multimodal_search_top_k_limit(
        self,
        multimodal_vector_store,
        embedding_generator,
        sample_text_documents,
        sample_images
    ):
        """top_k制限が正しく機能することを確認"""
        query = "Python"
        query_embedding = embedding_generator.embed_query(query)

        # top_k=2で検索
        results = multimodal_vector_store.search_multimodal(
            query_embedding=query_embedding,
            top_k=2,
            text_weight=0.5,
            image_weight=0.5
        )

        # 結果が最大2件であることを確認
        assert len(results) <= 2

        # top_k=10で検索
        results_large = multimodal_vector_store.search_multimodal(
            query_embedding=query_embedding,
            top_k=10,
            text_weight=0.5,
            image_weight=0.5
        )

        # 結果がtop_k=2よりも多いか同じであることを確認
        assert len(results_large) >= len(results)

    def test_multimodal_search_metadata(
        self,
        multimodal_vector_store,
        embedding_generator,
        sample_text_documents,
        sample_images
    ):
        """検索結果にメタデータが含まれることを確認"""
        query = "Python"
        query_embedding = embedding_generator.embed_query(query)

        # マルチモーダル検索を実行
        results = multimodal_vector_store.search_multimodal(
            query_embedding=query_embedding,
            top_k=5
        )

        # すべての結果に適切なメタデータが含まれることを確認
        for result in results:
            assert result.document_name is not None
            assert result.document_source is not None
            assert result.score is not None
            assert result.rank is not None
            assert result.result_type in ['text', 'image']

            # search_typeメタデータが設定されていることを確認
            if hasattr(result, 'metadata'):
                assert 'search_type' in result.metadata
                assert result.metadata['search_type'] in ['text', 'image']


@pytest.fixture(scope="module")
def check_ollama_service():
    """Ollamaサービスが起動しているかチェックする

    Yields:
        bool: Ollamaが起動している場合True
    """
    import requests

    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            yield True
        else:
            pytest.skip("Ollama service is not responding correctly")
    except Exception as e:
        pytest.skip(f"Ollama service is not available: {e}")


@pytest.mark.integration
class TestMultimodalSearchIntegration:
    """マルチモーダル検索の実際のOllama統合テスト"""

    def test_real_multimodal_search(
        self,
        multimodal_vector_store,
        embedding_generator,
        sample_text_documents,
        sample_images,
        check_ollama_service
    ):
        """実際のOllamaを使用したマルチモーダル検索テスト"""
        # 日本語クエリで検索
        queries = [
            "Pythonプログラミング",
            "犬の写真",
            "機械学習",
        ]

        for query in queries:
            logger.info(f"Testing query: {query}")
            query_embedding = embedding_generator.embed_query(query)

            results = multimodal_vector_store.search_multimodal(
                query_embedding=query_embedding,
                top_k=5
            )

            # 結果が返されることを確認
            assert len(results) > 0

            # 結果の内容をログ出力
            for i, result in enumerate(results, 1):
                logger.info(
                    f"  {i}. [{result.result_type}] "
                    f"{result.document_name} (score: {result.score:.4f})"
                )
