"""マルチモーダルRAGの統合テスト

実際のOllama、ChromaDB、マルチモーダルLLMを使用してエンドツーエンドのテストを実行します。

前提条件:
- Ollamaがローカルで起動していること
- 以下のモデルがインストールされていること:
  - ollama pull gemma3 (マルチモーダルLLM)
  - ollama pull llava (ビジョンモデル)
  - ollama pull nomic-embed-text (テキスト埋め込み)
"""

import pytest
import shutil
from pathlib import Path

from src.rag.multimodal_engine import MultimodalRAGEngine
from src.rag.vector_store import VectorStore
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vision_embeddings import VisionEmbeddings
from src.rag.image_processor import ImageProcessor
from src.rag.document_processor import DocumentProcessor
from src.utils.config import get_config


@pytest.fixture(scope="module")
def test_chroma_dir(tmp_path_factory):
    """テスト用のChromaDB保存ディレクトリ"""
    chroma_dir = tmp_path_factory.mktemp("test_chroma_multimodal")
    yield chroma_dir
    # テスト終了後にクリーンアップ
    if chroma_dir.exists():
        shutil.rmtree(chroma_dir)


@pytest.fixture(scope="module")
def config(test_chroma_dir):
    """テスト用の設定"""
    config = get_config()
    # テスト用のChromaDBディレクトリを設定
    config.chroma_persist_directory = str(test_chroma_dir)
    return config


@pytest.fixture(scope="module")
def vector_store(config):
    """VectorStoreのインスタンスを作成"""
    store = VectorStore(config)
    store.initialize()
    yield store
    store.close()


@pytest.fixture(scope="module")
def text_embeddings(config):
    """テキスト埋め込み生成器のインスタンスを作成"""
    return EmbeddingGenerator(config)


@pytest.fixture(scope="module")
def vision_embeddings(config):
    """ビジョン埋め込み生成器のインスタンスを作成"""
    return VisionEmbeddings(config)


@pytest.fixture(scope="module")
def multimodal_engine(config, vector_store, text_embeddings, vision_embeddings):
    """MultimodalRAGEngineのインスタンスを作成"""
    engine = MultimodalRAGEngine(
        config=config,
        vector_store=vector_store,
        text_embeddings=text_embeddings,
        vision_embeddings=vision_embeddings
    )
    return engine


@pytest.fixture(scope="module")
def sample_images(tmp_path_factory):
    """テスト用のサンプル画像を作成"""
    images_dir = tmp_path_factory.mktemp("test_images")

    # シンプルなPNG画像を作成（1x1ピクセルの赤い画像）
    from PIL import Image

    # 赤い画像
    red_image = Image.new('RGB', (100, 100), color='red')
    red_path = images_dir / "red.png"
    red_image.save(red_path)

    # 青い画像
    blue_image = Image.new('RGB', (100, 100), color='blue')
    blue_path = images_dir / "blue.png"
    blue_image.save(blue_path)

    yield [red_path, blue_path]

    # クリーンアップ
    if images_dir.exists():
        shutil.rmtree(images_dir)


@pytest.fixture(scope="module")
def sample_documents(tmp_path_factory):
    """テスト用のサンプルドキュメントを作成"""
    docs_dir = tmp_path_factory.mktemp("test_docs")

    # テキストドキュメント1
    doc1_path = docs_dir / "colors.txt"
    doc1_path.write_text(
        "赤色は暖色の一つで、情熱や愛を象徴します。\n"
        "赤は視覚的に強い印象を与える色です。"
    )

    # テキストドキュメント2
    doc2_path = docs_dir / "blue.txt"
    doc2_path.write_text(
        "青色は寒色で、冷静さや信頼を表します。\n"
        "青は空や海を連想させる色です。"
    )

    yield [doc1_path, doc2_path]

    # クリーンアップ
    if docs_dir.exists():
        shutil.rmtree(docs_dir)


@pytest.mark.integration
@pytest.mark.multimodal
class TestMultimodalRAGIntegration:
    """マルチモーダルRAG統合テスト"""

    def test_setup_documents_and_images(
        self,
        vector_store,
        text_embeddings,
        vision_embeddings,
        sample_documents,
        sample_images
    ):
        """テキストドキュメントと画像をベクトルストアに追加"""
        # テキストドキュメントを追加
        doc_processor = DocumentProcessor(text_embeddings)

        for doc_path in sample_documents:
            document = doc_processor.load_document(doc_path)
            chunks = doc_processor.split_document(document)
            embeddings = text_embeddings.embed_documents(
                [chunk.content for chunk in chunks]
            )
            vector_store.add_documents(chunks, embeddings)

        # ドキュメント数を確認
        doc_count = vector_store.get_document_count()
        assert doc_count > 0, "テキストドキュメントが追加されていません"

        # 画像を追加
        image_processor = ImageProcessor(vision_embeddings)

        images = []
        for img_path in sample_images:
            image_doc = image_processor.load_image(
                img_path,
                auto_caption=True  # キャプション自動生成
            )
            images.append(image_doc)

        # 画像の埋め込みを生成
        image_embeddings = vision_embeddings.embed_images(
            [img.file_path for img in images]
        )

        # 画像をベクトルストアに追加
        image_ids = vector_store.add_images(images, image_embeddings)

        assert len(image_ids) == len(sample_images), "画像が正しく追加されていません"

    def test_search_images_with_text_query(self, multimodal_engine):
        """テキストクエリで画像を検索"""
        results = multimodal_engine.search_images(
            query="red color",
            top_k=2
        )

        # 検索結果があることを確認
        assert len(results) > 0, "画像検索結果が空です"

        # 検索結果の構造を確認
        for result in results:
            assert result.result_type == 'image'
            assert result.image_path is not None
            assert result.caption is not None
            assert 0 <= result.score <= 1

    def test_search_multimodal(self, multimodal_engine):
        """マルチモーダル検索（テキストと画像の両方）"""
        results = multimodal_engine.search_multimodal(
            query="blue",
            top_k=5
        )

        # テキストと画像の両方の結果が含まれることを確認
        assert len(results) > 0, "マルチモーダル検索結果が空です"

        # 結果タイプの確認
        result_types = {r.result_type for r in results}
        # 少なくともテキストまたは画像のいずれかが含まれている
        assert result_types.issubset({'text', 'image'})

    def test_query_with_images(self, multimodal_engine, sample_images):
        """画像付き質問応答"""
        # 画像を添付して質問
        result = multimodal_engine.query_with_images(
            query="この画像は何色ですか？",
            image_paths=[str(sample_images[0])],  # 赤い画像
            n_results=3
        )

        # 回答の確認
        assert 'answer' in result
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0

        # メタデータの確認
        assert 'context_count' in result
        assert 'images_used' in result
        assert result['images_used'] == 1

        # 情報源の確認
        if 'sources' in result:
            assert isinstance(result['sources'], list)

    def test_query_without_images(self, multimodal_engine):
        """画像なしの質問応答"""
        result = multimodal_engine.query_with_images(
            query="赤色について教えてください",
            n_results=3
        )

        # 回答の確認
        assert 'answer' in result
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0

        # 画像が使用されていないことを確認
        assert result['images_used'] == 0

    def test_chat_multimodal(self, multimodal_engine):
        """マルチモーダルチャット（会話履歴あり）"""
        # チャット履歴をクリア
        multimodal_engine.clear_chat_history()

        # 1回目のメッセージ
        result1 = multimodal_engine.chat_multimodal(
            message="青色の特徴は何ですか？",
            n_results=2
        )

        assert 'answer' in result1
        assert result1['history_length'] == 2  # user + assistant

        # 2回目のメッセージ（履歴を考慮）
        result2 = multimodal_engine.chat_multimodal(
            message="それは何色ですか？",  # 前の文脈を参照
            n_results=2
        )

        assert 'answer' in result2
        assert result2['history_length'] == 4  # 2往復分

        # 履歴の確認
        history = multimodal_engine.get_chat_history()
        assert len(history) == 4
        assert history[0]['role'] == 'user'
        assert history[1]['role'] == 'assistant'

    def test_get_status(self, multimodal_engine):
        """ステータス情報の取得"""
        status = multimodal_engine.get_status()

        assert 'multimodal_llm_model' in status
        assert 'text_embedding_model' in status
        assert 'vision_embedding_model' in status
        assert 'vector_store_info' in status
        assert 'chat_history_length' in status


@pytest.mark.integration
@pytest.mark.multimodal
class TestMultimodalRAGErrorHandling:
    """マルチモーダルRAG エラーハンドリングのテスト"""

    def test_query_with_nonexistent_image(self, multimodal_engine):
        """存在しない画像パスで質問しても例外が発生しない"""
        # 存在しない画像パスを指定
        result = multimodal_engine.query_with_images(
            query="この画像について教えて",
            image_paths=["/path/to/nonexistent/image.jpg"],
            n_results=3
        )

        # 警告は出るが、処理は継続される
        # 画像が使用されないことを確認
        assert result['images_used'] == 0

    def test_empty_query(self, multimodal_engine):
        """空のクエリでエラーが発生"""
        from src.rag.multimodal_engine import MultimodalRAGEngineError

        with pytest.raises(MultimodalRAGEngineError):
            multimodal_engine.query_with_images(query="")


@pytest.mark.integration
@pytest.mark.multimodal
@pytest.mark.performance
class TestMultimodalRAGPerformance:
    """マルチモーダルRAG パフォーマンステスト"""

    def test_search_performance(self, multimodal_engine):
        """検索のパフォーマンス確認"""
        import time

        start_time = time.time()
        results = multimodal_engine.search_multimodal(
            query="test query",
            top_k=10
        )
        elapsed_time = time.time() - start_time

        # 検索は2秒以内に完了すること
        assert elapsed_time < 2.0, f"検索が遅すぎます: {elapsed_time:.2f}秒"

    def test_query_with_images_performance(self, multimodal_engine, sample_images):
        """画像付き質問応答のパフォーマンス確認"""
        import time

        start_time = time.time()
        result = multimodal_engine.query_with_images(
            query="この画像の色は？",
            image_paths=[str(sample_images[0])],
            n_results=3
        )
        elapsed_time = time.time() - start_time

        # LLM応答を含めて10秒以内に完了すること
        assert elapsed_time < 10.0, f"質問応答が遅すぎます: {elapsed_time:.2f}秒"
