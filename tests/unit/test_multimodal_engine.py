"""マルチモーダルRAGエンジンのユニットテスト

MultimodalRAGEngine クラスの機能を検証します。
外部依存（Ollama、VectorStore、EmbeddingGenerator、VisionEmbeddings）はモック化してテストします。
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.rag.multimodal_engine import (
    MultimodalRAGEngine,
    MultimodalRAGEngineError,
    create_multimodal_rag_engine
)
from src.utils.config import Config
from src.models.document import SearchResult, Chunk, ChatMessage


class TestMultimodalRAGEngineInitialization:
    """MultimodalRAGEngine - 初期化のテスト"""

    def test_initialization_with_default_config(self, monkeypatch, tmp_path):
        """デフォルト設定での初期化（モック）"""
        # 環境変数をクリア
        for key in [
            "OLLAMA_BASE_URL",
            "OLLAMA_MULTIMODAL_LLM_MODEL",
            "OLLAMA_EMBEDDING_MODEL",
            "OLLAMA_VISION_MODEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        # 空の.envファイルを作成
        empty_env_file = tmp_path / "empty.env"
        empty_env_file.write_text("")

        # 各コンポーネントのモック
        with patch("src.rag.multimodal_engine.VectorStore") as mock_vector_store_cls, \
             patch("src.rag.multimodal_engine.EmbeddingGenerator") as mock_embedding_cls, \
             patch("src.rag.multimodal_engine.VisionEmbeddings") as mock_vision_cls, \
             patch("src.rag.multimodal_engine.ollama.Client") as mock_ollama_client_cls:

            mock_vector_store = Mock()
            mock_text_embeddings = Mock()
            mock_vision_embeddings = Mock()
            mock_ollama_client = Mock()

            # listメソッドのモック（モデル存在確認用）
            mock_ollama_client.list.return_value = {
                'models': [{'name': 'gemma3:latest'}]
            }

            mock_vector_store_cls.return_value = mock_vector_store
            mock_embedding_cls.return_value = mock_text_embeddings
            mock_vision_cls.return_value = mock_vision_embeddings
            mock_ollama_client_cls.return_value = mock_ollama_client

            # Config を明示的に作成
            config = Config(env_file=str(empty_env_file))
            engine = MultimodalRAGEngine(config=config)

            # 初期化の確認
            assert engine.config == config
            assert engine.vector_store == mock_vector_store
            assert engine.text_embeddings == mock_text_embeddings
            assert engine.vision_embeddings == mock_vision_embeddings
            assert engine.ollama_client == mock_ollama_client
            assert engine.chat_history is not None
            assert len(engine.chat_history) == 0

    def test_initialization_with_custom_components(self):
        """カスタムコンポーネントでの初期化"""
        custom_config = Mock(spec=Config)
        custom_config.ollama_base_url = "http://custom:11434"
        custom_config.ollama_embedding_model = "custom-embedding"

        custom_vector_store = Mock()
        custom_text_embeddings = Mock()
        custom_vision_embeddings = Mock()

        # Ollama Clientのモック
        with patch("src.rag.multimodal_engine.ollama.Client") as mock_ollama_client_cls:
            mock_ollama_client = Mock()
            mock_ollama_client.list.return_value = {
                'models': [{'name': 'gemma3:latest'}]
            }
            mock_ollama_client_cls.return_value = mock_ollama_client

            engine = MultimodalRAGEngine(
                config=custom_config,
                vector_store=custom_vector_store,
                text_embeddings=custom_text_embeddings,
                vision_embeddings=custom_vision_embeddings,
                llm_model="gemma3"
            )

            # カスタムインスタンスが使用されていることを確認
            assert engine.config == custom_config
            assert engine.vector_store == custom_vector_store
            assert engine.text_embeddings == custom_text_embeddings
            assert engine.vision_embeddings == custom_vision_embeddings
            assert engine.llm_model == "gemma3"

    def test_initialization_fails_when_model_not_available(self, tmp_path):
        """モデルが利用できない場合に初期化が失敗する"""
        empty_env_file = tmp_path / "empty.env"
        empty_env_file.write_text("")
        config = Config(env_file=str(empty_env_file))

        with patch("src.rag.multimodal_engine.VectorStore"), \
             patch("src.rag.multimodal_engine.EmbeddingGenerator"), \
             patch("src.rag.multimodal_engine.VisionEmbeddings"), \
             patch("src.rag.multimodal_engine.ollama.Client") as mock_ollama_client_cls:

            mock_ollama_client = Mock()
            # モデルリストに目的のモデルが存在しない
            mock_ollama_client.list.return_value = {
                'models': [{'name': 'other-model:latest'}]
            }
            mock_ollama_client_cls.return_value = mock_ollama_client

            with pytest.raises(MultimodalRAGEngineError, match="is not available"):
                MultimodalRAGEngine(config=config, llm_model="gemma3")


class TestMultimodalRAGEngineSearchImages:
    """MultimodalRAGEngine - 画像検索のテスト"""

    @pytest.fixture
    def engine(self):
        """テスト用のエンジンインスタンスを作成"""
        mock_config = Mock(spec=Config)
        mock_config.ollama_base_url = "http://localhost:11434"

        mock_vector_store = Mock()
        mock_text_embeddings = Mock()
        mock_vision_embeddings = Mock()

        with patch("src.rag.multimodal_engine.ollama.Client") as mock_ollama_client_cls:
            mock_ollama_client = Mock()
            mock_ollama_client.list.return_value = {
                'models': [{'name': 'gemma3:latest'}]
            }
            mock_ollama_client_cls.return_value = mock_ollama_client

            engine = MultimodalRAGEngine(
                config=mock_config,
                vector_store=mock_vector_store,
                text_embeddings=mock_text_embeddings,
                vision_embeddings=mock_vision_embeddings,
                llm_model="gemma3"
            )
            return engine

    def test_search_images_success(self, engine):
        """画像検索が成功する"""
        # テキスト埋め込みのモック
        mock_embedding = [0.1, 0.2, 0.3]
        engine.text_embeddings.embed_query.return_value = mock_embedding

        # 検索結果のモック
        mock_chunk = Chunk(
            content="犬の画像",
            chunk_id="img_001",
            document_id="img_001",
            chunk_index=0,
            start_char=0,
            end_char=10
        )
        mock_result = SearchResult(
            chunk=mock_chunk,
            score=0.95,
            document_name="dog.jpg",
            document_source="/path/to/dog.jpg",
            rank=1,
            result_type='image',
            image_path=Path("/path/to/dog.jpg"),
            caption="犬の画像"
        )
        engine.vector_store.search_images.return_value = [mock_result]

        # 画像検索を実行
        results = engine.search_images(query="犬の写真", top_k=5)

        # 検証
        assert len(results) == 1
        assert results[0] == mock_result
        engine.text_embeddings.embed_query.assert_called_once_with("犬の写真")
        engine.vector_store.search_images.assert_called_once_with(
            query_embedding=mock_embedding,
            top_k=5,
            collection_name="images"
        )

    def test_search_images_with_empty_query(self, engine):
        """空のクエリで検索するとエラーが発生"""
        with pytest.raises(MultimodalRAGEngineError, match="検索クエリが空です"):
            engine.search_images(query="")

    def test_search_images_returns_empty_list_when_no_results(self, engine):
        """検索結果がない場合は空のリストを返す"""
        engine.text_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        engine.vector_store.search_images.return_value = []

        results = engine.search_images(query="存在しない画像", top_k=5)

        assert results == []


class TestMultimodalRAGEngineSearchMultimodal:
    """MultimodalRAGEngine - マルチモーダル検索のテスト"""

    @pytest.fixture
    def engine(self):
        """テスト用のエンジンインスタンスを作成"""
        mock_config = Mock(spec=Config)
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.multimodal_search_text_weight = 0.6
        mock_config.multimodal_search_image_weight = 0.4

        mock_vector_store = Mock()
        mock_text_embeddings = Mock()
        mock_vision_embeddings = Mock()

        with patch("src.rag.multimodal_engine.ollama.Client") as mock_ollama_client_cls:
            mock_ollama_client = Mock()
            mock_ollama_client.list.return_value = {
                'models': [{'name': 'gemma3:latest'}]
            }
            mock_ollama_client_cls.return_value = mock_ollama_client

            engine = MultimodalRAGEngine(
                config=mock_config,
                vector_store=mock_vector_store,
                text_embeddings=mock_text_embeddings,
                vision_embeddings=mock_vision_embeddings,
                llm_model="gemma3"
            )
            return engine

    def test_search_multimodal_success(self, engine):
        """マルチモーダル検索が成功する"""
        # 埋め込みのモック
        mock_embedding = [0.1, 0.2, 0.3]
        engine.text_embeddings.embed_query.return_value = mock_embedding

        # テキスト検索結果のモック
        text_chunk = Chunk(
            content="犬に関するテキスト",
            chunk_id="text_001",
            document_id="doc_001",
            chunk_index=0,
            start_char=0,
            end_char=20
        )
        text_result = SearchResult(
            chunk=text_chunk,
            score=0.8,
            document_name="dog_article.txt",
            document_source="/path/to/dog_article.txt",
            rank=1
        )

        # 画像検索結果のモック
        image_chunk = Chunk(
            content="犬の画像",
            chunk_id="img_001",
            document_id="img_001",
            chunk_index=0,
            start_char=0,
            end_char=10
        )
        image_result = SearchResult(
            chunk=image_chunk,
            score=0.9,
            document_name="dog.jpg",
            document_source="/path/to/dog.jpg",
            rank=1,
            result_type='image'
        )

        engine.vector_store.search.return_value = [text_result]
        engine.vector_store.search_images.return_value = [image_result]

        # マルチモーダル検索を実行
        results = engine.search_multimodal(query="犬", top_k=5)

        # 検証
        assert len(results) == 2
        # スコアが重み付けされていることを確認
        # image_result: 0.9 * 0.4 = 0.36
        # text_result: 0.8 * 0.6 = 0.48
        # text_resultの方がスコアが高いので最初に来る
        assert results[0].chunk.content == "犬に関するテキスト"
        assert results[1].chunk.content == "犬の画像"

    def test_search_multimodal_with_custom_weights(self, engine):
        """カスタム重みでマルチモーダル検索を実行"""
        mock_embedding = [0.1, 0.2, 0.3]
        engine.text_embeddings.embed_query.return_value = mock_embedding

        text_chunk = Chunk(
            content="テキスト",
            chunk_id="text_001",
            document_id="doc_001",
            chunk_index=0,
            start_char=0,
            end_char=5
        )
        text_result = SearchResult(
            chunk=text_chunk,
            score=0.5,
            document_name="text.txt",
            document_source="/path/to/text.txt",
            rank=1
        )

        engine.vector_store.search.return_value = [text_result]
        engine.vector_store.search_images.return_value = []

        # カスタム重みで検索
        results = engine.search_multimodal(
            query="テスト",
            top_k=5,
            text_weight=0.7,
            image_weight=0.3
        )

        # テキスト結果のスコアが 0.5 * 0.7 = 0.35 に調整される
        assert len(results) == 1
        assert results[0].score == pytest.approx(0.35)


class TestMultimodalRAGEngineQueryWithImages:
    """MultimodalRAGEngine - 画像付き質問応答のテスト"""

    @pytest.fixture
    def engine(self):
        """テスト用のエンジンインスタンスを作成"""
        mock_config = Mock(spec=Config)
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.multimodal_search_text_weight = 0.5
        mock_config.multimodal_search_image_weight = 0.5

        mock_vector_store = Mock()
        mock_text_embeddings = Mock()
        mock_vision_embeddings = Mock()

        with patch("src.rag.multimodal_engine.ollama.Client") as mock_ollama_client_cls:
            mock_ollama_client = Mock()
            mock_ollama_client.list.return_value = {
                'models': [{'name': 'gemma3:latest'}]
            }
            mock_ollama_client_cls.return_value = mock_ollama_client

            engine = MultimodalRAGEngine(
                config=mock_config,
                vector_store=mock_vector_store,
                text_embeddings=mock_text_embeddings,
                vision_embeddings=mock_vision_embeddings,
                llm_model="gemma3"
            )
            return engine

    def test_query_with_images_success(self, engine, tmp_path):
        """画像付き質問応答が成功する"""
        # ダミー画像ファイルを作成
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"fake image data")

        # 埋め込みのモック
        mock_embedding = [0.1, 0.2, 0.3]
        engine.text_embeddings.embed_query.return_value = mock_embedding

        # 検索結果のモック
        text_chunk = Chunk(
            content="犬に関する情報",
            chunk_id="text_001",
            document_id="doc_001",
            chunk_index=0,
            start_char=0,
            end_char=15
        )
        text_result = SearchResult(
            chunk=text_chunk,
            score=0.8,
            document_name="dog_info.txt",
            document_source="/path/to/dog_info.txt",
            rank=1,
            result_type='text'
        )
        engine.vector_store.search.return_value = [text_result]
        engine.vector_store.search_images.return_value = []

        # Ollama chat APIのモック
        engine.ollama_client.chat.return_value = {
            'message': {
                'content': 'この画像には犬が写っています。'
            }
        }

        # 画像付き質問を実行
        result = engine.query_with_images(
            query="この画像について説明して",
            image_paths=[str(image_file)],
            n_results=5
        )

        # 検証
        assert result['answer'] == 'この画像には犬が写っています。'
        assert result['context_count'] == 1
        assert result['images_used'] == 1
        assert 'sources' in result

        # Ollama chat APIが正しいパラメータで呼ばれたことを確認
        engine.ollama_client.chat.assert_called_once()
        call_args = engine.ollama_client.chat.call_args
        assert call_args[1]['model'] == 'gemma3'
        assert 'images' in call_args[1]['messages'][0]
        assert str(image_file) in call_args[1]['messages'][0]['images']

    def test_query_with_images_empty_query(self, engine):
        """空の質問でエラーが発生"""
        with pytest.raises(MultimodalRAGEngineError, match="質問が空です"):
            engine.query_with_images(query="")

    def test_query_with_images_no_images(self, engine):
        """画像なしでも質問応答が実行できる"""
        mock_embedding = [0.1, 0.2, 0.3]
        engine.text_embeddings.embed_query.return_value = mock_embedding

        engine.vector_store.search.return_value = []
        engine.vector_store.search_images.return_value = []

        engine.ollama_client.chat.return_value = {
            'message': {
                'content': '提供された情報では回答できません'
            }
        }

        result = engine.query_with_images(query="犬について教えて")

        assert result['answer'] == '提供された情報では回答できません'
        assert result['images_used'] == 0


class TestMultimodalRAGEngineChatMultimodal:
    """MultimodalRAGEngine - マルチモーダルチャットのテスト"""

    @pytest.fixture
    def engine(self):
        """テスト用のエンジンインスタンスを作成"""
        mock_config = Mock(spec=Config)
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.multimodal_search_text_weight = 0.5
        mock_config.multimodal_search_image_weight = 0.5

        mock_vector_store = Mock()
        mock_text_embeddings = Mock()
        mock_vision_embeddings = Mock()

        with patch("src.rag.multimodal_engine.ollama.Client") as mock_ollama_client_cls:
            mock_ollama_client = Mock()
            mock_ollama_client.list.return_value = {
                'models': [{'name': 'gemma3:latest'}]
            }
            mock_ollama_client_cls.return_value = mock_ollama_client

            engine = MultimodalRAGEngine(
                config=mock_config,
                vector_store=mock_vector_store,
                text_embeddings=mock_text_embeddings,
                vision_embeddings=mock_vision_embeddings,
                llm_model="gemma3"
            )
            return engine

    def test_chat_multimodal_adds_messages_to_history(self, engine):
        """チャットメッセージが履歴に追加される"""
        # モックの設定
        engine.text_embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
        engine.vector_store.search.return_value = []
        engine.vector_store.search_images.return_value = []
        engine.ollama_client.chat.return_value = {
            'message': {'content': '回答です'}
        }

        # チャットを実行
        result = engine.chat_multimodal(message="こんにちは")

        # 履歴の確認
        assert len(engine.chat_history) == 2
        assert engine.chat_history.messages[0].role == "user"
        assert engine.chat_history.messages[0].content == "こんにちは"
        assert engine.chat_history.messages[1].role == "assistant"
        assert engine.chat_history.messages[1].content == "回答です"
        assert result['history_length'] == 2

    def test_clear_chat_history(self, engine):
        """チャット履歴のクリア"""
        # 履歴に追加
        engine.chat_history.add_message(role="user", content="テスト")
        assert len(engine.chat_history) == 1

        # クリア
        engine.clear_chat_history()
        assert len(engine.chat_history) == 0


class TestMultimodalRAGEngineUtilities:
    """MultimodalRAGEngine - ユーティリティ機能のテスト"""

    @pytest.fixture
    def engine(self):
        """テスト用のエンジンインスタンスを作成"""
        mock_config = Mock(spec=Config)
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_embedding_model = "nomic-embed-text"

        mock_vector_store = Mock()
        mock_text_embeddings = Mock()
        mock_vision_embeddings = Mock()
        mock_vision_embeddings.model_name = "llava"

        with patch("src.rag.multimodal_engine.ollama.Client") as mock_ollama_client_cls:
            mock_ollama_client = Mock()
            mock_ollama_client.list.return_value = {
                'models': [{'name': 'gemma3:latest'}]
            }
            mock_ollama_client_cls.return_value = mock_ollama_client

            engine = MultimodalRAGEngine(
                config=mock_config,
                vector_store=mock_vector_store,
                text_embeddings=mock_text_embeddings,
                vision_embeddings=mock_vision_embeddings,
                llm_model="gemma3"
            )
            return engine

    def test_get_status(self, engine):
        """ステータス情報の取得"""
        engine.vector_store.get_collection_info.return_value = {
            'name': 'documents',
            'count': 10
        }

        status = engine.get_status()

        assert status['multimodal_llm_model'] == 'gemma3'
        assert status['text_embedding_model'] == 'nomic-embed-text'
        assert status['vision_embedding_model'] == 'llava'
        assert status['vector_store_info']['count'] == 10
        assert status['chat_history_length'] == 0

    def test_initialize(self, engine):
        """初期化処理"""
        engine.initialize()

        engine.vector_store.initialize.assert_called_once()

    def test_context_manager(self, engine):
        """コンテキストマネージャーとして使用できる"""
        with engine as e:
            assert e == engine
            e.vector_store.initialize.assert_called_once()

        engine.vector_store.close.assert_called_once()


class TestCreateMultimodalRAGEngine:
    """create_multimodal_rag_engine 関数のテスト"""

    def test_create_multimodal_rag_engine_default(self, tmp_path):
        """デフォルト設定でエンジンを作成"""
        empty_env_file = tmp_path / "empty.env"
        empty_env_file.write_text("")
        config = Config(env_file=str(empty_env_file))

        with patch("src.rag.multimodal_engine.VectorStore"), \
             patch("src.rag.multimodal_engine.EmbeddingGenerator"), \
             patch("src.rag.multimodal_engine.VisionEmbeddings"), \
             patch("src.rag.multimodal_engine.ollama.Client") as mock_ollama_client_cls:

            mock_ollama_client = Mock()
            mock_ollama_client.list.return_value = {
                'models': [{'name': 'gemma3:latest'}]
            }
            mock_ollama_client_cls.return_value = mock_ollama_client

            engine = create_multimodal_rag_engine(config=config)

            assert isinstance(engine, MultimodalRAGEngine)
            assert engine.config == config

    def test_create_multimodal_rag_engine_with_custom_model(self, tmp_path):
        """カスタムモデルでエンジンを作成"""
        empty_env_file = tmp_path / "empty.env"
        empty_env_file.write_text("")
        config = Config(env_file=str(empty_env_file))

        with patch("src.rag.multimodal_engine.VectorStore"), \
             patch("src.rag.multimodal_engine.EmbeddingGenerator"), \
             patch("src.rag.multimodal_engine.VisionEmbeddings"), \
             patch("src.rag.multimodal_engine.ollama.Client") as mock_ollama_client_cls:

            mock_ollama_client = Mock()
            mock_ollama_client.list.return_value = {
                'models': [{'name': 'custom-model:latest'}]
            }
            mock_ollama_client_cls.return_value = mock_ollama_client

            engine = create_multimodal_rag_engine(
                config=config,
                llm_model="custom-model"
            )

            assert isinstance(engine, MultimodalRAGEngine)
            assert engine.llm_model == "custom-model"
