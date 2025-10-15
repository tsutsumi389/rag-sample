"""RAGエンジンのユニットテスト

RAGEngine クラスの機能を検証します。
外部依存（Ollama、VectorStore、EmbeddingGenerator）はモック化してテストします。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.rag.engine import (
    RAGEngine,
    RAGEngineError,
    create_rag_engine
)
from src.utils.config import Config


class TestRAGEngineInitialization:
    """RAGEngine - 初期化のテスト"""

    def test_initialization_with_default_config(self, monkeypatch, tmp_path):
        """デフォルト設定での初期化（モック）"""
        # 環境変数をクリアしてデフォルト値を使用
        for key in [
            "OLLAMA_BASE_URL",
            "OLLAMA_LLM_MODEL",
            "OLLAMA_EMBEDDING_MODEL",
            "CHROMA_PERSIST_DIRECTORY",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "LOG_LEVEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        # 空の.envファイルを作成
        empty_env_file = tmp_path / "empty.env"
        empty_env_file.write_text("")

        # 各コンポーネントのモック
        with patch("src.rag.engine.VectorStore") as mock_vector_store_cls, \
             patch("src.rag.engine.EmbeddingGenerator") as mock_embedding_cls, \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_vector_store = Mock()
            mock_embedding_generator = Mock()
            mock_llm = Mock()

            mock_vector_store_cls.return_value = mock_vector_store
            mock_embedding_cls.return_value = mock_embedding_generator
            mock_llm_cls.return_value = mock_llm

            # Config を明示的に作成
            config = Config(env_file=str(empty_env_file))
            engine = RAGEngine(config=config)

            # 初期化の確認
            assert engine.config == config
            assert engine.vector_store == mock_vector_store
            assert engine.embedding_generator == mock_embedding_generator
            assert engine.llm == mock_llm
            assert engine.chat_history is not None
            assert len(engine.chat_history) == 0

            # VectorStoreが正しいパラメータで呼ばれたことを確認
            mock_vector_store_cls.assert_called_once_with(config)

            # EmbeddingGeneratorが正しいパラメータで呼ばれたことを確認
            mock_embedding_cls.assert_called_once_with(config)

            # OllamaLLMが正しいパラメータで呼ばれたことを確認
            mock_llm_cls.assert_called_once_with(
                model=Config.DEFAULT_OLLAMA_LLM_MODEL,
                base_url=Config.DEFAULT_OLLAMA_BASE_URL
            )

    def test_initialization_with_custom_vector_store_and_embedding_generator(self):
        """カスタムvector_store/embedding_generatorでの初期化"""
        # カスタムのインスタンスを作成
        custom_config = Mock(spec=Config)
        custom_config.ollama_llm_model = "custom-llm-model"
        custom_config.ollama_base_url = "http://custom:11434"

        custom_vector_store = Mock()
        custom_embedding_generator = Mock()

        # OllamaLLMのモック
        with patch("src.rag.engine.OllamaLLM") as mock_llm_cls:
            mock_llm = Mock()
            mock_llm_cls.return_value = mock_llm

            engine = RAGEngine(
                config=custom_config,
                vector_store=custom_vector_store,
                embedding_generator=custom_embedding_generator
            )

            # カスタムインスタンスが使用されていることを確認
            assert engine.config == custom_config
            assert engine.vector_store == custom_vector_store
            assert engine.embedding_generator == custom_embedding_generator
            assert engine.llm == mock_llm

            # OllamaLLMがカスタム設定で呼ばれたことを確認
            mock_llm_cls.assert_called_once_with(
                model="custom-llm-model",
                base_url="http://custom:11434"
            )

    def test_initialization_with_custom_llm_model(self, monkeypatch, tmp_path):
        """カスタムLLMモデル名での初期化"""
        # 環境変数をクリア
        for key in [
            "OLLAMA_BASE_URL",
            "OLLAMA_LLM_MODEL",
            "OLLAMA_EMBEDDING_MODEL",
            "CHROMA_PERSIST_DIRECTORY",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "LOG_LEVEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        empty_env_file = tmp_path / "empty.env"
        empty_env_file.write_text("")

        custom_llm_model = "llama3.3"

        # 各コンポーネントのモック
        with patch("src.rag.engine.VectorStore") as mock_vector_store_cls, \
             patch("src.rag.engine.EmbeddingGenerator") as mock_embedding_cls, \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_vector_store_cls.return_value = Mock()
            mock_embedding_cls.return_value = Mock()
            mock_llm_cls.return_value = Mock()

            config = Config(env_file=str(empty_env_file))
            engine = RAGEngine(config=config, llm_model=custom_llm_model)

            # カスタムLLMモデルが使用されていることを確認
            mock_llm_cls.assert_called_once_with(
                model=custom_llm_model,
                base_url=Config.DEFAULT_OLLAMA_BASE_URL
            )

    def test_initialization_with_max_chat_history(self):
        """max_chat_historyパラメータの設定"""
        custom_max_history = 20

        # 各コンポーネントのモック
        with patch("src.rag.engine.VectorStore") as mock_vector_store_cls, \
             patch("src.rag.engine.EmbeddingGenerator") as mock_embedding_cls, \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_vector_store_cls.return_value = Mock()
            mock_embedding_cls.return_value = Mock()
            mock_llm_cls.return_value = Mock()

            engine = RAGEngine(max_chat_history=custom_max_history)

            # チャット履歴の最大値が設定されていることを確認
            assert engine.chat_history.max_messages == custom_max_history

    def test_llm_initialization_failure_raises_rag_engine_error(self):
        """LLM初期化失敗時にRAGEngineErrorがraise（モック）"""
        # VectorStoreとEmbeddingGeneratorのモック
        with patch("src.rag.engine.VectorStore") as mock_vector_store_cls, \
             patch("src.rag.engine.EmbeddingGenerator") as mock_embedding_cls, \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_vector_store_cls.return_value = Mock()
            mock_embedding_cls.return_value = Mock()

            # OllamaLLMの初期化時に例外を発生させる
            mock_llm_cls.side_effect = ConnectionError("Cannot connect to Ollama")

            # RAGEngineErrorがraiseされることを確認
            with pytest.raises(RAGEngineError) as exc_info:
                RAGEngine()

            # エラーメッセージに必要な情報が含まれることを確認
            error_message = str(exc_info.value)
            assert "LLMの初期化に失敗しました" in error_message
            assert "Ollamaが" in error_message
            assert "で起動しており" in error_message
            assert "Cannot connect to Ollama" in error_message

    def test_default_config_usage_when_not_provided(self):
        """configが省略された場合にデフォルト設定が使用される"""
        # get_configのモック
        with patch("src.rag.engine.get_config") as mock_get_config, \
             patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM"):

            mock_config = Mock(spec=Config)
            mock_config.ollama_llm_model = "default-model"
            mock_config.ollama_base_url = "http://localhost:11434"
            mock_get_config.return_value = mock_config

            engine = RAGEngine()

            # get_configが呼ばれたことを確認
            mock_get_config.assert_called_once()
            assert engine.config == mock_config


class TestCreateRAGEngine:
    """create_rag_engine便利関数のテスト"""

    def test_create_rag_engine_with_defaults(self):
        """デフォルト設定でRAGエンジンを作成"""
        with patch("src.rag.engine.RAGEngine") as mock_rag_engine_cls:
            mock_engine = Mock()
            mock_rag_engine_cls.return_value = mock_engine

            result = create_rag_engine()

            # RAGEngineが正しいパラメータで呼ばれたことを確認
            mock_rag_engine_cls.assert_called_once_with(
                config=None,
                llm_model=None
            )
            assert result == mock_engine

    def test_create_rag_engine_with_custom_config_and_model(self):
        """カスタム設定とモデルでRAGエンジンを作成"""
        custom_config = Mock(spec=Config)
        custom_model = "llama3.3"

        with patch("src.rag.engine.RAGEngine") as mock_rag_engine_cls:
            mock_engine = Mock()
            mock_rag_engine_cls.return_value = mock_engine

            result = create_rag_engine(
                config=custom_config,
                llm_model=custom_model
            )

            # RAGEngineが正しいパラメータで呼ばれたことを確認
            mock_rag_engine_cls.assert_called_once_with(
                config=custom_config,
                llm_model=custom_model
            )
            assert result == mock_engine
