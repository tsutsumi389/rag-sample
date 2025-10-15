"""埋め込み生成のユニットテスト

EmbeddingGenerator クラスの機能を検証します。
外部依存（Ollama）はモック化してテストします。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.rag.embeddings import (
    EmbeddingGenerator,
    EmbeddingError,
    create_embedding_generator
)
from src.utils.config import Config


class TestEmbeddingGeneratorInitialization:
    """EmbeddingGenerator - 初期化のテスト"""

    def test_initialization_with_default_config(self, monkeypatch, tmp_path):
        """デフォルト設定での初期化"""
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

        # OllamaEmbeddingsのモック
        with patch("src.rag.embeddings.OllamaEmbeddings") as mock_ollama:
            mock_embeddings_instance = Mock()
            mock_ollama.return_value = mock_embeddings_instance

            # Config を明示的に作成
            config = Config(env_file=str(empty_env_file))
            generator = EmbeddingGenerator(config=config)

            # デフォルト設定値の確認
            assert generator.model_name == Config.DEFAULT_OLLAMA_EMBEDDING_MODEL
            assert generator.base_url == Config.DEFAULT_OLLAMA_BASE_URL
            assert generator.config == config
            assert generator.embeddings == mock_embeddings_instance

            # OllamaEmbeddingsが正しいパラメータで呼ばれたことを確認
            mock_ollama.assert_called_once_with(
                model=Config.DEFAULT_OLLAMA_EMBEDDING_MODEL,
                base_url=Config.DEFAULT_OLLAMA_BASE_URL
            )

    def test_initialization_with_custom_model_and_url(self):
        """カスタムmodel_name/base_urlでの初期化"""
        custom_model = "custom-embedding-model"
        custom_url = "http://custom-server:9999"

        # OllamaEmbeddingsのモック
        with patch("src.rag.embeddings.OllamaEmbeddings") as mock_ollama:
            mock_embeddings_instance = Mock()
            mock_ollama.return_value = mock_embeddings_instance

            generator = EmbeddingGenerator(
                model_name=custom_model,
                base_url=custom_url
            )

            # カスタム設定値の確認
            assert generator.model_name == custom_model
            assert generator.base_url == custom_url
            assert generator.embeddings == mock_embeddings_instance

            # OllamaEmbeddingsが正しいパラメータで呼ばれたことを確認
            mock_ollama.assert_called_once_with(
                model=custom_model,
                base_url=custom_url
            )

    def test_initialization_failure_raises_embedding_error(self):
        """Ollama接続失敗時にEmbeddingErrorがraise（モック）"""
        # OllamaEmbeddingsの初期化時に例外を発生させる
        with patch("src.rag.embeddings.OllamaEmbeddings") as mock_ollama:
            mock_ollama.side_effect = ConnectionError("Cannot connect to Ollama")

            # EmbeddingErrorがraiseされることを確認
            with pytest.raises(EmbeddingError) as exc_info:
                EmbeddingGenerator()

            # エラーメッセージに必要な情報が含まれることを確認
            error_message = str(exc_info.value)
            assert "Failed to initialize OllamaEmbeddings" in error_message
            assert "Make sure Ollama is running" in error_message
            assert "Cannot connect to Ollama" in error_message

    def test_repr_method(self):
        """__repr__メソッドが正しい文字列を返すことを確認"""
        custom_model = "test-model"
        custom_url = "http://test:8080"

        with patch("src.rag.embeddings.OllamaEmbeddings"):
            generator = EmbeddingGenerator(
                model_name=custom_model,
                base_url=custom_url
            )

            repr_str = repr(generator)
            assert "EmbeddingGenerator" in repr_str
            assert custom_model in repr_str
            assert custom_url in repr_str
