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


class TestEmbeddingGeneratorDocumentEmbedding:
    """EmbeddingGenerator - ドキュメント埋め込みのテスト"""

    def test_embed_documents_returns_correct_vectors(self):
        """embed_documents()で正しいベクトルリストが返される（モック）"""
        texts = ["テキスト1", "テキスト2", "テキスト3"]
        mock_vectors = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]

        with patch("src.rag.embeddings.OllamaEmbeddings") as mock_ollama:
            mock_embeddings_instance = Mock()
            mock_embeddings_instance.embed_documents.return_value = mock_vectors
            mock_ollama.return_value = mock_embeddings_instance

            generator = EmbeddingGenerator()
            result = generator.embed_documents(texts)

            # 結果の確認
            assert result == mock_vectors
            assert len(result) == len(texts)

            # embed_documentsが正しい引数で呼ばれたことを確認
            mock_embeddings_instance.embed_documents.assert_called_once_with(texts)

    def test_embed_documents_with_empty_list_raises_value_error(self):
        """空リストでValueErrorがraise"""
        with patch("src.rag.embeddings.OllamaEmbeddings"):
            generator = EmbeddingGenerator()

            with pytest.raises(ValueError) as exc_info:
                generator.embed_documents([])

            error_message = str(exc_info.value)
            assert "texts cannot be empty" in error_message

    def test_embed_documents_with_empty_string_raises_value_error(self):
        """空文字列を含むリストでValueErrorがraise"""
        with patch("src.rag.embeddings.OllamaEmbeddings"):
            generator = EmbeddingGenerator()

            # 空文字列を含むリスト
            texts_with_empty = ["text1", "", "text2"]

            with pytest.raises(ValueError) as exc_info:
                generator.embed_documents(texts_with_empty)

            error_message = str(exc_info.value)
            assert "texts cannot contain empty strings" in error_message

    def test_embed_documents_batch_processing(self):
        """バッチ処理が正しく動作する（モック）"""
        # 大量のテキストを準備
        texts = [f"テキスト{i}" for i in range(100)]
        mock_vectors = [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(100)]

        with patch("src.rag.embeddings.OllamaEmbeddings") as mock_ollama:
            mock_embeddings_instance = Mock()
            mock_embeddings_instance.embed_documents.return_value = mock_vectors
            mock_ollama.return_value = mock_embeddings_instance

            generator = EmbeddingGenerator()
            result = generator.embed_documents(texts)

            # 結果の確認
            assert len(result) == 100
            assert result == mock_vectors

            # embed_documentsが呼ばれたことを確認
            mock_embeddings_instance.embed_documents.assert_called_once_with(texts)


class TestEmbeddingGeneratorQueryEmbedding:
    """EmbeddingGenerator - クエリ埋め込みのテスト"""

    def test_embed_query_returns_correct_vector(self):
        """embed_query()で正しいベクトルが返される（モック）"""
        query = "これはテストクエリです"
        mock_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

        with patch("src.rag.embeddings.OllamaEmbeddings") as mock_ollama:
            mock_embeddings_instance = Mock()
            mock_embeddings_instance.embed_query.return_value = mock_vector
            mock_ollama.return_value = mock_embeddings_instance

            generator = EmbeddingGenerator()
            result = generator.embed_query(query)

            # 結果の確認
            assert result == mock_vector
            assert isinstance(result, list)

            # embed_queryが正しい引数で呼ばれたことを確認
            mock_embeddings_instance.embed_query.assert_called_once_with(query)

    def test_embed_query_with_empty_string_raises_value_error(self):
        """空文字列でValueErrorがraise"""
        with patch("src.rag.embeddings.OllamaEmbeddings"):
            generator = EmbeddingGenerator()

            with pytest.raises(ValueError) as exc_info:
                generator.embed_query("")

            error_message = str(exc_info.value)
            assert "text cannot be empty" in error_message


class TestEmbeddingGeneratorDimension:
    """EmbeddingGenerator - 次元数取得のテスト"""

    def test_get_embedding_dimension_returns_correct_dimension(self):
        """get_embedding_dimension()で正しい次元数が返される（モック）"""
        # 768次元のベクトルを模擬
        mock_vector = [0.1] * 768

        with patch("src.rag.embeddings.OllamaEmbeddings") as mock_ollama:
            mock_embeddings_instance = Mock()
            mock_embeddings_instance.embed_query.return_value = mock_vector
            mock_ollama.return_value = mock_embeddings_instance

            generator = EmbeddingGenerator()
            dimension = generator.get_embedding_dimension()

            # 次元数の確認
            assert dimension == 768
            assert isinstance(dimension, int)

            # embed_queryが"sample text"で呼ばれたことを確認
            mock_embeddings_instance.embed_query.assert_called_once_with("sample text")
