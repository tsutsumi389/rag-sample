"""ベクトルストアのユニットテスト

VectorStore クラスの機能を検証します。
外部依存（ChromaDB）はモック化してテストします。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.rag.vector_store import (
    VectorStore,
    VectorStoreError
)
from src.utils.config import Config


class TestVectorStoreInitialization:
    """VectorStore - 初期化のテスト"""

    def test_vector_store_instance_creation(self, monkeypatch, tmp_path):
        """VectorStoreインスタンスの作成"""
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

        # 空の.envファイルを作成
        empty_env_file = tmp_path / "empty.env"
        empty_env_file.write_text("")

        # Config を作成
        config = Config(env_file=str(empty_env_file))

        # VectorStoreインスタンスの作成
        vector_store = VectorStore(
            config=config,
            collection_name="test_collection"
        )

        # 属性の確認
        assert vector_store.config == config
        assert vector_store.collection_name == "test_collection"
        assert vector_store.client is None  # 初期化前はNone
        assert vector_store.collection is None  # 初期化前はNone

    def test_initialize_creates_client_and_collection(self, monkeypatch, tmp_path):
        """initialize()でChromaDBクライアントとコレクションが初期化される（モック）"""
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

        # テスト用ディレクトリを設定
        chroma_dir = tmp_path / "chroma_db"
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(chroma_dir))

        # 空の.envファイルを作成
        empty_env_file = tmp_path / "empty.env"
        empty_env_file.write_text("")

        config = Config(env_file=str(empty_env_file))

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            # モッククライアントとコレクションの作成
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 0

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config, collection_name="documents")
            vector_store.initialize()

            # クライアントが作成されたことを確認
            assert vector_store.client == mock_client
            assert vector_store.collection == mock_collection

            # PersistentClientが正しいパラメータで呼ばれたことを確認
            call_args = mock_client_class.call_args
            assert call_args is not None
            assert "path" in call_args.kwargs
            assert str(chroma_dir) in call_args.kwargs["path"]
            assert "settings" in call_args.kwargs

            # get_or_create_collectionが呼ばれたことを確認
            mock_client.get_or_create_collection.assert_called_once()
            call_kwargs = mock_client.get_or_create_collection.call_args.kwargs
            assert call_kwargs["name"] == "documents"
            assert "metadata" in call_kwargs

            # ディレクトリが作成されたことを確認
            assert chroma_dir.exists()

    def test_initialize_failure_raises_vector_store_error(self, monkeypatch, tmp_path):
        """初期化失敗時にVectorStoreErrorがraise（モック）"""
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

        # 空の.envファイルを作成
        empty_env_file = tmp_path / "empty.env"
        empty_env_file.write_text("")

        config = Config(env_file=str(empty_env_file))

        # PersistentClientの初期化時に例外を発生させる
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client_class.side_effect = Exception("Database connection failed")

            vector_store = VectorStore(config=config)

            # VectorStoreErrorがraiseされることを確認
            with pytest.raises(VectorStoreError) as exc_info:
                vector_store.initialize()

            # エラーメッセージに必要な情報が含まれることを確認
            error_message = str(exc_info.value)
            assert "ChromaDBの初期化に失敗しました" in error_message
            assert "Database connection failed" in error_message
