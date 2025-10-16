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


class TestVectorStoreAddDocuments:
    """VectorStore - ドキュメント追加のテスト"""

    def test_add_documents_successfully(self, monkeypatch, tmp_path):
        """add_documents()で正しくChunkが追加される（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 0

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config, collection_name="documents")
            vector_store.initialize()

            # テスト用Chunkの作成
            from src.models.document import Chunk

            chunks = [
                Chunk(
                    content="これはテストチャンク1です。",
                    chunk_id="doc1_chunk_0000",
                    document_id="doc1",
                    chunk_index=0,
                    start_char=0,
                    end_char=15,
                    metadata={
                        "document_name": "test.txt",
                        "source": "/path/to/test.txt",
                        "doc_type": "txt"
                    }
                ),
                Chunk(
                    content="これはテストチャンク2です。",
                    chunk_id="doc1_chunk_0001",
                    document_id="doc1",
                    chunk_index=1,
                    start_char=15,
                    end_char=30,
                    metadata={
                        "document_name": "test.txt",
                        "source": "/path/to/test.txt",
                        "doc_type": "txt"
                    }
                )
            ]

            # テスト用埋め込みベクトル
            embeddings = [
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.2, 0.3, 0.4, 0.5, 0.6]
            ]

            # ドキュメント追加
            vector_store.add_documents(chunks, embeddings)

            # collection.add()が正しいパラメータで呼ばれたことを確認
            mock_collection.add.assert_called_once()
            call_kwargs = mock_collection.add.call_args.kwargs

            assert call_kwargs["ids"] == ["doc1_chunk_0000", "doc1_chunk_0001"]
            assert call_kwargs["documents"] == [
                "これはテストチャンク1です。",
                "これはテストチャンク2です。"
            ]
            assert call_kwargs["embeddings"] == embeddings
            assert len(call_kwargs["metadatas"]) == 2
            assert call_kwargs["metadatas"][0]["document_name"] == "test.txt"

    def test_add_documents_mismatched_lengths_raises_error(self, monkeypatch, tmp_path):
        """chunksとembeddingsの長さが不一致でVectorStoreErrorがraise"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 0

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # テスト用Chunkの作成
            from src.models.document import Chunk

            chunks = [
                Chunk(
                    content="テスト",
                    chunk_id="doc1_chunk_0000",
                    document_id="doc1",
                    chunk_index=0,
                    start_char=0,
                    end_char=3,
                )
            ]

            # 長さが異なる埋め込みベクトル（2つ）
            embeddings = [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6]
            ]

            # VectorStoreErrorがraiseされることを確認
            with pytest.raises(VectorStoreError) as exc_info:
                vector_store.add_documents(chunks, embeddings)

            error_message = str(exc_info.value)
            assert "チャンク数(1)と埋め込み数(2)が一致しません" in error_message

    def test_add_documents_empty_list_logs_warning(self, monkeypatch, tmp_path, caplog):
        """空リストの追加で警告ログが出力される"""
        import logging

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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 0

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # 空のリストで追加
            with caplog.at_level(logging.WARNING):
                vector_store.add_documents([], [])

            # 警告ログが出力されたことを確認
            assert "追加するチャンクがありません" in caplog.text

            # collection.add()が呼ばれていないことを確認
            mock_collection.add.assert_not_called()

    def test_add_documents_without_initialization_raises_error(self, monkeypatch, tmp_path):
        """コレクション未初期化でVectorStoreErrorがraise"""
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

        # VectorStoreの作成（初期化なし）
        vector_store = VectorStore(config=config)

        # テスト用Chunkの作成
        from src.models.document import Chunk

        chunks = [
            Chunk(
                content="テスト",
                chunk_id="doc1_chunk_0000",
                document_id="doc1",
                chunk_index=0,
                start_char=0,
                end_char=3,
            )
        ]
        embeddings = [[0.1, 0.2, 0.3]]

        # VectorStoreErrorがraiseされることを確認
        with pytest.raises(VectorStoreError) as exc_info:
            vector_store.add_documents(chunks, embeddings)

        error_message = str(exc_info.value)
        assert "コレクションが初期化されていません" in error_message


class TestVectorStoreSearch:
    """VectorStore - 検索のテスト"""

    def test_search_returns_correct_results(self, monkeypatch, tmp_path):
        """search()で正しいSearchResultリストが返される（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 10

            # 検索結果のモックデータ
            mock_collection.query.return_value = {
                'ids': [['chunk_1', 'chunk_2', 'chunk_3']],
                'documents': [['ドキュメント1の内容', 'ドキュメント2の内容', 'ドキュメント3の内容']],
                'metadatas': [[
                    {
                        'document_id': 'doc1',
                        'document_name': 'test1.txt',
                        'source': '/path/to/test1.txt',
                        'chunk_index': 0,
                        'start_char': 0,
                        'end_char': 10,
                        'size': 10
                    },
                    {
                        'document_id': 'doc2',
                        'document_name': 'test2.txt',
                        'source': '/path/to/test2.txt',
                        'chunk_index': 0,
                        'start_char': 0,
                        'end_char': 10,
                        'size': 10
                    },
                    {
                        'document_id': 'doc3',
                        'document_name': 'test3.txt',
                        'source': '/path/to/test3.txt',
                        'chunk_index': 0,
                        'start_char': 0,
                        'end_char': 10,
                        'size': 10
                    }
                ]],
                'distances': [[0.1, 0.3, 0.5]]
            }

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config, collection_name="documents")
            vector_store.initialize()

            # 検索実行
            query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
            results = vector_store.search(query_embedding, n_results=3)

            # 結果の検証
            assert len(results) == 3
            assert results[0].chunk.content == 'ドキュメント1の内容'
            assert results[0].chunk.chunk_id == 'chunk_1'
            assert results[0].document_name == 'test1.txt'
            assert results[0].document_source == '/path/to/test1.txt'
            assert results[0].rank == 1

            # スコアの検証（距離0.1から変換）
            expected_score_1 = 1.0 / (1.0 + 0.1)
            assert abs(results[0].score - expected_score_1) < 0.01

            assert results[1].chunk.content == 'ドキュメント2の内容'
            assert results[1].rank == 2

            assert results[2].chunk.content == 'ドキュメント3の内容'
            assert results[2].rank == 3

            # collection.query()が正しいパラメータで呼ばれたことを確認
            mock_collection.query.assert_called_once()
            call_kwargs = mock_collection.query.call_args.kwargs
            assert call_kwargs['query_embeddings'] == [query_embedding]
            assert call_kwargs['n_results'] == 3
            assert call_kwargs['include'] == ["documents", "metadatas", "distances"]

    def test_search_with_where_filter(self, monkeypatch, tmp_path):
        """whereフィルタが正しく適用される（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 10

            # フィルタリングされた検索結果のモックデータ
            mock_collection.query.return_value = {
                'ids': [['chunk_1']],
                'documents': [['フィルタされたドキュメント']],
                'metadatas': [[
                    {
                        'document_id': 'doc1',
                        'document_name': 'test1.txt',
                        'source': '/path/to/test1.txt',
                        'chunk_index': 0,
                        'start_char': 0,
                        'end_char': 15,
                        'size': 15
                    }
                ]],
                'distances': [[0.2]]
            }

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # whereフィルタを使用して検索
            query_embedding = [0.1, 0.2, 0.3]
            where_filter = {"document_id": "doc1"}
            results = vector_store.search(
                query_embedding,
                n_results=5,
                where=where_filter
            )

            # 結果の検証
            assert len(results) == 1
            assert results[0].chunk.document_id == 'doc1'

            # whereフィルタが正しく渡されたことを確認
            mock_collection.query.assert_called_once()
            call_kwargs = mock_collection.query.call_args.kwargs
            assert call_kwargs['where'] == where_filter

    def test_search_with_n_results_parameter(self, monkeypatch, tmp_path):
        """n_resultsパラメータが機能する（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 10

            # 10件の検索結果のモックデータを作成
            ids = [[f'chunk_{i}' for i in range(10)]]
            documents = [[f'ドキュメント{i}の内容' for i in range(10)]]
            metadatas = [[{
                'document_id': f'doc{i}',
                'document_name': f'test{i}.txt',
                'source': f'/path/to/test{i}.txt',
                'chunk_index': 0,
                'start_char': 0,
                'end_char': 10,
                'size': 10
            } for i in range(10)]]
            distances = [[0.1 * i for i in range(10)]]

            mock_collection.query.return_value = {
                'ids': ids,
                'documents': documents,
                'metadatas': metadatas,
                'distances': distances
            }

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # n_results=10で検索
            query_embedding = [0.1, 0.2, 0.3]
            results = vector_store.search(query_embedding, n_results=10)

            # 結果の検証
            assert len(results) == 10

            # n_resultsが正しく渡されたことを確認
            mock_collection.query.assert_called_once()
            call_kwargs = mock_collection.query.call_args.kwargs
            assert call_kwargs['n_results'] == 10

    def test_search_returns_empty_list_when_no_results(self, monkeypatch, tmp_path):
        """検索結果が空の場合に空リストが返される（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 0

            # 空の検索結果
            mock_collection.query.return_value = {
                'ids': [[]],
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # 検索実行
            query_embedding = [0.1, 0.2, 0.3]
            results = vector_store.search(query_embedding, n_results=5)

            # 空のリストが返されることを確認
            assert results == []

    def test_search_score_calculation(self, monkeypatch, tmp_path):
        """スコア計算（距離から類似度への変換）が正しい"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 3

            # 異なる距離の検索結果をモック
            mock_collection.query.return_value = {
                'ids': [['chunk_1', 'chunk_2', 'chunk_3']],
                'documents': [['Doc1', 'Doc2', 'Doc3']],
                'metadatas': [[
                    {
                        'document_id': 'doc1',
                        'document_name': 'test1.txt',
                        'source': '/path/to/test1.txt',
                        'chunk_index': 0,
                        'start_char': 0,
                        'end_char': 4,
                        'size': 4
                    },
                    {
                        'document_id': 'doc2',
                        'document_name': 'test2.txt',
                        'source': '/path/to/test2.txt',
                        'chunk_index': 0,
                        'start_char': 0,
                        'end_char': 4,
                        'size': 4
                    },
                    {
                        'document_id': 'doc3',
                        'document_name': 'test3.txt',
                        'source': '/path/to/test3.txt',
                        'chunk_index': 0,
                        'start_char': 0,
                        'end_char': 4,
                        'size': 4
                    }
                ]],
                'distances': [[0.0, 1.0, 4.0]]  # 距離: 0.0, 1.0, 4.0
            }

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # 検索実行
            query_embedding = [0.1, 0.2, 0.3]
            results = vector_store.search(query_embedding, n_results=3)

            # スコア計算の検証
            # score = 1.0 / (1.0 + distance)

            # 距離0.0のスコア: 1.0 / (1.0 + 0.0) = 1.0
            assert abs(results[0].score - 1.0) < 0.01

            # 距離1.0のスコア: 1.0 / (1.0 + 1.0) = 0.5
            assert abs(results[1].score - 0.5) < 0.01

            # 距離4.0のスコア: 1.0 / (1.0 + 4.0) = 0.2
            assert abs(results[2].score - 0.2) < 0.01

            # スコアが降順（類似度が高い順）になっていることを確認
            assert results[0].score > results[1].score > results[2].score


class TestVectorStoreDelete:
    """VectorStore - 削除のテスト"""

    def test_delete_by_document_id(self, monkeypatch, tmp_path):
        """delete()でdocument_id指定による削除（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()

            # 初期化時: 10、削除前: 10、削除後: 7（3件削除）
            mock_collection.count.side_effect = [10, 10, 7]
            mock_collection.delete = Mock()

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # document_id指定で削除
            deleted_count = vector_store.delete(document_id="doc1")

            # 削除メソッドが正しく呼ばれたことを確認
            mock_collection.delete.assert_called_once_with(
                where={"document_id": "doc1"}
            )

            # 削除件数が正しく返されることを確認
            assert deleted_count == 3

    def test_delete_by_chunk_ids(self, monkeypatch, tmp_path):
        """delete()でchunk_ids指定による削除（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()

            # 初期化時: 10、削除前: 10、削除後: 8（2件削除）
            mock_collection.count.side_effect = [10, 10, 8]
            mock_collection.delete = Mock()

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # chunk_ids指定で削除
            chunk_ids_to_delete = ["chunk_1", "chunk_2"]
            deleted_count = vector_store.delete(chunk_ids=chunk_ids_to_delete)

            # 削除メソッドが正しく呼ばれたことを確認
            mock_collection.delete.assert_called_once_with(ids=chunk_ids_to_delete)

            # 削除件数が正しく返されることを確認
            assert deleted_count == 2

    def test_delete_by_where_filter(self, monkeypatch, tmp_path):
        """delete()でwhereフィルタによる削除（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()

            # 初期化時: 15、削除前: 15、削除後: 10（5件削除）
            mock_collection.count.side_effect = [15, 15, 10]
            mock_collection.delete = Mock()

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # whereフィルタで削除
            where_filter = {"doc_type": "txt"}
            deleted_count = vector_store.delete(where=where_filter)

            # 削除メソッドが正しく呼ばれたことを確認
            mock_collection.delete.assert_called_once_with(where=where_filter)

            # 削除件数が正しく返されることを確認
            assert deleted_count == 5

    def test_delete_without_conditions_raises_error(self, monkeypatch, tmp_path):
        """削除条件未指定でVectorStoreErrorがraise"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 10

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # 削除条件なしで呼び出し
            with pytest.raises(VectorStoreError) as exc_info:
                vector_store.delete()

            # エラーメッセージの確認
            error_message = str(exc_info.value)
            assert "削除条件が指定されていません" in error_message

    def test_delete_returns_correct_count(self, monkeypatch, tmp_path):
        """削除件数が正しく返される（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()

            # 初期化時: 100、削除前: 100、削除後: 75（25件削除）
            mock_collection.count.side_effect = [100, 100, 75]
            mock_collection.delete = Mock()

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # 削除実行
            deleted_count = vector_store.delete(document_id="large_doc")

            # 削除件数が正しいことを確認
            assert deleted_count == 25

            # countが3回呼ばれたことを確認（初期化時1回 + 削除前後2回）
            assert mock_collection.count.call_count == 3


class TestVectorStoreOtherOperations:
    """VectorStore - その他操作のテスト"""

    def test_list_documents(self, monkeypatch, tmp_path):
        """list_documents()でドキュメント一覧が取得できる（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 5

            # 複数ドキュメントのチャンクを含むデータ
            mock_collection.get.return_value = {
                'ids': ['doc1_chunk_0', 'doc1_chunk_1', 'doc2_chunk_0', 'doc2_chunk_1', 'doc2_chunk_2'],
                'documents': ['Content1', 'Content2', 'Content3', 'Content4', 'Content5'],
                'metadatas': [
                    {
                        'document_id': 'doc1',
                        'document_name': 'test1.txt',
                        'source': '/path/to/test1.txt',
                        'doc_type': 'txt',
                        'size': 100
                    },
                    {
                        'document_id': 'doc1',
                        'document_name': 'test1.txt',
                        'source': '/path/to/test1.txt',
                        'doc_type': 'txt',
                        'size': 120
                    },
                    {
                        'document_id': 'doc2',
                        'document_name': 'test2.md',
                        'source': '/path/to/test2.md',
                        'doc_type': 'md',
                        'size': 80
                    },
                    {
                        'document_id': 'doc2',
                        'document_name': 'test2.md',
                        'source': '/path/to/test2.md',
                        'doc_type': 'md',
                        'size': 90
                    },
                    {
                        'document_id': 'doc2',
                        'document_name': 'test2.md',
                        'source': '/path/to/test2.md',
                        'doc_type': 'md',
                        'size': 110
                    }
                ]
            }

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # ドキュメント一覧を取得
            documents = vector_store.list_documents()

            # 結果の検証
            assert len(documents) == 2  # 2つのドキュメント

            # doc1の検証
            doc1 = next(d for d in documents if d['document_id'] == 'doc1')
            assert doc1['document_name'] == 'test1.txt'
            assert doc1['source'] == '/path/to/test1.txt'
            assert doc1['doc_type'] == 'txt'
            assert doc1['chunk_count'] == 2
            assert doc1['total_size'] == 220  # 100 + 120

            # doc2の検証
            doc2 = next(d for d in documents if d['document_id'] == 'doc2')
            assert doc2['document_name'] == 'test2.md'
            assert doc2['source'] == '/path/to/test2.md'
            assert doc2['doc_type'] == 'md'
            assert doc2['chunk_count'] == 3
            assert doc2['total_size'] == 280  # 80 + 90 + 110

            # collection.get()が呼ばれたことを確認
            mock_collection.get.assert_called_once()
            call_kwargs = mock_collection.get.call_args.kwargs
            assert call_kwargs['include'] == ["metadatas", "documents"]

    def test_clear(self, monkeypatch, tmp_path):
        """clear()で全データが削除される（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()

            # 初期化時: 10個、clear実行時: 10個
            mock_collection.count.side_effect = [10, 10]

            # 新しいコレクションのモック
            new_collection = Mock()
            new_collection.count.return_value = 0

            mock_client.get_or_create_collection.side_effect = [
                mock_collection,  # 初回の初期化
                new_collection    # clear後の再作成
            ]
            mock_client.delete_collection = Mock()
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config, collection_name="test_collection")
            vector_store.initialize()

            # clear実行
            vector_store.clear()

            # delete_collectionが呼ばれたことを確認
            mock_client.delete_collection.assert_called_once_with("test_collection")

            # get_or_create_collectionが2回呼ばれたことを確認
            assert mock_client.get_or_create_collection.call_count == 2

            # コレクションが再作成されたことを確認
            assert vector_store.collection == new_collection

    def test_get_document_count(self, monkeypatch, tmp_path):
        """get_document_count()で正しいカウントが返される（モック）"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()

            # 初期化時: 42個、get_document_count呼び出し時: 42個
            mock_collection.count.side_effect = [42, 42]

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config)
            vector_store.initialize()

            # ドキュメント数を取得
            count = vector_store.get_document_count()

            # 結果の検証
            assert count == 42

            # collection.count()が呼ばれたことを確認
            assert mock_collection.count.call_count == 2

    def test_get_collection_info(self, monkeypatch, tmp_path):
        """get_collection_info()でコレクション情報が取得できる（モック）"""
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
            mock_client = Mock()
            mock_collection = Mock()

            # count()の呼び出しパターン
            # 1回目: 初期化時
            # 2回目: list_documents()内のcount()
            # 3回目: get_collection_info()内のcount()
            mock_collection.count.side_effect = [15, 15, 15]
            mock_collection.metadata = {
                "description": "RAG application document store"
            }

            # list_documents用のモックデータ
            mock_collection.get.return_value = {
                'ids': ['doc1_chunk_0', 'doc1_chunk_1', 'doc2_chunk_0'],
                'documents': ['Content1', 'Content2', 'Content3'],
                'metadatas': [
                    {
                        'document_id': 'doc1',
                        'document_name': 'test1.txt',
                        'source': '/path/to/test1.txt',
                        'doc_type': 'txt',
                        'size': 100
                    },
                    {
                        'document_id': 'doc1',
                        'document_name': 'test1.txt',
                        'source': '/path/to/test1.txt',
                        'doc_type': 'txt',
                        'size': 120
                    },
                    {
                        'document_id': 'doc2',
                        'document_name': 'test2.md',
                        'source': '/path/to/test2.md',
                        'doc_type': 'md',
                        'size': 80
                    }
                ]
            }

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # VectorStoreの作成と初期化
            vector_store = VectorStore(config=config, collection_name="my_collection")
            vector_store.initialize()

            # コレクション情報を取得
            info = vector_store.get_collection_info()

            # 結果の検証
            assert info['collection_name'] == 'my_collection'
            assert info['total_chunks'] == 15
            assert info['unique_documents'] == 2
            assert str(chroma_dir) in info['persist_directory']
            assert info['metadata'] == {"description": "RAG application document store"}

    def test_context_manager(self, monkeypatch, tmp_path):
        """`with`文で初期化・クローズが自動実行される"""
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

        # ChromaDBのモック
        with patch("src.rag.vector_store.chromadb.PersistentClient") as mock_client_class:
            mock_client = Mock()
            mock_collection = Mock()
            mock_collection.count.return_value = 5

            mock_client.get_or_create_collection.return_value = mock_collection
            mock_client_class.return_value = mock_client

            # コンテキストマネージャーとして使用
            vector_store = VectorStore(config=config)

            # with文の前はNone
            assert vector_store.collection is None
            assert vector_store.client is None

            with vector_store as vs:
                # with文の中では初期化されている
                assert vs.collection is not None
                assert vs.client is not None
                assert vs.collection == mock_collection
                assert vs.client == mock_client

            # with文の外では閉じられている
            assert vector_store.collection is None
            assert vector_store.client is None
