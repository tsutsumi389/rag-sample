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
from src.models.document import SearchResult, Chunk


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


class TestRAGEngineRetrieve:
    """RAGEngine - 検索のテスト"""

    def test_retrieve_returns_search_results(self):
        """retrieve()で正しいSearchResultリストが返される（モック）"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()

        # 検索結果のモックデータ
        mock_chunk = Chunk(
            content="これはテストドキュメントです。",
            chunk_id="chunk_001",
            document_id="doc_001",
            chunk_index=0,
            start_char=0,
            end_char=16,
            metadata={"source": "test.txt"}
        )
        mock_search_results = [
            SearchResult(
                chunk=mock_chunk,
                score=0.95,
                document_name="test.txt",
                document_source="/path/to/test.txt",
                rank=1
            )
        ]

        # 埋め込みベクトルのモック
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding

        # ベクトルストアの検索結果をモック
        mock_vector_store.search.return_value = mock_search_results

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM"):

            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # 検索実行
            query = "テストクエリ"
            results = engine.retrieve(query)

            # 結果の検証
            assert results == mock_search_results
            assert len(results) == 1
            assert results[0].score == 0.95
            assert results[0].document_name == "test.txt"

            # embed_queryが正しく呼ばれたことを確認
            mock_embedding_generator.embed_query.assert_called_once_with(query)

            # vector_store.searchが正しいパラメータで呼ばれたことを確認
            mock_vector_store.search.assert_called_once_with(
                query_embedding=mock_query_embedding,
                n_results=5,
                where=None
            )

    def test_retrieve_with_empty_query_raises_error(self):
        """空クエリでRAGEngineErrorがraise"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM"):

            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # 空のクエリでエラーが発生することを確認
            with pytest.raises(RAGEngineError) as exc_info:
                engine.retrieve("")

            assert "検索クエリが空です" in str(exc_info.value)

            # 空白のみのクエリでもエラーが発生することを確認
            with pytest.raises(RAGEngineError) as exc_info:
                engine.retrieve("   ")

            assert "検索クエリが空です" in str(exc_info.value)

            # embed_queryとsearchが呼ばれていないことを確認
            mock_embedding_generator.embed_query.assert_not_called()
            mock_vector_store.search.assert_not_called()

    def test_retrieve_passes_n_results_and_where_parameters(self):
        """n_results/whereパラメータが正しく渡される（モック）"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()

        # 検索結果のモック
        mock_search_results = []
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding
        mock_vector_store.search.return_value = mock_search_results

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM"):

            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # カスタムパラメータで検索実行
            query = "テストクエリ"
            n_results = 10
            where = {"document_id": "doc123"}

            results = engine.retrieve(query, n_results=n_results, where=where)

            # 結果の検証
            assert results == mock_search_results

            # vector_store.searchがカスタムパラメータで呼ばれたことを確認
            mock_vector_store.search.assert_called_once_with(
                query_embedding=mock_query_embedding,
                n_results=n_results,
                where=where
            )

    def test_retrieve_handles_embedding_error(self):
        """埋め込み生成エラー時にRAGEngineErrorがraise"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()

        # embed_queryで例外を発生させる
        mock_embedding_generator.embed_query.side_effect = Exception("Embedding failed")

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM"):

            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # エラーが発生することを確認
            with pytest.raises(RAGEngineError) as exc_info:
                engine.retrieve("テストクエリ")

            error_message = str(exc_info.value)
            assert "ドキュメントの検索に失敗しました" in error_message
            assert "Embedding failed" in error_message

            # vector_store.searchが呼ばれていないことを確認
            mock_vector_store.search.assert_not_called()

    def test_retrieve_handles_vector_store_error(self):
        """ベクトルストア検索エラー時にRAGEngineErrorがraise"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()

        # 埋め込みは成功するが、検索で失敗
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding
        mock_vector_store.search.side_effect = Exception("Vector store search failed")

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM"):

            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # エラーが発生することを確認
            with pytest.raises(RAGEngineError) as exc_info:
                engine.retrieve("テストクエリ")

            error_message = str(exc_info.value)
            assert "ドキュメントの検索に失敗しました" in error_message
            assert "Vector store search failed" in error_message


class TestRAGEngineGenerateAnswer:
    """RAGEngine - 回答生成のテスト"""

    def test_generate_answer_returns_answer_dict(self):
        """generate_answer()で正しい回答辞書が返される（モック）"""
        # モックの準備
        mock_llm = Mock()
        mock_llm.invoke.return_value = "これはテスト回答です。"

        # 検索結果のモックデータ
        mock_chunk = Chunk(
            content="Pythonは高レベルプログラミング言語です。",
            chunk_id="chunk_001",
            document_id="doc_001",
            chunk_index=0,
            start_char=0,
            end_char=26,
            metadata={"source": "python.txt"}
        )
        context_results = [
            SearchResult(
                chunk=mock_chunk,
                score=0.95,
                document_name="python.txt",
                document_source="/path/to/python.txt",
                rank=1
            )
        ]

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine()

            # 回答生成
            question = "Pythonとは何ですか？"
            result = engine.generate_answer(question, context_results)

            # 結果の検証
            assert result["answer"] == "これはテスト回答です。"
            assert result["context_count"] == 1
            assert "sources" in result
            assert len(result["sources"]) == 1
            assert result["sources"][0]["name"] == "python.txt"
            assert result["sources"][0]["source"] == "/path/to/python.txt"
            assert result["sources"][0]["score"] == 0.95

            # LLMが呼ばれたことを確認
            mock_llm.invoke.assert_called_once()
            call_args = mock_llm.invoke.call_args[0][0]
            assert "Pythonは高レベルプログラミング言語です。" in call_args
            assert "Pythonとは何ですか？" in call_args

    def test_generate_answer_with_empty_question_raises_error(self):
        """空の質問でRAGEngineErrorがraise"""
        # モックの準備
        mock_llm = Mock()

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine()

            # 空の質問でエラーが発生することを確認
            with pytest.raises(RAGEngineError) as exc_info:
                engine.generate_answer("", [])

            assert "質問が空です" in str(exc_info.value)

            # 空白のみの質問でもエラーが発生することを確認
            with pytest.raises(RAGEngineError) as exc_info:
                engine.generate_answer("   ", [])

            assert "質問が空です" in str(exc_info.value)

            # LLMが呼ばれていないことを確認
            mock_llm.invoke.assert_not_called()

    def test_generate_answer_with_empty_context(self):
        """コンテキストが空の場合の処理"""
        # モックの準備
        mock_llm = Mock()
        mock_llm.invoke.return_value = "提供された情報では回答できません。"

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine()

            # 空のコンテキストで回答生成
            question = "テスト質問"
            result = engine.generate_answer(question, [])

            # 結果の検証
            assert result["answer"] == "提供された情報では回答できません。"
            assert result["context_count"] == 0
            assert "sources" not in result  # 空のコンテキストなので情報源なし

            # LLMが呼ばれたことを確認
            mock_llm.invoke.assert_called_once()
            call_args = mock_llm.invoke.call_args[0][0]
            assert "関連する情報が見つかりませんでした。" in call_args
            assert "テスト質問" in call_args

    def test_generate_answer_with_include_sources_true(self):
        """include_sources=Trueで情報源が含まれる"""
        # モックの準備
        mock_llm = Mock()
        mock_llm.invoke.return_value = "テスト回答"

        # 複数の検索結果（同じドキュメントと異なるドキュメント）
        mock_chunk1 = Chunk(
            content="コンテンツ1",
            chunk_id="chunk_001",
            document_id="doc_001",
            chunk_index=0,
            start_char=0,
            end_char=6,
            metadata={"source": "doc1.txt"}
        )
        mock_chunk2 = Chunk(
            content="コンテンツ2",
            chunk_id="chunk_002",
            document_id="doc_001",
            chunk_index=1,
            start_char=6,
            end_char=12,
            metadata={"source": "doc1.txt"}
        )
        mock_chunk3 = Chunk(
            content="コンテンツ3",
            chunk_id="chunk_003",
            document_id="doc_002",
            chunk_index=0,
            start_char=0,
            end_char=6,
            metadata={"source": "doc2.txt"}
        )

        context_results = [
            SearchResult(
                chunk=mock_chunk1,
                score=0.95,
                document_name="doc1.txt",
                document_source="/path/to/doc1.txt",
                rank=1
            ),
            SearchResult(
                chunk=mock_chunk2,
                score=0.90,
                document_name="doc1.txt",
                document_source="/path/to/doc1.txt",  # 同じドキュメント
                rank=2
            ),
            SearchResult(
                chunk=mock_chunk3,
                score=0.85,
                document_name="doc2.txt",
                document_source="/path/to/doc2.txt",
                rank=3
            )
        ]

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine()

            # include_sources=Trueで回答生成
            result = engine.generate_answer("質問", context_results, include_sources=True)

            # 結果の検証
            assert "sources" in result
            assert len(result["sources"]) == 2  # 重複を除いて2つのドキュメント

            # ソースが正しく含まれていることを確認
            source_names = [s["name"] for s in result["sources"]]
            assert "doc1.txt" in source_names
            assert "doc2.txt" in source_names

    def test_generate_answer_with_include_sources_false(self):
        """include_sources=Falseで情報源が含まれない"""
        # モックの準備
        mock_llm = Mock()
        mock_llm.invoke.return_value = "テスト回答"

        mock_chunk = Chunk(
            content="テストコンテンツ",
            chunk_id="chunk_001",
            document_id="doc_001",
            chunk_index=0,
            start_char=0,
            end_char=8,
            metadata={"source": "test.txt"}
        )
        context_results = [
            SearchResult(
                chunk=mock_chunk,
                score=0.95,
                document_name="test.txt",
                document_source="/path/to/test.txt",
                rank=1
            )
        ]

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine()

            # include_sources=Falseで回答生成
            result = engine.generate_answer("質問", context_results, include_sources=False)

            # 結果の検証
            assert "sources" not in result
            assert result["answer"] == "テスト回答"
            assert result["context_count"] == 1

    def test_generate_answer_with_custom_template(self):
        """プロンプトテンプレートのカスタマイズが機能する"""
        # モックの準備
        mock_llm = Mock()
        mock_llm.invoke.return_value = "カスタム回答"

        mock_chunk = Chunk(
            content="テストコンテンツ",
            chunk_id="chunk_001",
            document_id="doc_001",
            chunk_index=0,
            start_char=0,
            end_char=8,
            metadata={"source": "test.txt"}
        )
        context_results = [
            SearchResult(
                chunk=mock_chunk,
                score=0.95,
                document_name="test.txt",
                document_source="/path/to/test.txt",
                rank=1
            )
        ]

        # カスタムテンプレート
        custom_template = """カスタムプロンプト:
コンテキスト: {context}
質問: {question}
回答してください。"""

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine()

            # カスタムテンプレートで回答生成
            result = engine.generate_answer(
                "テスト質問",
                context_results,
                qa_template=custom_template
            )

            # 結果の検証
            assert result["answer"] == "カスタム回答"

            # カスタムテンプレートが使用されたことを確認
            mock_llm.invoke.assert_called_once()
            call_args = mock_llm.invoke.call_args[0][0]
            assert "カスタムプロンプト:" in call_args
            assert "回答してください。" in call_args

    def test_generate_answer_handles_llm_error(self):
        """LLMエラー時にRAGEngineErrorがraise"""
        # モックの準備
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("LLM invocation failed")

        mock_chunk = Chunk(
            content="テストコンテンツ",
            chunk_id="chunk_001",
            document_id="doc_001",
            chunk_index=0,
            start_char=0,
            end_char=8,
            metadata={"source": "test.txt"}
        )
        context_results = [
            SearchResult(
                chunk=mock_chunk,
                score=0.95,
                document_name="test.txt",
                document_source="/path/to/test.txt",
                rank=1
            )
        ]

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine()

            # エラーが発生することを確認
            with pytest.raises(RAGEngineError) as exc_info:
                engine.generate_answer("質問", context_results)

            error_message = str(exc_info.value)
            assert "回答の生成に失敗しました" in error_message
            assert "LLM invocation failed" in error_message


class TestRAGEngineQuery:
    """RAGEngine - 統合クエリのテスト"""

    def test_query_executes_retrieve_and_generate_answer(self):
        """query()で検索と回答生成が一度に実行される（モック）"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()
        mock_llm = Mock()

        # 検索結果のモック
        mock_chunk = Chunk(
            content="Pythonは汎用プログラミング言語です。",
            chunk_id="chunk_001",
            document_id="doc_001",
            chunk_index=0,
            start_char=0,
            end_char=20,
            metadata={"source": "python.txt"}
        )
        mock_search_results = [
            SearchResult(
                chunk=mock_chunk,
                score=0.95,
                document_name="python.txt",
                document_source="/path/to/python.txt",
                rank=1
            )
        ]

        # 埋め込みベクトルのモック
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding
        mock_vector_store.search.return_value = mock_search_results

        # LLMの回答
        mock_llm.invoke.return_value = "Pythonは汎用プログラミング言語です。"

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # 統合クエリを実行
            question = "Pythonとは何ですか？"
            result = engine.query(question)

            # 結果の検証
            assert result["answer"] == "Pythonは汎用プログラミング言語です。"
            assert result["context_count"] == 1
            assert "sources" in result
            assert len(result["sources"]) == 1

            # retrieveが実行されたことを確認（embed_queryとsearchが呼ばれた）
            mock_embedding_generator.embed_query.assert_called_once_with(question)
            mock_vector_store.search.assert_called_once_with(
                query_embedding=mock_query_embedding,
                n_results=5,
                where=None
            )

            # generate_answerが実行されたことを確認（LLMが呼ばれた）
            mock_llm.invoke.assert_called_once()

    def test_query_passes_parameters_to_retrieve(self):
        """query()のパラメータがretrieve()に正しく渡される"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()
        mock_llm = Mock()

        # モックの設定
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding
        mock_vector_store.search.return_value = []
        mock_llm.invoke.return_value = "回答"

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # カスタムパラメータでクエリを実行
            question = "テスト質問"
            n_results = 10
            where = {"document_id": "doc123"}

            result = engine.query(
                question=question,
                n_results=n_results,
                where=where,
                include_sources=False
            )

            # retrieveにパラメータが渡されたことを確認
            mock_vector_store.search.assert_called_once_with(
                query_embedding=mock_query_embedding,
                n_results=n_results,
                where=where
            )

            # include_sources=Falseが機能していることを確認
            assert "sources" not in result

    def test_query_handles_retrieve_error(self):
        """query()でretrieve()がエラーを起こした場合の処理"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()
        mock_llm = Mock()

        # embed_queryでエラーを発生させる
        mock_embedding_generator.embed_query.side_effect = Exception("Embedding error")

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # エラーが発生することを確認
            with pytest.raises(RAGEngineError) as exc_info:
                engine.query("質問")

            # retrieveのエラーメッセージが含まれることを確認
            error_message = str(exc_info.value)
            assert "ドキュメントの検索に失敗しました" in error_message
            assert "Embedding error" in error_message

            # LLMが呼ばれていないことを確認（retrieveで失敗したため）
            mock_llm.invoke.assert_not_called()

    def test_query_handles_generate_answer_error(self):
        """query()でgenerate_answer()がエラーを起こした場合の処理"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()
        mock_llm = Mock()

        # retrieveは成功するが、generate_answerで失敗
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding
        mock_vector_store.search.return_value = []
        mock_llm.invoke.side_effect = Exception("LLM error")

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # エラーが発生することを確認
            with pytest.raises(RAGEngineError) as exc_info:
                engine.query("質問")

            # generate_answerのエラーメッセージが含まれることを確認
            error_message = str(exc_info.value)
            assert "回答の生成に失敗しました" in error_message
            assert "LLM error" in error_message

            # retrieveは実行されたことを確認
            mock_embedding_generator.embed_query.assert_called_once()
            mock_vector_store.search.assert_called_once()


class TestRAGEngineChat:
    """RAGEngine - チャット機能のテスト"""

    def test_chat_generates_chat_response(self):
        """chat()でチャット形式の回答が生成される（モック）"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()
        mock_llm = Mock()

        # 検索結果のモック
        mock_chunk = Chunk(
            content="Pythonは動的型付け言語です。",
            chunk_id="chunk_001",
            document_id="doc_001",
            chunk_index=0,
            start_char=0,
            end_char=16,
            metadata={"source": "python.txt"}
        )
        mock_search_results = [
            SearchResult(
                chunk=mock_chunk,
                score=0.95,
                document_name="python.txt",
                document_source="/path/to/python.txt",
                rank=1
            )
        ]

        # モックの設定
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding
        mock_vector_store.search.return_value = mock_search_results
        mock_llm.invoke.return_value = "Pythonは動的型付け言語です。"

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # チャット実行
            message = "Pythonの型について教えて"
            result = engine.chat(message)

            # 結果の検証
            assert result["answer"] == "Pythonは動的型付け言語です。"
            assert result["context_count"] == 1
            assert result["history_length"] == 2  # user + assistant
            assert "sources" in result
            assert len(result["sources"]) == 1

            # LLMが呼ばれたことを確認
            mock_llm.invoke.assert_called_once()

    def test_chat_adds_messages_to_history(self):
        """chat_historyにメッセージが追加される"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()
        mock_llm = Mock()

        # モックの設定
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding
        mock_vector_store.search.return_value = []
        mock_llm.invoke.return_value = "回答1"

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # 初期状態の確認
            assert len(engine.chat_history) == 0

            # 最初のチャット
            result1 = engine.chat("質問1")
            assert len(engine.chat_history) == 2  # user + assistant
            assert result1["history_length"] == 2

            # チャット履歴の内容を確認
            messages = engine.chat_history.messages
            assert messages[0].role == "user"
            assert messages[0].content == "質問1"
            assert messages[1].role == "assistant"
            assert messages[1].content == "回答1"

            # 2回目のチャット
            mock_llm.invoke.return_value = "回答2"
            result2 = engine.chat("質問2")
            assert len(engine.chat_history) == 4  # (user + assistant) * 2
            assert result2["history_length"] == 4

    def test_chat_includes_history_in_prompt(self):
        """履歴がプロンプトに含まれる"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()
        mock_llm = Mock()

        # モックの設定
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding
        mock_vector_store.search.return_value = []
        mock_llm.invoke.return_value = "回答"

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # 最初のチャット
            engine.chat("最初の質問")
            mock_llm.reset_mock()

            # 2回目のチャット
            engine.chat("次の質問")

            # プロンプトに履歴が含まれていることを確認
            mock_llm.invoke.assert_called_once()
            prompt = mock_llm.invoke.call_args[0][0]

            # 過去の会話が含まれていることを確認
            assert "過去の会話:" in prompt
            assert "user: 最初の質問" in prompt
            assert "assistant: 回答" in prompt
            assert "次の質問" in prompt

    def test_chat_respects_max_chat_history(self):
        """max_chat_historyによる履歴制限が機能する"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()
        mock_llm = Mock()

        # モックの設定
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding
        mock_vector_store.search.return_value = []
        mock_llm.invoke.return_value = "回答"

        # RAGEngineの初期化（max_chat_history=4: 2往復分）
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator,
                max_chat_history=4
            )

            # 3往復のチャットを実行（6メッセージ）
            engine.chat("質問1")
            engine.chat("質問2")
            engine.chat("質問3")

            # 履歴が最大4メッセージに制限されていることを確認
            assert len(engine.chat_history) == 4
            assert engine.chat_history.max_messages == 4

            # 古いメッセージが削除され、新しいメッセージが残っていることを確認
            messages = engine.chat_history.messages
            # 最新の2往復（質問2,回答2,質問3,回答3）が残っている
            assert messages[0].content == "質問2"
            assert messages[2].content == "質問3"

    def test_chat_with_empty_search_results(self):
        """検索結果が空の場合のチャット動作"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()
        mock_llm = Mock()

        # モックの設定（検索結果なし）
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding
        mock_vector_store.search.return_value = []
        mock_llm.invoke.return_value = "情報が見つかりませんでした。"

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # チャット実行
            result = engine.chat("質問")

            # 結果の検証
            assert result["context_count"] == 0
            assert "sources" not in result  # 検索結果がないのでsourcesなし

            # プロンプトに「関連する情報が見つかりませんでした」が含まれることを確認
            prompt = mock_llm.invoke.call_args[0][0]
            assert "関連する情報が見つかりませんでした。" in prompt

    def test_chat_handles_retrieve_error(self):
        """chat()でretrieve()がエラーを起こした場合の処理"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()
        mock_llm = Mock()

        # embed_queryでエラーを発生させる
        mock_embedding_generator.embed_query.side_effect = Exception("Search error")

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # エラーが発生することを確認
            with pytest.raises(RAGEngineError) as exc_info:
                engine.chat("質問")

            error_message = str(exc_info.value)
            assert "ドキュメントの検索に失敗しました" in error_message

            # ユーザーメッセージは履歴に追加されているが、アシスタント応答はない
            assert len(engine.chat_history) == 1
            assert engine.chat_history.messages[0].role == "user"

    def test_chat_handles_llm_error(self):
        """chat()でLLMがエラーを起こした場合の処理"""
        # モックの準備
        mock_vector_store = Mock()
        mock_embedding_generator = Mock()
        mock_llm = Mock()

        # retrieveは成功するが、LLMで失敗
        mock_query_embedding = [0.1, 0.2, 0.3]
        mock_embedding_generator.embed_query.return_value = mock_query_embedding
        mock_vector_store.search.return_value = []
        mock_llm.invoke.side_effect = Exception("LLM error")

        # RAGEngineの初期化
        with patch("src.rag.engine.VectorStore"), \
             patch("src.rag.engine.EmbeddingGenerator"), \
             patch("src.rag.engine.OllamaLLM") as mock_llm_cls:

            mock_llm_cls.return_value = mock_llm
            engine = RAGEngine(
                vector_store=mock_vector_store,
                embedding_generator=mock_embedding_generator
            )

            # エラーが発生することを確認
            with pytest.raises(RAGEngineError) as exc_info:
                engine.chat("質問")

            error_message = str(exc_info.value)
            assert "チャット回答の生成に失敗しました" in error_message
            assert "LLM error" in error_message

            # ユーザーメッセージは履歴に追加されているが、アシスタント応答はない
            assert len(engine.chat_history) == 1
            assert engine.chat_history.messages[0].role == "user"
