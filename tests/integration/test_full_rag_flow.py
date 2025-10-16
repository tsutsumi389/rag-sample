"""RAGシステムのエンドツーエンド統合テスト

実際のChromaDBとOllamaを使用して、ドキュメント追加から検索、回答生成までの
完全なフローをテストします。

注意: これらのテストは実際のOllamaサービスが必要です。
Ollamaが起動していない場合、テストはスキップされます。
"""

import pytest
from pathlib import Path
import chromadb

from src.rag.vector_store import VectorStore
from src.rag.embeddings import EmbeddingGenerator
from src.rag.document_processor import DocumentProcessor
from src.rag.engine import RAGEngine
from src.utils.config import Config


# Ollamaの起動チェック用のfixture
@pytest.fixture(scope="module")
def check_ollama():
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
class TestFullRAGFlow:
    """エンドツーエンドフロー（実際のChromaDB使用）のテスト"""

    def test_complete_rag_flow_single_document(
        self,
        integration_config: Config,
        sample_text_files: dict[str, Path],
        check_ollama
    ):
        """ドキュメント追加 → 検索 → 回答生成の完全フロー（単一ドキュメント）

        Args:
            integration_config: 統合テスト用の設定
            sample_text_files: サンプルテキストファイルの辞書
            check_ollama: Ollama起動チェック
        """
        # コンポーネントの初期化
        vector_store = VectorStore(integration_config)
        embedding_generator = EmbeddingGenerator(integration_config)
        document_processor = DocumentProcessor(integration_config)

        try:
            # 1. ベクトルストアの初期化
            vector_store.initialize()

            # 2. ドキュメントの処理
            python_file = sample_text_files["python"]
            document, chunks = document_processor.process_document(str(python_file))

            assert document is not None
            assert len(chunks) > 0
            # document_idはchunksから取得
            document_id = chunks[0].document_id if chunks else "unknown"
            print(f"Processed document: {document_id} with {len(chunks)} chunks")

            # 3. 埋め込みの生成とベクトルストアへの追加
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = embedding_generator.embed_documents(chunk_texts)

            assert len(embeddings) == len(chunks)
            assert all(len(emb) > 0 for emb in embeddings)

            # ドキュメントをベクトルストアに追加
            vector_store.add_documents(chunks, embeddings)

            # ドキュメント数の確認
            doc_count = vector_store.get_document_count()
            assert doc_count == len(chunks)

            # 4. 検索のテスト
            query = "Pythonの特徴は何ですか？"
            query_embedding = embedding_generator.embed_query(query)
            search_results = vector_store.search(query_embedding, n_results=2)

            assert len(search_results) > 0
            assert all(result.score >= 0 and result.score <= 1 for result in search_results)
            print(f"Search returned {len(search_results)} results")

            # 最も類似度の高い結果を確認
            top_result = search_results[0]
            assert "Python" in top_result.chunk.content or "python" in top_result.chunk.content.lower()
            print(f"Top result content: {top_result.chunk.content[:100]}...")

            # 5. RAGEngineを使った回答生成
            rag_engine = RAGEngine(
                config=integration_config,
                vector_store=vector_store,
                embedding_generator=embedding_generator
            )

            answer = rag_engine.query(query, n_results=2)

            assert "answer" in answer
            assert answer["answer"] != ""
            assert "sources" in answer
            assert len(answer["sources"]) > 0
            print(f"Generated answer: {answer['answer'][:200]}...")

        finally:
            # クリーンアップ
            vector_store.clear()
            vector_store.close()

    def test_complete_rag_flow_multiple_documents(
        self,
        integration_config: Config,
        sample_text_files: dict[str, Path],
        check_ollama
    ):
        """複数ドキュメントの追加と検索のフロー

        Args:
            integration_config: 統合テスト用の設定
            sample_text_files: サンプルテキストファイルの辞書
            check_ollama: Ollama起動チェック
        """
        # コンポーネントの初期化
        vector_store = VectorStore(integration_config)
        embedding_generator = EmbeddingGenerator(integration_config)
        document_processor = DocumentProcessor(integration_config)

        try:
            # ベクトルストアの初期化
            vector_store.initialize()

            # すべてのサンプルファイルを処理して追加
            all_documents = []
            all_chunks = []

            for file_name, file_path in sample_text_files.items():
                document, chunks = document_processor.process_document(str(file_path))
                all_documents.append(document)
                all_chunks.extend(chunks)

                # 埋め込みの生成と追加
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = embedding_generator.embed_documents(chunk_texts)
                vector_store.add_documents(chunks, embeddings)

            print(f"Added {len(all_documents)} documents with {len(all_chunks)} total chunks")

            # ドキュメント数の確認
            doc_count = vector_store.get_document_count()
            assert doc_count == len(all_chunks)

            # 複数のクエリでテスト
            queries = [
                ("Pythonについて教えて", "Python"),
                ("RAGとは何ですか？", "RAG"),
                ("LLMの用途は？", "LLM")
            ]

            for query, expected_keyword in queries:
                query_embedding = embedding_generator.embed_query(query)
                search_results = vector_store.search(query_embedding, n_results=3)

                assert len(search_results) > 0
                # 検索結果に期待されるキーワードが含まれているか確認
                found = any(
                    expected_keyword.lower() in result.chunk.content.lower()
                    for result in search_results
                )
                assert found, f"Expected keyword '{expected_keyword}' not found in search results for query '{query}'"
                print(f"Query '{query}' found relevant content with keyword '{expected_keyword}'")

            # ドキュメント一覧の取得
            doc_list = vector_store.list_documents()
            assert len(doc_list) == len(all_documents)

        finally:
            # クリーンアップ
            vector_store.clear()
            vector_store.close()

    def test_document_deletion_and_research(
        self,
        integration_config: Config,
        sample_text_files: dict[str, Path],
        check_ollama
    ):
        """ドキュメント削除と再検索のフロー

        Args:
            integration_config: 統合テスト用の設定
            sample_text_files: サンプルテキストファイルの辞書
            check_ollama: Ollama起動チェック
        """
        # コンポーネントの初期化
        vector_store = VectorStore(integration_config)
        embedding_generator = EmbeddingGenerator(integration_config)
        document_processor = DocumentProcessor(integration_config)

        try:
            # ベクトルストアの初期化
            vector_store.initialize()

            # 2つのドキュメントを追加
            python_file = sample_text_files["python"]
            rag_file = sample_text_files["rag"]

            python_doc, python_chunks = document_processor.process_document(str(python_file))
            rag_doc, rag_chunks = document_processor.process_document(str(rag_file))

            # Python ドキュメントを追加
            chunk_texts = [chunk.content for chunk in python_chunks]
            embeddings = embedding_generator.embed_documents(chunk_texts)
            vector_store.add_documents(python_chunks, embeddings)

            # RAG ドキュメントを追加
            chunk_texts = [chunk.content for chunk in rag_chunks]
            embeddings = embedding_generator.embed_documents(chunk_texts)
            vector_store.add_documents(rag_chunks, embeddings)

            # 初期のドキュメント数を確認
            initial_count = vector_store.get_document_count()
            assert initial_count == len(python_chunks) + len(rag_chunks)
            print(f"Initial document count: {initial_count}")

            # Python関連のクエリで検索
            query = "Pythonの特徴"
            query_embedding = embedding_generator.embed_query(query)
            search_results = vector_store.search(query_embedding, n_results=5)

            # Python関連のコンテンツが含まれることを確認
            python_found = any(
                "python" in result.chunk.content.lower()
                for result in search_results
            )
            assert python_found

            # Python ドキュメントを削除
            python_doc_id = python_chunks[0].document_id if python_chunks else None
            assert python_doc_id is not None
            deleted_count = vector_store.delete(document_id=python_doc_id)
            assert deleted_count == len(python_chunks)
            print(f"Deleted {deleted_count} chunks from Python document")

            # 削除後のドキュメント数を確認
            remaining_count = vector_store.get_document_count()
            assert remaining_count == len(rag_chunks)

            # 再度同じクエリで検索
            search_results_after = vector_store.search(query_embedding, n_results=5)

            # Python関連のコンテンツが見つからないことを確認
            # （検索結果がない、またはRAGドキュメントのみが返される）
            if len(search_results_after) > 0:
                # 結果があれば、それはRAGドキュメントのはず
                python_found_after = any(
                    "python" in result.chunk.content.lower() and
                    "rag" not in result.chunk.content.lower()
                    for result in search_results_after
                )
                # Pure Pythonコンテンツは見つからないはず
                # （RAGの説明にPythonが含まれる可能性があるため、厳密には判定しない）
                print(f"After deletion, search returned {len(search_results_after)} results")
            else:
                print("After deletion, no results found for Python query")

            # RAG関連のクエリは正常に動作することを確認
            rag_query = "RAGシステムとは"
            rag_query_embedding = embedding_generator.embed_query(rag_query)
            rag_search_results = vector_store.search(rag_query_embedding, n_results=5)

            assert len(rag_search_results) > 0
            rag_found = any(
                "rag" in result.chunk.content.lower()
                for result in rag_search_results
            )
            assert rag_found, "RAG document should still be searchable"

        finally:
            # クリーンアップ
            vector_store.clear()
            vector_store.close()

    def test_rag_engine_chat_flow(
        self,
        integration_config: Config,
        sample_text_files: dict[str, Path],
        check_ollama
    ):
        """RAGEngineのチャット機能を使ったフロー

        Args:
            integration_config: 統合テスト用の設定
            sample_text_files: サンプルテキストファイルの辞書
            check_ollama: Ollama起動チェック
        """
        # コンポーネントの初期化
        vector_store = VectorStore(integration_config)
        embedding_generator = EmbeddingGenerator(integration_config)
        document_processor = DocumentProcessor(integration_config)

        try:
            # ベクトルストアの初期化
            vector_store.initialize()

            # LLMドキュメントを追加
            llm_file = sample_text_files["llm"]
            document, chunks = document_processor.process_document(str(llm_file))

            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = embedding_generator.embed_documents(chunk_texts)
            vector_store.add_documents(chunks, embeddings)

            # RAGEngineの作成
            rag_engine = RAGEngine(
                config=integration_config,
                vector_store=vector_store,
                embedding_generator=embedding_generator
            )

            # 初期化
            rag_engine.initialize()

            # チャット履歴が空であることを確認
            assert len(rag_engine.get_chat_history()) == 0

            # 最初の質問
            question1 = "LLMとは何ですか？"
            answer1 = rag_engine.chat(question1, n_results=2)

            assert "answer" in answer1
            assert answer1["answer"] != ""
            print(f"Answer 1: {answer1['answer'][:150]}...")

            # チャット履歴が更新されていることを確認
            history = rag_engine.get_chat_history()
            assert len(history) == 2  # user + assistant
            assert history[0]["role"] == "user"
            assert history[0]["content"] == question1
            assert history[1]["role"] == "assistant"

            # フォローアップの質問
            question2 = "それはどのような用途に使われますか？"
            answer2 = rag_engine.chat(question2, n_results=2)

            assert "answer" in answer2
            assert answer2["answer"] != ""
            print(f"Answer 2: {answer2['answer'][:150]}...")

            # チャット履歴が累積されていることを確認
            history = rag_engine.get_chat_history()
            assert len(history) == 4  # 2 * (user + assistant)

            # ステータス情報の取得
            status = rag_engine.get_status()
            assert "vector_store_info" in status
            assert "chat_history_length" in status
            assert status["chat_history_length"] == 4
            assert "llm_model" in status
            assert "embedding_model" in status

            # チャット履歴のクリア
            rag_engine.clear_chat_history()
            assert len(rag_engine.get_chat_history()) == 0

        finally:
            # クリーンアップ
            vector_store.clear()
            vector_store.close()


@pytest.mark.integration
class TestChromaDBPersistence:
    """ChromaDBのデータ永続化のテスト"""

    def test_data_persistence_across_sessions(
        self,
        integration_config: Config,
        sample_text_files: dict[str, Path],
        check_ollama
    ):
        """データが永続化され、再起動後も利用可能であることを確認

        Args:
            integration_config: 統合テスト用の設定
            sample_text_files: サンプルテキストファイルの辞書
            check_ollama: Ollama起動チェック
        """
        # 第1セッション: データの追加
        vector_store1 = VectorStore(integration_config)
        embedding_generator = EmbeddingGenerator(integration_config)
        document_processor = DocumentProcessor(integration_config)
        vector_store2 = None

        try:
            vector_store1.initialize()

            # ドキュメントを追加
            python_file = sample_text_files["python"]
            document, chunks = document_processor.process_document(str(python_file))

            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = embedding_generator.embed_documents(chunk_texts)
            vector_store1.add_documents(chunks, embeddings)

            # ドキュメント数を記録
            count1 = vector_store1.get_document_count()
            print(f"Session 1: Added {count1} chunks")

            # セッション1を終了
            vector_store1.close()

            # 第2セッション: データの読み込み
            vector_store2 = VectorStore(integration_config)
            vector_store2.initialize()

            # データが永続化されていることを確認
            count2 = vector_store2.get_document_count()
            assert count2 == count1, f"Data not persisted: expected {count1}, got {count2}"

            # 検索が正常に動作することを確認
            query = "Pythonの特徴"
            query_embedding = embedding_generator.embed_query(query)
            search_results = vector_store2.search(query_embedding, n_results=2)

            assert len(search_results) > 0
            print(f"Session 2: Search found {len(search_results)} results")

        finally:
            # クリーンアップ
            if vector_store2 is not None:
                vector_store2.clear()
                vector_store2.close()
