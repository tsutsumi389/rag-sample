"""Ollama統合テスト

実際のOllamaサービスを使用して、埋め込み生成とLLM回答生成の
機能をテストします。

注意: これらのテストは実際のOllamaサービスが必要です。
Ollamaが起動していない場合、テストはスキップされます。
"""

import pytest
import requests
from unittest.mock import patch

from src.rag.embeddings import EmbeddingGenerator, EmbeddingError
from src.rag.engine import RAGEngine, RAGEngineError
from src.utils.config import Config


# Ollamaの起動チェック用のfixture
@pytest.fixture(scope="module")
def check_ollama_service():
    """Ollamaサービスが起動しているかチェックする

    Yields:
        bool: Ollamaが起動している場合True
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            yield True
        else:
            pytest.skip("Ollama service is not responding correctly")
    except Exception as e:
        pytest.skip(f"Ollama service is not available: {e}")


@pytest.mark.integration
class TestOllamaEmbeddings:
    """Ollama埋め込み生成の統合テスト"""

    def test_embedding_generation_single_text(
        self,
        integration_config: Config,
        check_ollama_service
    ):
        """埋め込み生成の実行（単一テキスト）

        Args:
            integration_config: 統合テスト用の設定
            check_ollama_service: Ollama起動チェック
        """
        # EmbeddingGeneratorの初期化
        generator = EmbeddingGenerator(integration_config)

        # 単一テキストの埋め込み生成
        text = "これはテスト用のテキストです。"
        embedding = generator.embed_query(text)

        # 埋め込みベクトルの検証
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(val, float) for val in embedding)

        # 埋め込みの次元数を確認
        dimension = generator.get_embedding_dimension()
        assert len(embedding) == dimension
        print(f"Embedding dimension: {dimension}")

    def test_embedding_generation_multiple_texts(
        self,
        integration_config: Config,
        check_ollama_service
    ):
        """埋め込み生成の実行（複数テキスト）

        Args:
            integration_config: 統合テスト用の設定
            check_ollama_service: Ollama起動チェック
        """
        # EmbeddingGeneratorの初期化
        generator = EmbeddingGenerator(integration_config)

        # 複数テキストの埋め込み生成
        texts = [
            "Pythonは人気のあるプログラミング言語です。",
            "機械学習とデータサイエンスに広く使用されています。",
            "シンプルで読みやすい構文が特徴です。"
        ]
        embeddings = generator.embed_documents(texts)

        # 埋め込みベクトルのリストの検証
        assert embeddings is not None
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)

        # 各埋め込みベクトルの検証
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) > 0
            assert all(isinstance(val, float) for val in embedding)

        # すべての埋め込みが同じ次元であることを確認
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1, "All embeddings should have the same dimension"
        print(f"Generated {len(embeddings)} embeddings with dimension {dimensions[0]}")

    def test_embedding_similarity(
        self,
        integration_config: Config,
        check_ollama_service
    ):
        """意味的に類似したテキストの埋め込みが近いことを確認

        Args:
            integration_config: 統合テスト用の設定
            check_ollama_service: Ollama起動チェック
        """
        # EmbeddingGeneratorの初期化
        generator = EmbeddingGenerator(integration_config)

        # 類似したテキストと異なるテキスト
        similar_text1 = "犬は忠実なペットです。"
        similar_text2 = "犬は人間の良い友達です。"
        different_text = "Pythonはプログラミング言語です。"

        # 埋め込み生成
        emb1 = generator.embed_query(similar_text1)
        emb2 = generator.embed_query(similar_text2)
        emb3 = generator.embed_query(different_text)

        # コサイン類似度を計算
        def cosine_similarity(vec1, vec2):
            import math
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = math.sqrt(sum(a * a for a in vec1))
            magnitude2 = math.sqrt(sum(b * b for b in vec2))
            return dot_product / (magnitude1 * magnitude2)

        similarity_similar = cosine_similarity(emb1, emb2)
        similarity_different1 = cosine_similarity(emb1, emb3)
        similarity_different2 = cosine_similarity(emb2, emb3)

        print(f"Similarity (similar texts): {similarity_similar:.4f}")
        print(f"Similarity (different texts 1): {similarity_different1:.4f}")
        print(f"Similarity (different texts 2): {similarity_different2:.4f}")

        # 類似したテキストの方が高い類似度を持つことを確認
        assert similarity_similar > similarity_different1
        assert similarity_similar > similarity_different2

    def test_embedding_error_on_empty_input(
        self,
        integration_config: Config,
        check_ollama_service
    ):
        """空の入力でエラーが発生することを確認

        Args:
            integration_config: 統合テスト用の設定
            check_ollama_service: Ollama起動チェック
        """
        generator = EmbeddingGenerator(integration_config)

        # 空文字列でエラーが発生することを確認
        with pytest.raises(ValueError, match="cannot be empty"):
            generator.embed_query("")

        # 空リストでエラーが発生することを確認
        with pytest.raises(ValueError, match="cannot be empty"):
            generator.embed_documents([])

        # 空文字列を含むリストでエラーが発生することを確認
        with pytest.raises(ValueError, match="cannot contain empty"):
            generator.embed_documents(["valid text", "", "another text"])


@pytest.mark.integration
class TestOllamaLLM:
    """Ollama LLM回答生成の統合テスト"""

    def test_llm_answer_generation(
        self,
        integration_config: Config,
        check_ollama_service
    ):
        """LLMによる回答生成の実行

        Args:
            integration_config: 統合テスト用の設定
            check_ollama_service: Ollama起動チェック
        """
        from src.rag.vector_store import BaseVectorStore, create_vector_store
        from src.rag.embeddings import EmbeddingGenerator
        from src.models.document import SearchResult, Chunk

        # コンポーネントの初期化
        vector_store = VectorStore(integration_config)
        embedding_generator = EmbeddingGenerator(integration_config)

        # RAGEngineの作成
        rag_engine = RAGEngine(
            config=integration_config,
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )

        # コンテキストと質問を用意
        # SearchResultオブジェクトのリストを作成
        context_results = [
            SearchResult(
                chunk=Chunk(
                    content="Pythonは1991年にGuido van Rossumによって開発されました。",
                    chunk_id="test_chunk_001",
                    document_id="test_doc_001",
                    chunk_index=0,
                    start_char=0,
                    end_char=50,
                    metadata={}
                ),
                score=0.9,
                document_name="python_history.txt",
                document_source="test_source_1",
                rank=1,
                metadata={}
            ),
            SearchResult(
                chunk=Chunk(
                    content="Pythonはシンプルで読みやすい構文が特徴です。",
                    chunk_id="test_chunk_002",
                    document_id="test_doc_001",
                    chunk_index=1,
                    start_char=50,
                    end_char=100,
                    metadata={}
                ),
                score=0.85,
                document_name="python_features.txt",
                document_source="test_source_2",
                rank=2,
                metadata={}
            )
        ]
        question = "Pythonの特徴は何ですか？"

        # 回答生成
        answer = rag_engine.generate_answer(
            question=question,
            context_results=context_results,
            include_sources=True
        )

        # 回答の検証
        assert "answer" in answer
        assert answer["answer"] != ""
        assert isinstance(answer["answer"], str)
        assert len(answer["answer"]) > 0

        # 情報源が含まれていることを確認
        assert "sources" in answer
        assert len(answer["sources"]) > 0

        print(f"Generated answer: {answer['answer'][:200]}...")
        print(f"Number of sources: {len(answer['sources'])}")

    def test_llm_answer_with_empty_context(
        self,
        integration_config: Config,
        check_ollama_service
    ):
        """空のコンテキストでも回答が生成されることを確認

        Args:
            integration_config: 統合テスト用の設定
            check_ollama_service: Ollama起動チェック
        """
        from src.rag.vector_store import BaseVectorStore, create_vector_store
        from src.rag.embeddings import EmbeddingGenerator

        # コンポーネントの初期化
        vector_store = VectorStore(integration_config)
        embedding_generator = EmbeddingGenerator(integration_config)

        # RAGEngineの作成
        rag_engine = RAGEngine(
            config=integration_config,
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )

        # 空のコンテキストで回答生成
        question = "こんにちは、調子はどうですか？"
        answer = rag_engine.generate_answer(
            question=question,
            context_results=[],
            include_sources=False
        )

        # 回答が生成されることを確認
        assert "answer" in answer
        assert answer["answer"] != ""
        assert isinstance(answer["answer"], str)

        # include_sources=Falseなので、sourcesキーは含まれない
        assert "sources" not in answer

        print(f"Answer without context: {answer['answer'][:150]}...")

    def test_llm_answer_with_custom_template(
        self,
        integration_config: Config,
        check_ollama_service
    ):
        """カスタムプロンプトテンプレートの使用

        Args:
            integration_config: 統合テスト用の設定
            check_ollama_service: Ollama起動チェック
        """
        from src.rag.vector_store import BaseVectorStore, create_vector_store
        from src.rag.embeddings import EmbeddingGenerator
        from src.models.document import SearchResult, Chunk

        # コンポーネントの初期化
        vector_store = VectorStore(integration_config)
        embedding_generator = EmbeddingGenerator(integration_config)

        # RAGEngineの作成
        rag_engine = RAGEngine(
            config=integration_config,
            vector_store=vector_store,
            embedding_generator=embedding_generator
        )

        # カスタムテンプレートを使用
        custom_template = """以下の情報に基づいて、質問に簡潔に答えてください。

情報:
{context}

質問: {question}

回答（50文字以内）:"""

        # SearchResultオブジェクトを作成
        context_results = [
            SearchResult(
                chunk=Chunk(
                    content="Pythonは1991年に開発されました。",
                    chunk_id="test_chunk_001",
                    document_id="test_doc_001",
                    chunk_index=0,
                    start_char=0,
                    end_char=30,
                    metadata={}
                ),
                score=0.9,
                document_name="python_history.txt",
                document_source="test_source",
                rank=1,
                metadata={}
            )
        ]
        question = "Pythonはいつ開発されましたか？"

        # カスタムテンプレートで回答生成
        answer = rag_engine.generate_answer(
            question=question,
            context_results=context_results,
            qa_template=custom_template,
            include_sources=False
        )

        # 回答が生成されることを確認
        assert "answer" in answer
        assert answer["answer"] != ""
        print(f"Answer with custom template: {answer['answer']}")


@pytest.mark.integration
class TestOllamaErrorHandling:
    """Ollama未起動時のエラーハンドリングのテスト"""

    def test_embedding_error_when_ollama_not_running(
        self,
        integration_config: Config
    ):
        """Ollama未起動時に埋め込み生成がエラーになることを確認

        Args:
            integration_config: 統合テスト用の設定
        """
        # OllamaのベースURLを存在しないアドレスに変更
        with patch.object(integration_config, 'ollama_base_url', 'http://non-existent-host:11434'):
            generator = EmbeddingGenerator(integration_config)

            # 埋め込み生成がエラーになることを確認
            with pytest.raises(EmbeddingError):
                generator.embed_query("test text")

    def test_llm_error_when_ollama_not_running(
        self,
        integration_config: Config
    ):
        """Ollama未起動時にLLM回答生成がエラーになることを確認

        Args:
            integration_config: 統合テスト用の設定
        """
        from src.rag.vector_store import BaseVectorStore, create_vector_store
        from src.rag.embeddings import EmbeddingGenerator
        from src.models.document import SearchResult, Chunk

        # OllamaのベースURLを存在しないアドレスに変更
        with patch.object(integration_config, 'ollama_base_url', 'http://non-existent-host:11434'):
            vector_store = VectorStore(integration_config)
            embedding_generator = EmbeddingGenerator(integration_config)

            # RAGEngineの作成
            rag_engine = RAGEngine(
                config=integration_config,
                vector_store=vector_store,
                embedding_generator=embedding_generator
            )

            # SearchResultオブジェクトを作成
            context_results = [
                SearchResult(
                    chunk=Chunk(
                        content="test context",
                        chunk_id="test_chunk_001",
                        document_id="test_doc_001",
                        chunk_index=0,
                        start_char=0,
                        end_char=12,
                        metadata={}
                    ),
                    score=0.9,
                    document_name="test.txt",
                    document_source="test_source",
                    rank=1,
                    metadata={}
                )
            ]

            # LLM回答生成がエラーになることを確認
            with pytest.raises(RAGEngineError):
                rag_engine.generate_answer(
                    question="test question",
                    context_results=context_results
                )

    def test_embedding_with_invalid_model(
        self,
        integration_config: Config,
        check_ollama_service
    ):
        """存在しないモデルでエラーが発生することを確認

        Args:
            integration_config: 統合テスト用の設定
            check_ollama_service: Ollama起動チェック
        """
        # 存在しないモデル名に変更
        with patch.object(integration_config, 'ollama_embedding_model', 'non-existent-model'):
            generator = EmbeddingGenerator(integration_config)

            # 埋め込み生成がエラーになることを確認
            with pytest.raises(EmbeddingError):
                generator.embed_query("test text")
