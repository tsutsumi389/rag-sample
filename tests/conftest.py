"""pytestの共通fixture定義

ユニットテストと統合テストで共有されるfixtureを定義します。
"""

import pytest
from pathlib import Path
from unittest.mock import Mock
import tempfile
import shutil

from src.utils.config import Config
from src.models.document import Document, Chunk


@pytest.fixture
def sample_config(tmp_path):
    """テスト用設定

    Args:
        tmp_path: pytestが提供する一時ディレクトリ

    Returns:
        Config: テスト用のConfig オブジェクト
    """
    # 一時的なChromaDBディレクトリを作成
    chroma_dir = tmp_path / "test_chroma_db"
    chroma_dir.mkdir(exist_ok=True)

    # 環境変数をオーバーライドするために、テスト用の.envファイルを作成
    env_file = tmp_path / "test.env"
    env_file.write_text(f"""
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CHROMA_PERSIST_DIRECTORY={chroma_dir}
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LOG_LEVEL=INFO
""")

    return Config(env_file=str(env_file))


@pytest.fixture
def sample_document():
    """テスト用Documentオブジェクト

    Returns:
        Document: サンプルのDocumentオブジェクト
    """
    return Document(
        file_path=Path("tests/fixtures/sample.txt"),
        content="これはテストドキュメントです。\nRAGシステムのテストに使用します。",
        document_id="test_doc_001",
        metadata={"source": "test", "type": "txt"}
    )


@pytest.fixture
def sample_chunks():
    """テスト用Chunkリスト

    Returns:
        list[Chunk]: サンプルのChunkオブジェクトのリスト
    """
    doc_id = "test_doc_001"
    return [
        Chunk(
            chunk_id=f"{doc_id}_chunk_0000",
            document_id=doc_id,
            content="これはテストドキュメントです。",
            start_char=0,
            end_char=16,
            metadata={"source": "test", "chunk_index": 0}
        ),
        Chunk(
            chunk_id=f"{doc_id}_chunk_0001",
            document_id=doc_id,
            content="RAGシステムのテストに使用します。",
            start_char=17,
            end_char=40,
            metadata={"source": "test", "chunk_index": 1}
        )
    ]


@pytest.fixture
def mock_embeddings(mocker):
    """モック化された埋め込み生成器

    Args:
        mocker: pytest-mockのmocker fixture

    Returns:
        Mock: EmbeddingGeneratorのモック
    """
    mock = mocker.Mock()
    # embed_query は単一のベクトルを返す
    mock.embed_query.return_value = [0.1] * 768
    # embed_documents は複数のベクトルを返す
    mock.embed_documents.return_value = [[0.1] * 768, [0.2] * 768]
    # get_embedding_dimension は次元数を返す
    mock.get_embedding_dimension.return_value = 768
    return mock


@pytest.fixture
def mock_vector_store(mocker):
    """モック化されたベクトルストア

    Args:
        mocker: pytest-mockのmocker fixture

    Returns:
        Mock: VectorStoreのモック
    """
    mock = mocker.Mock()
    # initialize は成功する
    mock.initialize.return_value = None
    # add_documents は正常に完了
    mock.add_documents.return_value = None
    # search は空のリストを返す（テストごとにオーバーライド）
    mock.search.return_value = []
    # get_document_count はドキュメント数を返す
    mock.get_document_count.return_value = 0
    return mock


@pytest.fixture
def temp_chroma_db(tmp_path):
    """一時的なChromaDBディレクトリ

    統合テストで実際のChromaDBを使用する際に利用します。
    テスト終了後に自動的にクリーンアップされます。

    Args:
        tmp_path: pytestが提供する一時ディレクトリ

    Yields:
        Path: ChromaDBの永続化ディレクトリパス
    """
    chroma_path = tmp_path / "integration_test_chroma_db"
    chroma_path.mkdir(exist_ok=True)

    yield chroma_path

    # クリーンアップ
    if chroma_path.exists():
        shutil.rmtree(chroma_path)


@pytest.fixture
def integration_config(temp_chroma_db, tmp_path, monkeypatch):
    """統合テスト用の設定

    実際のChromaDBを使用するための設定を作成します。
    OllamaはモックせずにlocalのOllamaサービスを使用します。

    Args:
        temp_chroma_db: 一時的なChromaDBディレクトリ
        tmp_path: pytestが提供する一時ディレクトリ
        monkeypatch: 環境変数を上書きするためのfixture

    Returns:
        Config: 統合テスト用のConfigオブジェクト
    """
    # 環境変数を直接設定して、既存の設定をオーバーライド
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_LLM_MODEL", "gpt-oss:latest")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(temp_chroma_db))
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    env_file = tmp_path / "integration_test.env"
    env_file.write_text(f"""
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=gpt-oss:latest
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
CHROMA_PERSIST_DIRECTORY={temp_chroma_db}
CHUNK_SIZE=500
CHUNK_OVERLAP=100
LOG_LEVEL=DEBUG
""")

    return Config(env_file=str(env_file))


@pytest.fixture
def sample_text_files(tmp_path):
    """統合テスト用のサンプルテキストファイル

    複数のテキストファイルを作成して、エンドツーエンドテストで使用します。

    Args:
        tmp_path: pytestが提供する一時ディレクトリ

    Returns:
        dict[str, Path]: ファイル名とパスの辞書
    """
    files = {}

    # サンプルファイル1: Python について
    file1 = tmp_path / "python_intro.txt"
    file1.write_text("""Pythonは、1991年にGuido van Rossumによって開発されたプログラミング言語です。
Pythonはシンプルで読みやすい構文が特徴で、初心者にも学びやすい言語として人気があります。
データサイエンス、機械学習、Web開発など、幅広い分野で使用されています。
""", encoding="utf-8")
    files["python"] = file1

    # サンプルファイル2: RAGについて
    file2 = tmp_path / "rag_overview.txt"
    file2.write_text("""RAG（Retrieval-Augmented Generation）は、検索と生成を組み合わせたAI技術です。
大規模言語モデルに外部知識を組み込むことで、より正確で最新の情報を提供できます。
RAGシステムは、ベクトルデータベースを使用してドキュメントを検索し、その結果を基に回答を生成します。
""", encoding="utf-8")
    files["rag"] = file2

    # サンプルファイル3: LLMについて
    file3 = tmp_path / "llm_basics.txt"
    file3.write_text("""LLM（Large Language Model）は、大量のテキストデータで訓練された大規模な言語モデルです。
GPT、Claude、Llama などが代表的なLLMです。
LLMは自然言語理解、文章生成、翻訳、要約など、様々なタスクに利用できます。
""", encoding="utf-8")
    files["llm"] = file3

    return files
