"""埋め込み生成モジュール

このモジュールはOllamaを使用してテキストのベクトル埋め込みを生成します。
ドキュメントのベクトル化とクエリのベクトル化を行い、
バッチ処理にも対応しています。
"""

from typing import Optional
from langchain_ollama import OllamaEmbeddings

from src.utils.config import Config, get_config


class EmbeddingError(Exception):
    """埋め込み生成エラー"""
    pass


class EmbeddingGenerator:
    """埋め込み生成クラス

    Ollamaの埋め込みモデルを使用してテキストをベクトル化します。
    デフォルトでnomic-embed-textモデルを使用します。

    Attributes:
        config: アプリケーション設定
        embeddings: LangChain OllamaEmbeddingsインスタンス
        model_name: 使用する埋め込みモデル名
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """埋め込み生成器の初期化

        Args:
            config: 設定インスタンス（省略時はデフォルト設定を使用）
            model_name: 埋め込みモデル名（省略時は設定ファイルの値を使用）
            base_url: Ollama APIのベースURL（省略時は設定ファイルの値を使用）

        Raises:
            EmbeddingError: Ollamaとの接続に失敗した場合
        """
        self.config = config or get_config()
        self.model_name = model_name or self.config.ollama_embedding_model
        self.base_url = base_url or self.config.ollama_base_url

        # OllamaEmbeddingsの初期化
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.model_name,
                base_url=self.base_url
            )
        except Exception as e:
            raise EmbeddingError(
                f"Failed to initialize OllamaEmbeddings: {e}\n"
                f"Make sure Ollama is running at {self.base_url} and "
                f"model '{self.model_name}' is available."
            ) from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """複数のドキュメントをベクトル化

        テキストのリストを受け取り、それぞれのベクトル表現を生成します。
        大量のドキュメントに対してバッチ処理を行います。

        Args:
            texts: ベクトル化するテキストのリスト

        Returns:
            list[list[float]]: 各テキストのベクトル表現のリスト

        Raises:
            EmbeddingError: ベクトル化に失敗した場合
            ValueError: textsが空の場合

        Example:
            >>> generator = EmbeddingGenerator()
            >>> texts = ["This is a document.", "Another document."]
            >>> embeddings = generator.embed_documents(texts)
            >>> len(embeddings) == len(texts)
            True
        """
        if not texts:
            raise ValueError("texts cannot be empty")

        # 空文字列のチェック
        if any(not text.strip() for text in texts):
            raise ValueError("texts cannot contain empty strings")

        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed documents (count: {len(texts)}): {e}"
            ) from e

    def embed_query(self, text: str) -> list[float]:
        """クエリをベクトル化

        検索クエリのテキストをベクトル表現に変換します。
        embed_documents()とは異なる最適化が行われる場合があります。

        Args:
            text: ベクトル化するクエリテキスト

        Returns:
            list[float]: クエリのベクトル表現

        Raises:
            EmbeddingError: ベクトル化に失敗した場合
            ValueError: textが空の場合

        Example:
            >>> generator = EmbeddingGenerator()
            >>> query = "What is the meaning of life?"
            >>> embedding = generator.embed_query(query)
            >>> isinstance(embedding, list)
            True
        """
        if not text.strip():
            raise ValueError("text cannot be empty")

        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to embed query: {e}"
            ) from e

    def get_embedding_dimension(self) -> int:
        """埋め込みベクトルの次元数を取得

        埋め込みモデルが生成するベクトルの次元数を返します。
        これはChromaDBのコレクション設定などで使用されます。

        Returns:
            int: ベクトルの次元数

        Raises:
            EmbeddingError: 次元数の取得に失敗した場合

        Example:
            >>> generator = EmbeddingGenerator()
            >>> dim = generator.get_embedding_dimension()
            >>> dim > 0
            True
        """
        try:
            # サンプルテキストで埋め込みを生成して次元数を取得
            sample_embedding = self.embed_query("sample text")
            return len(sample_embedding)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to get embedding dimension: {e}"
            ) from e

    def __repr__(self) -> str:
        """文字列表現"""
        return (
            f"EmbeddingGenerator("
            f"model='{self.model_name}', "
            f"base_url='{self.base_url}'"
            f")"
        )


# 便利関数: デフォルト設定で埋め込み生成器を作成
def create_embedding_generator(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> EmbeddingGenerator:
    """埋め込み生成器を作成

    デフォルト設定を使用して埋め込み生成器を作成します。

    Args:
        model_name: 埋め込みモデル名（省略時は設定ファイルの値を使用）
        base_url: Ollama APIのベースURL（省略時は設定ファイルの値を使用）

    Returns:
        EmbeddingGenerator: 埋め込み生成器インスタンス

    Example:
        >>> generator = create_embedding_generator()
        >>> embedding = generator.embed_query("Hello world")
    """
    return EmbeddingGenerator(model_name=model_name, base_url=base_url)
