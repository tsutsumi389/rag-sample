"""ベクトルストア抽象基底クラス

すべてのベクトルデータベース実装が継承すべき抽象基底クラスを定義します。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ...models.document import Chunk, SearchResult, ImageDocument
from ...utils.config import Config


class VectorStoreError(Exception):
    """ベクトルストア操作のエラー"""
    pass


class BaseVectorStore(ABC):
    """ベクトルストアの抽象基底クラス

    すべてのベクトルデータベース実装が実装すべきメソッドを定義します。

    Attributes:
        config: アプリケーション設定
        collection_name: コレクション名
    """

    def __init__(self, config: Config, collection_name: str = "documents"):
        """初期化

        Args:
            config: アプリケーション設定
            collection_name: コレクション名
        """
        self.config = config
        self.collection_name = collection_name

    @abstractmethod
    def initialize(self) -> None:
        """ベクトルストアの初期化

        クライアント接続、コレクション作成などを行います。

        Raises:
            VectorStoreError: 初期化に失敗した場合
        """
        pass

    @abstractmethod
    def add_documents(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]]
    ) -> None:
        """ドキュメントチャンクをベクトルストアに追加

        Args:
            chunks: 追加するChunkオブジェクトのリスト
            embeddings: 各チャンクの埋め込みベクトルのリスト

        Raises:
            VectorStoreError: 追加に失敗した場合
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        **kwargs
    ) -> list[SearchResult]:
        """埋め込みベクトルを使用して類似ドキュメントを検索

        Args:
            query_embedding: クエリの埋め込みベクトル
            n_results: 返す結果の最大数
            **kwargs: DB固有のフィルタ条件

        Returns:
            SearchResultオブジェクトのリスト（類似度の高い順）

        Raises:
            VectorStoreError: 検索に失敗した場合
        """
        pass

    @abstractmethod
    def delete(
        self,
        document_id: Optional[str] = None,
        chunk_ids: Optional[list[str]] = None,
        **kwargs
    ) -> int:
        """ドキュメントまたはチャンクを削除

        Args:
            document_id: 削除するドキュメントID
            chunk_ids: 削除する特定のチャンクIDのリスト
            **kwargs: DB固有の削除条件

        Returns:
            削除されたチャンク数

        Raises:
            VectorStoreError: 削除に失敗した場合
        """
        pass

    @abstractmethod
    def list_documents(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """ストア内のドキュメント一覧を取得

        Args:
            limit: 返すドキュメント数の上限

        Returns:
            ドキュメント情報の辞書のリスト

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """コレクション内のすべてのドキュメントを削除

        Raises:
            VectorStoreError: クリアに失敗した場合
        """
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """コレクション内のドキュメントチャンク数を取得

        Returns:
            チャンク数

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        pass

    def get_document_by_id(self, document_id: str) -> Optional[dict[str, Any]]:
        """ドキュメントIDで特定のドキュメントを取得

        デフォルト実装を提供しますが、各実装でオーバーライド可能です。

        Args:
            document_id: 取得するドキュメントのID

        Returns:
            ドキュメント情報の辞書、見つからない場合はNone

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        # デフォルトではlist_documentsから検索
        documents = self.list_documents()
        for doc in documents:
            if doc.get('document_id') == document_id:
                return doc
        return None

    def get_collection_info(self) -> dict[str, Any]:
        """コレクションの情報を取得

        デフォルト実装を提供します。

        Returns:
            コレクション情報の辞書

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        documents = self.list_documents()
        return {
            'collection_name': self.collection_name,
            'total_chunks': self.get_document_count(),
            'unique_documents': len(documents),
        }

    # ==================== 画像関連メソッド ====================

    def add_images(
        self,
        images: list[ImageDocument],
        embeddings: list[list[float]],
        collection_name: str = "images"
    ) -> list[str]:
        """画像ドキュメントを画像コレクションに追加

        デフォルトではNotImplementedErrorを発生させます。
        各実装で必要に応じてオーバーライドしてください。

        Args:
            images: 追加するImageDocumentオブジェクトのリスト
            embeddings: 各画像の埋め込みベクトルのリスト
            collection_name: 画像コレクション名（デフォルト: "images"）

        Returns:
            追加された画像のIDリスト

        Raises:
            NotImplementedError: 実装されていない場合
            VectorStoreError: 追加に失敗した場合
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement image support"
        )

    def search_images(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection_name: str = "images",
        **kwargs
    ) -> list[SearchResult]:
        """埋め込みベクトルを使用して類似画像を検索

        デフォルトではNotImplementedErrorを発生させます。

        Args:
            query_embedding: クエリの埋め込みベクトル
            top_k: 返す結果の最大数
            collection_name: 検索する画像コレクション名（デフォルト: "images"）
            **kwargs: DB固有のフィルタ条件

        Returns:
            SearchResultオブジェクトのリスト（類似度の高い順）

        Raises:
            NotImplementedError: 実装されていない場合
            VectorStoreError: 検索に失敗した場合
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement image search"
        )

    def get_image_by_id(
        self,
        image_id: str,
        collection_name: str = "images"
    ) -> Optional[ImageDocument]:
        """IDで画像ドキュメントを取得

        デフォルトではNotImplementedErrorを発生させます。

        Args:
            image_id: 取得する画像のID
            collection_name: 画像コレクション名（デフォルト: "images"）

        Returns:
            ImageDocumentオブジェクト、見つからない場合はNone

        Raises:
            NotImplementedError: 実装されていない場合
            VectorStoreError: 取得に失敗した場合
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement image retrieval"
        )

    def remove_image(
        self,
        image_id: str,
        collection_name: str = "images"
    ) -> bool:
        """画像をIDで削除

        デフォルトではNotImplementedErrorを発生させます。

        Args:
            image_id: 削除する画像のID
            collection_name: 画像コレクション名（デフォルト: "images"）

        Returns:
            削除に成功した場合True、画像が見つからない場合False

        Raises:
            NotImplementedError: 実装されていない場合
            VectorStoreError: 削除に失敗した場合
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement image removal"
        )

    def list_images(
        self,
        collection_name: str = "images",
        limit: Optional[int] = None
    ) -> list[ImageDocument]:
        """画像コレクション内の全画像を取得

        デフォルトではNotImplementedErrorを発生させます。

        Args:
            collection_name: 画像コレクション名（デフォルト: "images"）
            limit: 返す画像数の上限（Noneの場合は全件）

        Returns:
            ImageDocumentオブジェクトのリスト

        Raises:
            NotImplementedError: 実装されていない場合
            VectorStoreError: 取得に失敗した場合
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement image listing"
        )

    def search_multimodal(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        text_weight: Optional[float] = None,
        image_weight: Optional[float] = None
    ) -> list[SearchResult]:
        """テキストと画像を統合したマルチモーダル検索

        デフォルトではNotImplementedErrorを発生させます。

        Args:
            query_embedding: クエリの埋め込みベクトル
            top_k: 返す結果の最大数
            text_weight: テキスト検索結果の重み（Noneの場合は設定値を使用）
            image_weight: 画像検索結果の重み（Noneの場合は設定値を使用）

        Returns:
            SearchResultオブジェクトのリスト（スコアの高い順）

        Raises:
            NotImplementedError: 実装されていない場合
            VectorStoreError: 検索に失敗した場合
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement multimodal search"
        )

    @abstractmethod
    def close(self) -> None:
        """ベクトルストアのクライアント接続を閉じる

        リソース解放が必要な場合に使用します。
        """
        pass

    def __enter__(self):
        """コンテキストマネージャーのエントリ"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.close()
        return False
