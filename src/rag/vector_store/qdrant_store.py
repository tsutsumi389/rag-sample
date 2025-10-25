"""Qdrantベクトルストア実装

Qdrantベクトルデータベースの管理・操作を担当します。
高性能な類似検索、メタデータフィルタリング、スケーラブルな運用が可能です。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .base import BaseVectorStore, VectorStoreError
from ...models.document import Chunk, SearchResult, ImageDocument
from ...utils.config import Config

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Qdrantベクトルストアの管理クラス

    Qdrant クライアントを使用して高性能なベクトル検索を実現します。

    Attributes:
        config: アプリケーション設定
        collection_name: コレクション名
        client: Qdrantクライアント
        embedding_dim: 埋め込みベクトルの次元数
    """

    def __init__(
        self,
        config: Config,
        collection_name: str = "documents"
    ):
        """QdrantVectorStoreの初期化

        Args:
            config: アプリケーション設定
            collection_name: コレクション名（デフォルト: "documents"）
        """
        super().__init__(config, collection_name)
        self.client = None
        self.embedding_dim = 384  # nomic-embed-textのデフォルト次元数

    def initialize(self) -> None:
        """Qdrantクライアントとコレクションの初期化

        Qdrantサーバーに接続し、コレクションを作成します。

        Raises:
            VectorStoreError: 初期化に失敗した場合
        """
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams, PointStruct

            logger.info(
                f"Qdrantに接続中: {self.config.qdrant_host}:{self.config.qdrant_port}"
            )

            # Qdrantクライアントの作成
            if self.config.qdrant_api_key:
                # Qdrant Cloudを使用
                self.client = QdrantClient(
                    host=self.config.qdrant_host,
                    port=self.config.qdrant_port,
                    api_key=self.config.qdrant_api_key,
                    prefer_grpc=True
                )
            else:
                # ローカルQdrantを使用
                self.client = QdrantClient(
                    host=self.config.qdrant_host,
                    port=self.config.qdrant_port,
                    prefer_grpc=False
                )

            # コレクションの存在確認と作成
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )

            if not collection_exists:
                logger.info(f"コレクション '{self.collection_name}' を作成中...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"コレクション '{self.collection_name}' を作成しました")
            else:
                logger.info(f"コレクション '{self.collection_name}' は既に存在します")

            # ドキュメント数を取得
            count = self.client.count(collection_name=self.collection_name).count
            logger.info(
                f"コレクション '{self.collection_name}' を初期化しました "
                f"(ドキュメント数: {count})"
            )

        except ImportError:
            error_msg = (
                "qdrant-clientがインストールされていません。\n"
                "以下のコマンドでインストールしてください:\n"
                "  uv add qdrant-client"
            )
            logger.error(error_msg)
            raise VectorStoreError(error_msg)
        except Exception as e:
            error_msg = f"Qdrantの初期化に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def add_documents(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]]
    ) -> None:
        """ドキュメントチャンクをQdrantに追加

        Args:
            chunks: 追加するChunkオブジェクトのリスト
            embeddings: 各チャンクの埋め込みベクトルのリスト

        Raises:
            VectorStoreError: 追加に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("Qdrantクライアントが初期化されていません")

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"チャンク数({len(chunks)})と埋め込み数({len(embeddings)})が一致しません"
            )

        if not chunks:
            logger.warning("追加するチャンクがありません")
            return

        try:
            from qdrant_client.models import PointStruct

            logger.info(f"{len(chunks)}個のチャンクを追加中...")

            # Qdrant用のPointデータを準備
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                # メタデータとコンテンツを結合
                payload = {
                    "content": chunk.content,
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                }
                # メタデータをマージ
                if chunk.metadata:
                    payload.update(chunk.metadata)

                point = PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)

            # バッチでQdrantに追加
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            count = self.client.count(collection_name=self.collection_name).count
            logger.info(
                f"{len(chunks)}個のチャンクを正常に追加しました "
                f"(総ドキュメント数: {count})"
            )

        except Exception as e:
            error_msg = f"ドキュメントの追加に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
        **kwargs
    ) -> list[SearchResult]:
        """埋め込みベクトルを使用して類似ドキュメントを検索

        Args:
            query_embedding: クエリの埋め込みベクトル
            n_results: 返す結果の最大数
            where: メタデータフィルタ（例: {"document_id": "doc123"}）
            **kwargs: Qdrant固有のフィルタ条件

        Returns:
            SearchResultオブジェクトのリスト（類似度の高い順）

        Raises:
            VectorStoreError: 検索に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("Qdrantクライアントが初期化されていません")

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            logger.debug(f"類似検索を実行中（結果数: {n_results}）...")

            # Qdrantフィルタの構築
            query_filter = None
            if where:
                conditions = []
                for key, value in where.items():
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
                if conditions:
                    query_filter = Filter(must=conditions)

            # Qdrantで検索
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=n_results,
                query_filter=query_filter
            )

            # 結果が空の場合
            if not search_result:
                logger.info("検索結果が見つかりませんでした")
                return []

            # SearchResultオブジェクトに変換
            search_results = []
            for rank, scored_point in enumerate(search_result, start=1):
                payload = scored_point.payload

                # Chunkオブジェクトを再構築
                chunk = Chunk(
                    content=payload.get("content", ""),
                    chunk_id=payload.get("chunk_id", str(scored_point.id)),
                    document_id=payload.get("document_id", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    start_char=payload.get("start_char", 0),
                    end_char=payload.get("end_char", 0),
                    metadata=payload
                )

                # Qdrantのスコアは既にコサイン類似度（0〜1）
                score = scored_point.score

                search_result_obj = SearchResult(
                    chunk=chunk,
                    score=score,
                    document_name=payload.get("document_name", "Unknown"),
                    document_source=payload.get("source", "Unknown"),
                    rank=rank,
                    metadata=payload
                )
                search_results.append(search_result_obj)

            logger.info(f"{len(search_results)}件の検索結果を取得しました")
            return search_results

        except Exception as e:
            error_msg = f"検索に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

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
            **kwargs: Qdrant固有の削除条件

        Returns:
            削除されたチャンク数

        Raises:
            VectorStoreError: 削除に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("Qdrantクライアントが初期化されていません")

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            initial_count = self.client.count(collection_name=self.collection_name).count

            # document_idによる削除
            if document_id:
                logger.info(f"ドキュメント '{document_id}' を削除中...")
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )
                )

            # chunk_idsによる削除
            elif chunk_ids:
                logger.info(f"{len(chunk_ids)}個のチャンクを削除中...")
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=chunk_ids
                )

            else:
                raise VectorStoreError(
                    "削除条件が指定されていません（document_idまたはchunk_idsが必要）"
                )

            final_count = self.client.count(collection_name=self.collection_name).count
            deleted_count = initial_count - final_count

            logger.info(
                f"{deleted_count}個のチャンクを削除しました "
                f"(残りドキュメント数: {final_count})"
            )

            return deleted_count

        except Exception as e:
            error_msg = f"削除に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def list_documents(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """ストア内のドキュメント一覧を取得

        Args:
            limit: 返すドキュメント数の上限（Noneの場合は全件）

        Returns:
            ドキュメント情報の辞書のリスト

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("Qdrantクライアントが初期化されていません")

        try:
            count = self.client.count(collection_name=self.collection_name).count

            if count == 0:
                logger.info("ストアにドキュメントがありません")
                return []

            # すべてのポイントを取得
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit if limit else count,
                with_payload=True,
                with_vectors=False
            )

            points = scroll_result[0]

            # ドキュメントIDごとにグループ化
            documents_map: dict[str, dict[str, Any]] = {}

            for point in points:
                payload = point.payload
                doc_id = payload.get("document_id", "unknown")

                if doc_id not in documents_map:
                    documents_map[doc_id] = {
                        "document_id": doc_id,
                        "document_name": payload.get("document_name", "Unknown"),
                        "source": payload.get("source", "Unknown"),
                        "doc_type": payload.get("doc_type", "Unknown"),
                        "chunk_count": 0,
                        "total_size": 0
                    }

                documents_map[doc_id]["chunk_count"] += 1
                documents_map[doc_id]["total_size"] += payload.get("size", 0)

            documents = list(documents_map.values())
            logger.info(f"{len(documents)}個のドキュメントを取得しました")

            return documents

        except Exception as e:
            error_msg = f"ドキュメント一覧の取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def clear(self) -> None:
        """コレクション内のすべてのドキュメントを削除

        Raises:
            VectorStoreError: クリアに失敗した場合
        """
        if not self.client:
            raise VectorStoreError("Qdrantクライアントが初期化されていません")

        try:
            from qdrant_client.models import Distance, VectorParams

            count = self.client.count(collection_name=self.collection_name).count

            if count == 0:
                logger.info("コレクションは既に空です")
                return

            logger.warning(f"コレクション '{self.collection_name}' の全データを削除中...")

            # コレクションを削除して再作成
            self.client.delete_collection(collection_name=self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )

            logger.info(f"{count}個のドキュメントを削除しました")

        except Exception as e:
            error_msg = f"コレクションのクリアに失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def get_document_count(self) -> int:
        """コレクション内のドキュメントチャンク数を取得

        Returns:
            チャンク数

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("Qdrantクライアントが初期化されていません")

        try:
            return self.client.count(collection_name=self.collection_name).count
        except Exception as e:
            error_msg = f"ドキュメント数の取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def get_document_by_id(self, document_id: str) -> Optional[dict[str, Any]]:
        """ドキュメントIDで特定のドキュメントを取得

        Args:
            document_id: 取得するドキュメントのID

        Returns:
            ドキュメント情報の辞書、見つからない場合はNone

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("Qdrantクライアントが初期化されていません")

        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            logger.debug(f"ドキュメントID '{document_id}' を取得中...")

            # document_idでフィルタリングして取得
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False
            )

            points = scroll_result[0]

            if not points:
                logger.info(f"ドキュメントID '{document_id}' が見つかりませんでした")
                return None

            # チャンク情報を集約
            chunks = []
            metadata = None

            for point in points:
                payload = point.payload
                if metadata is None:
                    metadata = payload

                chunks.append({
                    "chunk_id": payload.get("chunk_id", str(point.id)),
                    "content": payload.get("content", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "start_char": payload.get("start_char", 0),
                    "end_char": payload.get("end_char", 0),
                    "size": payload.get("size", 0)
                })

            # チャンクをインデックス順にソート
            chunks.sort(key=lambda x: x["chunk_index"])

            document_info = {
                "document_id": document_id,
                "document_name": metadata.get("document_name", "Unknown"),
                "source": metadata.get("source", "Unknown"),
                "doc_type": metadata.get("doc_type", "Unknown"),
                "chunk_count": len(chunks),
                "total_size": sum(chunk.get("size", 0) for chunk in chunks),
                "chunks": chunks
            }

            logger.debug(f"ドキュメント '{document_info['document_name']}' を取得しました")
            return document_info

        except Exception as e:
            error_msg = f"ドキュメントの取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def get_collection_info(self) -> dict[str, Any]:
        """コレクションの情報を取得

        Returns:
            コレクション情報の辞書

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("Qdrantクライアントが初期化されていません")

        try:
            documents = self.list_documents()
            collection_info = self.client.get_collection(
                collection_name=self.collection_name
            )

            return {
                "collection_name": self.collection_name,
                "total_chunks": collection_info.points_count,
                "unique_documents": len(documents),
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance,
                "qdrant_host": self.config.qdrant_host,
                "qdrant_port": self.config.qdrant_port,
            }

        except Exception as e:
            error_msg = f"コレクション情報の取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def close(self) -> None:
        """Qdrantクライアントを閉じる

        明示的なリソース解放が必要な場合に使用します。
        """
        logger.info("Qdrantクライアントをクローズしています...")
        if self.client:
            self.client.close()
        self.client = None
