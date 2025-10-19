"""ChromaDBベクトルストアモジュール

このモジュールはChromaDBベクトルデータベースの管理・操作を担当します。
ドキュメントの追加、削除、検索、メタデータフィルタリングなどの機能を提供します。
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings

from ..models.document import Chunk, SearchResult, ImageDocument
from ..utils.config import Config

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """ベクトルストア操作のエラー"""
    pass


class VectorStore:
    """ChromaDBベクトルストアの管理クラス

    PersistentClientを使用してデータの永続化を行います。
    ドキュメントチャンクの追加、検索、削除、一覧取得などの操作を提供します。

    Attributes:
        config: アプリケーション設定
        client: ChromaDB永続化クライアント
        collection: 現在のChromaDBコレクション
        collection_name: コレクション名
    """

    def __init__(
        self,
        config: Config,
        collection_name: str = "documents"
    ):
        """VectorStoreの初期化

        Args:
            config: アプリケーション設定
            collection_name: コレクション名（デフォルト: "documents"）
        """
        self.config = config
        self.collection_name = collection_name
        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[Collection] = None

    def initialize(self) -> None:
        """ChromaDBクライアントとコレクションの初期化

        永続化ディレクトリが存在しない場合は作成し、
        ChromaDBクライアントとコレクションを初期化します。

        Raises:
            VectorStoreError: 初期化に失敗した場合
        """
        try:
            # ChromaDBディレクトリの作成
            self.config.ensure_chroma_directory()
            chroma_path = self.config.get_chroma_path()

            logger.info(f"ChromaDBを初期化中: {chroma_path}")

            # PersistentClientの作成
            self.client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # コレクションの取得または作成
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG application document store"}
            )

            logger.info(
                f"コレクション '{self.collection_name}' を初期化しました "
                f"(ドキュメント数: {self.collection.count()})"
            )

        except Exception as e:
            error_msg = f"ChromaDBの初期化に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

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
        if not self.collection:
            raise VectorStoreError("コレクションが初期化されていません")

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"チャンク数({len(chunks)})と埋め込み数({len(embeddings)})が一致しません"
            )

        if not chunks:
            logger.warning("追加するチャンクがありません")
            return

        try:
            # ChromaDB用のデータを準備
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]

            logger.info(f"{len(chunks)}個のチャンクを追加中...")

            # バッチでChromaDBに追加
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(
                f"{len(chunks)}個のチャンクを正常に追加しました "
                f"(総ドキュメント数: {self.collection.count()})"
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
        where_document: Optional[dict[str, Any]] = None
    ) -> list[SearchResult]:
        """埋め込みベクトルを使用して類似ドキュメントを検索

        Args:
            query_embedding: クエリの埋め込みベクトル
            n_results: 返す結果の最大数
            where: メタデータフィルタ（例: {"document_id": "doc123"}）
            where_document: ドキュメントコンテンツのフィルタ

        Returns:
            SearchResultオブジェクトのリスト（類似度の高い順）

        Raises:
            VectorStoreError: 検索に失敗した場合
        """
        if not self.collection:
            raise VectorStoreError("コレクションが初期化されていません")

        try:
            logger.debug(f"類似検索を実行中（結果数: {n_results}）...")

            # ChromaDBで検索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=["documents", "metadatas", "distances"]
            )

            # 結果が空の場合
            if not results['ids'][0]:
                logger.info("検索結果が見つかりませんでした")
                return []

            # SearchResultオブジェクトに変換
            search_results = []
            for rank, (doc_id, document, metadata, distance) in enumerate(
                zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ),
                start=1
            ):
                # Chunkオブジェクトを再構築
                chunk = Chunk(
                    content=document,
                    chunk_id=doc_id,
                    document_id=metadata.get('document_id', ''),
                    chunk_index=metadata.get('chunk_index', 0),
                    start_char=metadata.get('start_char', 0),
                    end_char=metadata.get('end_char', 0),
                    metadata=metadata
                )

                # 距離をスコアに変換（距離が小さいほどスコアが高い）
                # ChromaDBのL2距離を0-1のスコアに正規化
                score = 1.0 / (1.0 + distance)

                search_result = SearchResult(
                    chunk=chunk,
                    score=score,
                    document_name=metadata.get('document_name', 'Unknown'),
                    document_source=metadata.get('source', 'Unknown'),
                    rank=rank,
                    metadata=metadata
                )
                search_results.append(search_result)

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
        where: Optional[dict[str, Any]] = None
    ) -> int:
        """ドキュメントまたはチャンクを削除

        Args:
            document_id: 削除するドキュメントID（すべてのチャンクが削除されます）
            chunk_ids: 削除する特定のチャンクIDのリスト
            where: メタデータフィルタによる削除条件

        Returns:
            削除されたチャンク数

        Raises:
            VectorStoreError: 削除に失敗した場合
        """
        if not self.collection:
            raise VectorStoreError("コレクションが初期化されていません")

        try:
            initial_count = self.collection.count()

            # document_idによる削除
            if document_id:
                logger.info(f"ドキュメント '{document_id}' を削除中...")
                self.collection.delete(
                    where={"document_id": document_id}
                )

            # chunk_idsによる削除
            elif chunk_ids:
                logger.info(f"{len(chunk_ids)}個のチャンクを削除中...")
                self.collection.delete(ids=chunk_ids)

            # whereフィルタによる削除
            elif where:
                logger.info(f"フィルタ条件でドキュメントを削除中: {where}")
                self.collection.delete(where=where)

            else:
                raise VectorStoreError(
                    "削除条件が指定されていません（document_id、chunk_ids、whereのいずれかが必要）"
                )

            final_count = self.collection.count()
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
        if not self.collection:
            raise VectorStoreError("コレクションが初期化されていません")

        try:
            count = self.collection.count()

            if count == 0:
                logger.info("ストアにドキュメントがありません")
                return []

            # すべてのデータを取得
            results = self.collection.get(
                limit=limit,
                include=["metadatas", "documents"]
            )

            # ドキュメントIDごとにグループ化
            documents_map: dict[str, dict[str, Any]] = {}

            for chunk_id, metadata, document in zip(
                results['ids'],
                results['metadatas'],
                results['documents']
            ):
                doc_id = metadata.get('document_id', 'unknown')

                if doc_id not in documents_map:
                    documents_map[doc_id] = {
                        'document_id': doc_id,
                        'document_name': metadata.get('document_name', 'Unknown'),
                        'source': metadata.get('source', 'Unknown'),
                        'doc_type': metadata.get('doc_type', 'Unknown'),
                        'chunk_count': 0,
                        'total_size': 0
                    }

                documents_map[doc_id]['chunk_count'] += 1
                documents_map[doc_id]['total_size'] += metadata.get('size', 0)

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
        if not self.collection:
            raise VectorStoreError("コレクションが初期化されていません")

        try:
            count = self.collection.count()

            if count == 0:
                logger.info("コレクションは既に空です")
                return

            logger.warning(f"コレクション '{self.collection_name}' の全データを削除中...")

            # コレクションを削除して再作成
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG application document store"}
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
        if not self.collection:
            raise VectorStoreError("コレクションが初期化されていません")

        try:
            return self.collection.count()
        except Exception as e:
            error_msg = f"ドキュメント数の取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def get_collection_info(self) -> dict[str, Any]:
        """コレクションの情報を取得

        Returns:
            コレクション情報の辞書

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        if not self.collection:
            raise VectorStoreError("コレクションが初期化されていません")

        try:
            documents = self.list_documents()

            return {
                'collection_name': self.collection_name,
                'total_chunks': self.collection.count(),
                'unique_documents': len(documents),
                'persist_directory': str(self.config.get_chroma_path()),
                'metadata': self.collection.metadata
            }

        except Exception as e:
            error_msg = f"コレクション情報の取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    # ==================== 画像関連メソッド ====================

    def add_images(
        self,
        images: list[ImageDocument],
        embeddings: list[list[float]],
        collection_name: str = "images"
    ) -> list[str]:
        """画像ドキュメントを画像コレクションに追加

        Args:
            images: 追加するImageDocumentオブジェクトのリスト
            embeddings: 各画像の埋め込みベクトルのリスト
            collection_name: 画像コレクション名（デフォルト: "images"）

        Returns:
            追加された画像のIDリスト

        Raises:
            VectorStoreError: 追加に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        if len(images) != len(embeddings):
            raise VectorStoreError(
                f"画像数({len(images)})と埋め込み数({len(embeddings)})が一致しません"
            )

        if not images:
            logger.warning("追加する画像がありません")
            return []

        try:
            # 画像コレクションの取得または作成
            image_collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Multimodal RAG image store"}
            )

            # ChromaDB用のデータを準備
            ids = [img.id for img in images]
            documents = [img.caption for img in images]  # キャプションをドキュメントとして保存
            metadatas = []

            for img in images:
                metadata = {
                    'id': img.id,
                    'file_path': str(img.file_path),
                    'file_name': img.file_name,
                    'image_type': img.image_type,
                    'caption': img.caption,
                    'created_at': img.created_at.isoformat(),
                    'source': 'local',
                }
                # 追加のメタデータをマージ（ChromaDBがサポートする型のみ）
                # ChromaDBは str, int, float, bool, None のみサポート
                if img.metadata:
                    for k, v in img.metadata.items():
                        if k not in metadata:
                            # ChromaDBがサポートする型のみ追加
                            if isinstance(v, (str, int, float, bool, type(None))):
                                metadata[f"custom_{k}"] = v
                            elif isinstance(v, (list, dict)):
                                # リストや辞書は文字列に変換
                                metadata[f"custom_{k}"] = str(v)
                            else:
                                # その他の型も文字列に変換
                                metadata[f"custom_{k}"] = str(v)
                metadatas.append(metadata)

            logger.info(f"{len(images)}個の画像を{collection_name}コレクションに追加中...")

            # バッチでChromaDBに追加
            image_collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(
                f"{len(images)}個の画像を正常に追加しました "
                f"(総画像数: {image_collection.count()})"
            )

            return ids

        except Exception as e:
            error_msg = f"画像の追加に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def search_images(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection_name: str = "images",
        where: Optional[dict[str, Any]] = None
    ) -> list[SearchResult]:
        """埋め込みベクトルを使用して類似画像を検索

        Args:
            query_embedding: クエリの埋め込みベクトル
            top_k: 返す結果の最大数
            collection_name: 検索する画像コレクション名（デフォルト: "images"）
            where: メタデータフィルタ（例: {"image_type": "jpg"}）

        Returns:
            SearchResultオブジェクトのリスト（類似度の高い順）

        Raises:
            VectorStoreError: 検索に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        try:
            # 画像コレクションの取得
            image_collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Multimodal RAG image store"}
            )

            logger.debug(f"画像検索を実行中（結果数: {top_k}）...")

            # ChromaDBで検索
            results = image_collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"]
            )

            # 結果が空の場合
            if not results['ids'][0]:
                logger.info("画像検索結果が見つかりませんでした")
                return []

            # SearchResultオブジェクトに変換
            search_results = []
            for rank, (img_id, caption, metadata, distance) in enumerate(
                zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ),
                start=1
            ):
                # ダミーのChunkオブジェクトを作成（画像用）
                chunk = Chunk(
                    content=caption,
                    chunk_id=img_id,
                    document_id=img_id,
                    chunk_index=0,
                    start_char=0,
                    end_char=len(caption),
                    metadata=metadata
                )

                # 距離をスコアに変換
                score = 1.0 / (1.0 + distance)

                # 画像パスの取得
                image_path = Path(metadata.get('file_path', ''))

                search_result = SearchResult(
                    chunk=chunk,
                    score=score,
                    document_name=metadata.get('file_name', 'Unknown'),
                    document_source=metadata.get('file_path', 'Unknown'),
                    rank=rank,
                    metadata=metadata,
                    result_type='image',
                    image_path=image_path,
                    caption=caption
                )
                search_results.append(search_result)

            logger.info(f"{len(search_results)}件の画像検索結果を取得しました")
            return search_results

        except Exception as e:
            error_msg = f"画像検索に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def get_image_by_id(
        self,
        image_id: str,
        collection_name: str = "images"
    ) -> Optional[ImageDocument]:
        """IDで画像ドキュメントを取得

        Args:
            image_id: 取得する画像のID
            collection_name: 画像コレクション名（デフォルト: "images"）

        Returns:
            ImageDocumentオブジェクト、見つからない場合はNone

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        try:
            # 画像コレクションの取得
            image_collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Multimodal RAG image store"}
            )

            logger.debug(f"画像ID '{image_id}' を取得中...")

            # IDで検索
            results = image_collection.get(
                ids=[image_id],
                include=["documents", "metadatas"]
            )

            # 結果が空の場合
            if not results['ids']:
                logger.info(f"画像ID '{image_id}' が見つかりませんでした")
                return None

            # ImageDocumentオブジェクトに変換
            metadata = results['metadatas'][0]
            caption = results['documents'][0]

            # カスタムメタデータの抽出
            custom_metadata = {
                k.replace('custom_', ''): v
                for k, v in metadata.items()
                if k.startswith('custom_')
            }

            image_doc = ImageDocument(
                id=metadata['id'],
                file_path=Path(metadata['file_path']),
                file_name=metadata['file_name'],
                image_type=metadata['image_type'],
                caption=caption,
                metadata=custom_metadata,
                created_at=datetime.fromisoformat(metadata['created_at']),
                image_data=None
            )

            logger.debug(f"画像 '{image_doc.file_name}' を取得しました")
            return image_doc

        except Exception as e:
            error_msg = f"画像の取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def remove_image(
        self,
        image_id: str,
        collection_name: str = "images"
    ) -> bool:
        """画像をIDで削除

        Args:
            image_id: 削除する画像のID
            collection_name: 画像コレクション名（デフォルト: "images"）

        Returns:
            削除に成功した場合True、画像が見つからない場合False

        Raises:
            VectorStoreError: 削除に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        try:
            # 画像コレクションの取得
            image_collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Multimodal RAG image store"}
            )

            # 画像の存在確認
            results = image_collection.get(ids=[image_id])
            if not results['ids']:
                logger.warning(f"画像ID '{image_id}' が見つかりませんでした")
                return False

            logger.info(f"画像ID '{image_id}' を削除中...")
            initial_count = image_collection.count()

            # 削除実行
            image_collection.delete(ids=[image_id])

            final_count = image_collection.count()
            logger.info(
                f"画像を削除しました (残り画像数: {final_count})"
            )

            return True

        except Exception as e:
            error_msg = f"画像の削除に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def list_images(
        self,
        collection_name: str = "images",
        limit: Optional[int] = None
    ) -> list[ImageDocument]:
        """画像コレクション内の全画像を取得

        Args:
            collection_name: 画像コレクション名（デフォルト: "images"）
            limit: 返す画像数の上限（Noneの場合は全件）

        Returns:
            ImageDocumentオブジェクトのリスト

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        try:
            # 画像コレクションの取得
            image_collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Multimodal RAG image store"}
            )

            count = image_collection.count()

            if count == 0:
                logger.info("画像コレクションに画像がありません")
                return []

            logger.debug(f"画像コレクションから画像を取得中...")

            # すべての画像データを取得
            results = image_collection.get(
                limit=limit,
                include=["metadatas", "documents"]
            )

            # ImageDocumentオブジェクトのリストに変換
            images = []
            for img_id, caption, metadata in zip(
                results['ids'],
                results['documents'],
                results['metadatas']
            ):
                # カスタムメタデータの抽出
                custom_metadata = {
                    k.replace('custom_', ''): v
                    for k, v in metadata.items()
                    if k.startswith('custom_')
                }

                image_doc = ImageDocument(
                    id=metadata['id'],
                    file_path=Path(metadata['file_path']),
                    file_name=metadata['file_name'],
                    image_type=metadata['image_type'],
                    caption=caption,
                    metadata=custom_metadata,
                    created_at=datetime.fromisoformat(metadata['created_at']),
                    image_data=None
                )
                images.append(image_doc)

            logger.info(f"{len(images)}個の画像を取得しました")
            return images

        except Exception as e:
            error_msg = f"画像一覧の取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def search_multimodal(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        text_weight: Optional[float] = None,
        image_weight: Optional[float] = None
    ) -> list[SearchResult]:
        """テキストと画像を統合したマルチモーダル検索

        テキストコレクションと画像コレクションの両方を検索し、
        重み付けしてマージした結果を返します。

        Args:
            query_embedding: クエリの埋め込みベクトル
            top_k: 返す結果の最大数
            text_weight: テキスト検索結果の重み（Noneの場合は設定値を使用）
            image_weight: 画像検索結果の重み（Noneの場合は設定値を使用）

        Returns:
            SearchResultオブジェクトのリスト（スコアの高い順）

        Raises:
            VectorStoreError: 検索に失敗した場合
        """
        try:
            # 重みの設定（デフォルトは設定ファイルから）
            if text_weight is None:
                text_weight = self.config.multimodal_search_text_weight
            if image_weight is None:
                image_weight = self.config.multimodal_search_image_weight

            logger.info(
                f"マルチモーダル検索を実行中 "
                f"(text_weight: {text_weight}, image_weight: {image_weight})"
            )

            # テキスト検索
            text_results = []
            try:
                text_results = self.search(
                    query_embedding=query_embedding,
                    n_results=top_k
                )
            except Exception as e:
                logger.warning(f"テキスト検索に失敗: {e}")

            # 画像検索
            image_results = []
            try:
                image_results = self.search_images(
                    query_embedding=query_embedding,
                    top_k=top_k
                )
            except Exception as e:
                logger.warning(f"画像検索に失敗: {e}")

            # スコアの重み付け
            for result in text_results:
                result.score *= text_weight
                result.metadata['search_type'] = 'text'

            for result in image_results:
                result.score *= image_weight
                result.metadata['search_type'] = 'image'

            # 結果をマージしてスコアでソート
            all_results = text_results + image_results
            all_results.sort(key=lambda x: x.score, reverse=True)

            # top_k件に制限
            final_results = all_results[:top_k]

            # ランクを再設定
            for rank, result in enumerate(final_results, start=1):
                result.rank = rank

            logger.info(
                f"マルチモーダル検索完了: "
                f"テキスト{len(text_results)}件 + 画像{len(image_results)}件 "
                f"→ 上位{len(final_results)}件"
            )

            return final_results

        except Exception as e:
            error_msg = f"マルチモーダル検索に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    # ==================== 既存メソッド ====================

    def close(self) -> None:
        """ChromaDBクライアントを閉じる

        明示的なリソース解放が必要な場合に使用します。
        """
        logger.info("ChromaDBクライアントをクローズしています...")
        self.collection = None
        self.client = None

    def __enter__(self):
        """コンテキストマネージャーのエントリ"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.close()
        return False
