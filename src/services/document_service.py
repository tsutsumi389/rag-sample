"""ドキュメント管理サービス

ドキュメントと画像の追加・削除・一覧取得などのビジネスロジックを提供します。
CLIとMCPサーバーで共通利用されます。
"""

import logging
from pathlib import Path
from typing import Any

from ..rag.vector_store import create_vector_store, VectorStoreError
from ..rag.document_processor import DocumentProcessor, DocumentProcessorError, UnsupportedFileTypeError
from ..rag.embeddings import EmbeddingGenerator
from ..rag.image_processor import ImageProcessor, ImageProcessorError
from ..rag.vision_embeddings import VisionEmbeddings, VisionEmbeddingError
from ..utils.config import Config
from .file_utils import is_image_file

logger = logging.getLogger(__name__)


class DocumentServiceError(Exception):
    """ドキュメントサービスのエラー"""
    pass


class DocumentService:
    """ドキュメント管理サービスクラス

    ドキュメントと画像の追加、削除、一覧取得などの操作を提供します。

    Attributes:
        config: アプリケーション設定
        doc_vector_store: テキストドキュメント用ベクトルストア
        img_vector_store: 画像用ベクトルストア
        document_processor: ドキュメントプロセッサ
        embedding_generator: 埋め込み生成器
        image_processor: 画像プロセッサ
        vision_embeddings: ビジョン埋め込み生成器
    """

    def __init__(self, config: Config):
        """初期化

        Args:
            config: アプリケーション設定
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # テキストドキュメント用ベクトルストアの初期化（ファクトリーで作成）
        self.doc_vector_store = create_vector_store(
            config=self.config,
            collection_name="documents"
        )
        self.doc_vector_store.initialize()

        # 画像用ベクトルストアの初期化（ファクトリーで作成）
        self.img_vector_store = create_vector_store(
            config=self.config,
            collection_name="images"
        )
        self.img_vector_store.initialize()

        # 各種プロセッサと埋め込み生成器の初期化
        self.document_processor = DocumentProcessor(self.config)
        self.embedding_generator = EmbeddingGenerator(self.config)
        self.vision_embeddings = VisionEmbeddings(self.config)
        self.image_processor = ImageProcessor(self.vision_embeddings, self.config)

    def add_document_file(
        self,
        file_path: str,
        document_id: str | None = None
    ) -> dict[str, Any]:
        """テキストドキュメントファイルを追加

        Args:
            file_path: 追加するファイルのパス
            document_id: ドキュメントID（省略時は自動生成）

        Returns:
            追加結果を含む辞書

        Raises:
            UnsupportedFileTypeError: サポートされていないファイル形式の場合
            DocumentProcessorError: ドキュメント処理エラーの場合
            VectorStoreError: ベクトルストアエラーの場合
        """
        try:
            self.logger.info(f"テキストドキュメントを追加中: {file_path}")

            # ドキュメントの処理
            document, chunks = self.document_processor.process_document(
                file_path, document_id
            )

            # 埋め込みの生成
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_generator.embed_documents(chunk_texts)

            # ベクトルストアに追加
            self.doc_vector_store.add_documents(chunks, embeddings)

            success_msg = f"ドキュメント '{document.name}' を正常に追加しました"
            self.logger.info(success_msg)

            return {
                "success": True,
                "document_id": chunks[0].document_id if chunks else None,
                "document_name": document.name,
                "document_type": document.doc_type,
                "chunks_count": len(chunks),
                "total_size": document.size,
                "message": success_msg
            }

        except UnsupportedFileTypeError:
            raise
        except DocumentProcessorError:
            raise
        except Exception as e:
            error_msg = f"テキストドキュメント追加に失敗しました: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DocumentServiceError(error_msg) from e

    def add_image_file(
        self,
        image_path: str,
        caption: str | None = None,
        tags: list[str] | None = None
    ) -> dict[str, Any]:
        """画像ファイルを追加

        Args:
            image_path: 追加する画像ファイルのパス
            caption: 画像のキャプション（省略時は自動生成）
            tags: タグのリスト

        Returns:
            追加結果を含む辞書

        Raises:
            ImageProcessorError: 画像処理エラーの場合
            VisionEmbeddingError: ビジョン埋め込みエラーの場合
            VectorStoreError: ベクトルストアエラーの場合
        """
        try:
            self.logger.info(f"画像を追加中: {image_path}")

            # 画像の読み込み
            path = Path(image_path)
            image = self.image_processor.load_image(
                str(path),
                caption=caption,
                tags=tags or []
            )

            # 埋め込みの生成
            embeddings = self.vision_embeddings.embed_images([image.file_path])

            # ベクトルストアに追加
            image_ids = self.img_vector_store.add_images([image], embeddings)

            success_msg = f"画像 '{image.file_name}' を正常に追加しました"
            self.logger.info(success_msg)

            return {
                "success": True,
                "image_id": image_ids[0] if image_ids else None,
                "file_name": image.file_name,
                "image_type": image.image_type,
                "caption": image.caption,
                "tags": image.metadata.get('tags', []),
                "message": success_msg
            }

        except ImageProcessorError:
            raise
        except VisionEmbeddingError:
            raise
        except Exception as e:
            error_msg = f"画像追加に失敗しました: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise DocumentServiceError(error_msg) from e

    def add_file(
        self,
        file_path: str,
        document_id: str | None = None,
        caption: str | None = None,
        tags: list[str] | None = None
    ) -> dict[str, Any]:
        """ファイルを追加（自動判定）

        ファイルの拡張子から画像かドキュメントかを自動判定して追加します。

        Args:
            file_path: 追加するファイルのパス
            document_id: ドキュメントID（テキストファイルの場合、省略時は自動生成）
            caption: 画像のキャプション（画像ファイルの場合、省略時は自動生成）
            tags: タグのリスト（画像ファイルの場合のみ）

        Returns:
            追加結果を含む辞書
        """
        try:
            path = Path(file_path)

            # ファイルの存在確認
            if not path.exists():
                return {
                    "success": False,
                    "message": f"ファイルまたはディレクトリが見つかりません: {file_path}",
                    "error": "FileNotFoundError"
                }

            # ディレクトリの場合は未サポート
            if path.is_dir():
                return {
                    "success": False,
                    "message": "ディレクトリの一括追加は現在サポートされていません。個別のファイルを指定してください。",
                    "error": "DirectoryNotSupported"
                }

            # 画像ファイルかどうかで処理を分岐
            if is_image_file(file_path):
                return self.add_image_file(file_path, caption, tags)
            else:
                return self.add_document_file(file_path, document_id)

        except (UnsupportedFileTypeError, DocumentProcessorError,
                ImageProcessorError, VisionEmbeddingError, VectorStoreError) as e:
            # 既知のエラーはそのまま返す
            return {
                "success": False,
                "message": str(e),
                "error": type(e).__name__
            }
        except Exception as e:
            error_msg = f"ファイル追加に失敗しました: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "message": error_msg,
                "error": str(e)
            }

    def list_documents(
        self,
        limit: int | None = None,
        include_images: bool = True
    ) -> dict[str, Any]:
        """ドキュメント一覧を取得

        Args:
            limit: 返すドキュメント数の上限（Noneの場合は全件）
            include_images: 画像も含めるかどうか（デフォルト: True）

        Returns:
            ドキュメント一覧とメタデータを含む辞書
        """
        try:
            result = {
                "success": True,
                "documents": [],
                "images": [],
                "total_count": 0,
                "message": ""
            }

            # テキストドキュメント取得
            try:
                text_docs = self.doc_vector_store.list_documents(limit=limit)
                result["documents"] = text_docs
                self.logger.info(f"テキストドキュメント: {len(text_docs)}件")
            except VectorStoreError as e:
                self.logger.warning(f"テキストドキュメント取得エラー: {e}")
                result["documents"] = []

            # 画像ドキュメント取得
            if include_images:
                try:
                    image_docs = self.img_vector_store.list_images(limit=limit)
                    # ImageDocumentオブジェクトを辞書形式に変換
                    result["images"] = [img.to_dict() for img in image_docs]
                    self.logger.info(f"画像ドキュメント: {len(image_docs)}件")
                except VectorStoreError as e:
                    self.logger.warning(f"画像ドキュメント取得エラー: {e}")
                    result["images"] = []

            # 合計数を計算
            total = len(result["documents"]) + len(result["images"])
            result["total_count"] = total

            if total == 0:
                result["message"] = "登録されているドキュメントはありません"
            else:
                result["message"] = (
                    f"合計 {total}件のドキュメントを取得しました "
                    f"（テキスト: {len(result['documents'])}件, 画像: {len(result['images'])}件）"
                )

            self.logger.info(result["message"])
            return result

        except Exception as e:
            error_msg = f"ドキュメント一覧取得に失敗しました: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "documents": [],
                "images": [],
                "total_count": 0,
                "message": error_msg,
                "error": str(e)
            }

    def remove_document(
        self,
        item_id: str,
        item_type: str = "auto"
    ) -> dict[str, Any]:
        """ドキュメントまたは画像を削除

        Args:
            item_id: 削除するドキュメントIDまたは画像ID
            item_type: 削除するアイテムのタイプ（'document', 'image', 'auto'）

        Returns:
            削除結果とメタデータを含む辞書
        """
        try:
            self.logger.info(f"削除リクエスト: ID='{item_id}', タイプ='{item_type}'")

            # ドキュメントとして削除を試行
            if item_type in ["document", "auto"]:
                try:
                    # ドキュメントの存在確認
                    documents = self.doc_vector_store.list_documents()
                    target_doc = None
                    for doc in documents:
                        if doc['document_id'] == item_id:
                            target_doc = doc
                            break

                    if target_doc:
                        # ドキュメントとして削除
                        deleted_count = self.doc_vector_store.delete(document_id=item_id)
                        success_msg = f"ドキュメント '{target_doc['document_name']}' を削除しました"
                        self.logger.info(success_msg)

                        return {
                            "success": True,
                            "item_type": "document",
                            "item_id": item_id,
                            "document_name": target_doc['document_name'],
                            "deleted_chunks": deleted_count,
                            "message": success_msg
                        }
                except VectorStoreError as e:
                    self.logger.warning(f"ドキュメント削除エラー: {e}")
                    if item_type == "document":
                        # documentタイプ指定の場合はエラーとして返す
                        return {
                            "success": False,
                            "message": f"ドキュメントの削除に失敗しました: {str(e)}",
                            "error": "VectorStoreError"
                        }

            # 画像として削除を試行
            if item_type in ["image", "auto"]:
                try:
                    # 画像の存在確認と取得
                    image = self.img_vector_store.get_image_by_id(item_id)

                    if image:
                        # 画像として削除
                        success = self.img_vector_store.remove_image(item_id)
                        if success:
                            success_msg = f"画像 '{image.file_name}' を削除しました"
                            self.logger.info(success_msg)

                            return {
                                "success": True,
                                "item_type": "image",
                                "item_id": item_id,
                                "file_name": image.file_name,
                                "message": success_msg
                            }
                        else:
                            return {
                                "success": False,
                                "message": f"画像ID '{item_id}' の削除に失敗しました",
                                "error": "DeleteFailed"
                            }
                except VectorStoreError as e:
                    self.logger.warning(f"画像削除エラー: {e}")
                    if item_type == "image":
                        # imageタイプ指定の場合はエラーとして返す
                        return {
                            "success": False,
                            "message": f"画像の削除に失敗しました: {str(e)}",
                            "error": "VectorStoreError"
                        }

            # どちらでも見つからなかった場合
            return {
                "success": False,
                "message": f"ID '{item_id}' のドキュメントまたは画像が見つかりませんでした",
                "error": "NotFound"
            }

        except Exception as e:
            error_msg = f"削除に失敗しました（ID: '{item_id}'）: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "item_id": item_id,
                "message": error_msg,
                "error": str(e)
            }

    def search_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> dict[str, Any]:
        """ドキュメントを検索

        Args:
            query: 検索クエリ文字列
            top_k: 返す検索結果の最大数（デフォルト: 5）

        Returns:
            検索結果とメタデータを含む辞書
        """
        try:
            self.logger.info(f"検索クエリ: '{query}', top_k: {top_k}")

            # クエリの埋め込みを生成
            query_embedding = self.embedding_generator.embed_query(query)
            self.logger.debug(f"埋め込みベクトル生成完了: 次元数={len(query_embedding)}")

            # ベクトル検索を実行
            search_results = self.doc_vector_store.search(
                query_embedding=query_embedding,
                n_results=top_k
            )

            # 結果をJSON形式に変換
            results_list = []
            for result in search_results:
                result_dict = {
                    "content": result.chunk.content,
                    "score": result.score,
                    "metadata": result.chunk.metadata,
                    "document_name": result.chunk.metadata.get("document_name", "Unknown"),
                    "document_id": result.chunk.metadata.get("document_id", "Unknown"),
                    "chunk_index": result.chunk.metadata.get("chunk_index", 0)
                }
                results_list.append(result_dict)

            success_msg = f"{len(results_list)}件の検索結果を取得しました（クエリ: '{query}'）"
            self.logger.info(success_msg)

            return {
                "success": True,
                "query": query,
                "results": results_list,
                "count": len(results_list),
                "message": success_msg
            }

        except Exception as e:
            error_msg = f"検索に失敗しました（クエリ: '{query}'）: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "query": query,
                "results": [],
                "count": 0,
                "message": error_msg,
                "error": str(e)
            }

    def search_images(
        self,
        query: str,
        top_k: int = 5
    ) -> dict[str, Any]:
        """画像を検索

        テキストクエリを使用して、意味的に類似した画像を検索します。

        Args:
            query: 画像検索のためのテキストクエリ
            top_k: 返す検索結果の最大数（デフォルト: 5）

        Returns:
            検索結果とメタデータを含む辞書
        """
        try:
            self.logger.info(f"画像検索クエリ: '{query}', top_k: {top_k}")

            # テキストクエリの埋め込みを生成
            query_embedding = self.embedding_generator.embed_query(query)
            self.logger.debug(f"クエリ埋め込みベクトル生成完了: 次元数={len(query_embedding)}")

            # 画像ベクトルストアでベクトル検索を実行
            search_results = self.img_vector_store.search_images(
                query_embedding=query_embedding,
                top_k=top_k
            )

            # 結果をJSON形式に変換
            results_list = []
            for result in search_results:
                result_dict = {
                    "image_id": result.chunk.chunk_id,
                    "file_name": result.document_name,
                    "file_path": str(result.image_path) if result.image_path else result.document_source,
                    "caption": result.caption,
                    "image_type": result.metadata.get('image_type', 'Unknown'),
                    "score": result.score,
                    "rank": result.rank,
                    "tags": result.metadata.get('tags', []),
                    "added_at": result.metadata.get('added_at', 'Unknown')
                }
                results_list.append(result_dict)

            success_msg = f"{len(results_list)}件の画像検索結果を取得しました（クエリ: '{query}'）"
            self.logger.info(success_msg)

            return {
                "success": True,
                "query": query,
                "results": results_list,
                "count": len(results_list),
                "message": success_msg
            }

        except VectorStoreError as e:
            error_msg = f"画像検索に失敗しました（クエリ: '{query}'）: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "query": query,
                "results": [],
                "count": 0,
                "message": error_msg,
                "error": "VectorStoreError",
                "hint": "画像が登録されていることを確認してください"
            }
        except Exception as e:
            error_msg = f"画像検索に失敗しました（クエリ: '{query}'）: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "query": query,
                "results": [],
                "count": 0,
                "message": error_msg,
                "error": str(e)
            }

    def get_document_by_id(self, document_id: str) -> dict[str, Any]:
        """ドキュメントIDで特定のドキュメントを取得

        Args:
            document_id: 取得するドキュメントのID

        Returns:
            ドキュメント詳細情報を含む辞書
        """
        try:
            self.logger.info(f"ドキュメント取得: ID='{document_id}'")

            # テキストドキュメントとして検索
            doc_info = self.doc_vector_store.get_document_by_id(document_id)

            if doc_info:
                return {
                    "success": True,
                    "item_type": "document",
                    "document": doc_info,
                    "message": f"ドキュメント '{doc_info['document_name']}' を取得しました"
                }

            # 画像として検索
            image_doc = self.img_vector_store.get_image_by_id(document_id)

            if image_doc:
                return {
                    "success": True,
                    "item_type": "image",
                    "image": image_doc.to_dict(),
                    "message": f"画像 '{image_doc.file_name}' を取得しました"
                }

            # どちらでも見つからなかった
            return {
                "success": False,
                "message": f"ID '{document_id}' のドキュメントまたは画像が見つかりませんでした",
                "error": "NotFound"
            }

        except VectorStoreError as e:
            error_msg = f"ドキュメント取得に失敗しました（ID: '{document_id}'）: {str(e)}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "error": "VectorStoreError"
            }
        except Exception as e:
            error_msg = f"ドキュメント取得に失敗しました（ID: '{document_id}'）: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "message": error_msg,
                "error": str(e)
            }

    def clear_documents(
        self,
        clear_text: bool = True,
        clear_images: bool = True
    ) -> dict[str, Any]:
        """すべてのドキュメントと画像を削除

        Args:
            clear_text: テキストドキュメントを削除するか（デフォルト: True）
            clear_images: 画像を削除するか（デフォルト: True）

        Returns:
            削除結果とメタデータを含む辞書
        """
        try:
            self.logger.warning(
                f"ドキュメント削除を開始 "
                f"(テキスト: {clear_text}, 画像: {clear_images})"
            )

            deleted_text_count = 0
            deleted_image_count = 0
            errors = []

            # テキストドキュメントの削除
            if clear_text:
                try:
                    # 削除前のカウント取得
                    text_docs = self.doc_vector_store.list_documents()
                    deleted_text_count = len(text_docs)

                    # 削除実行
                    self.doc_vector_store.clear()
                    success_msg = f"テキストドキュメント {deleted_text_count}件を削除しました"
                    self.logger.info(success_msg)
                except VectorStoreError as e:
                    error_msg = f"テキストドキュメントの削除エラー: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)

            # 画像の削除
            if clear_images:
                try:
                    # 削除前のカウント取得
                    images = self.img_vector_store.list_images()
                    deleted_image_count = len(images)

                    # 削除実行
                    self.img_vector_store.clear()
                    success_msg = f"画像 {deleted_image_count}件を削除しました"
                    self.logger.info(success_msg)
                except VectorStoreError as e:
                    error_msg = f"画像の削除エラー: {str(e)}"
                    self.logger.error(error_msg)
                    errors.append(error_msg)

            # 結果の生成
            total_deleted = deleted_text_count + deleted_image_count

            if errors:
                return {
                    "success": False,
                    "deleted_text_count": deleted_text_count,
                    "deleted_image_count": deleted_image_count,
                    "total_deleted": total_deleted,
                    "message": f"一部のドキュメント削除に失敗しました（削除済み: {total_deleted}件）",
                    "errors": errors
                }
            else:
                success_msg = (
                    f"すべてのドキュメントを削除しました "
                    f"（テキスト: {deleted_text_count}件, 画像: {deleted_image_count}件, 合計: {total_deleted}件）"
                )
                self.logger.info(success_msg)
                return {
                    "success": True,
                    "deleted_text_count": deleted_text_count,
                    "deleted_image_count": deleted_image_count,
                    "total_deleted": total_deleted,
                    "message": success_msg
                }

        except Exception as e:
            error_msg = f"ドキュメント削除に失敗しました: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "deleted_text_count": 0,
                "deleted_image_count": 0,
                "total_deleted": 0,
                "message": error_msg,
                "error": str(e)
            }
