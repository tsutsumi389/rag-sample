"""MCPリクエストハンドラー。

MCPツール・リソースの呼び出しを実際のRAGコアロジックに橋渡しします。
"""

import logging
from pathlib import Path
from typing import Any

from ..rag.vector_store import VectorStore, VectorStoreError
from ..rag.document_processor import DocumentProcessor, DocumentProcessorError, UnsupportedFileTypeError
from ..rag.image_processor import ImageProcessor, ImageProcessorError
from ..rag.vision_embeddings import VisionEmbeddings, VisionEmbeddingError
from ..utils.config import get_config

logger = logging.getLogger(__name__)

# 画像ファイル拡張子の定義
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


def _is_image_file(file_path: str) -> bool:
    """ファイルが画像かどうかを拡張子から判定

    Args:
        file_path: チェックするファイルのパス

    Returns:
        画像ファイルの場合True
    """
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


class ToolHandler:
    """MCPツール呼び出しハンドラー。

    MCPクライアントからのツール呼び出しを受け取り、
    RAGコアの機能を実行して結果を返します。
    """

    def __init__(self):
        """初期化。

        アプリケーション設定を読み込み、RAGコンポーネントを初期化します。
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # VectorStoreの初期化（ドキュメント用）
        self.doc_vector_store = VectorStore(
            config=self.config,
            collection_name="documents"
        )
        self.doc_vector_store.initialize()

        # VectorStoreの初期化（画像用）
        self.img_vector_store = VectorStore(
            config=self.config,
            collection_name="images"
        )
        self.img_vector_store.initialize()

        # 埋め込み生成器の初期化（検索用）
        from ..rag.embeddings import EmbeddingGenerator
        self.embedding_generator = EmbeddingGenerator(self.config)

        # ドキュメントプロセッサの初期化（ドキュメント追加用）
        self.document_processor = DocumentProcessor(self.config)

        # 画像処理用コンポーネントの初期化（画像追加用）
        self.vision_embeddings = VisionEmbeddings(self.config)
        self.image_processor = ImageProcessor(self.vision_embeddings, self.config)

    async def handle_tool_call(self, name: str, arguments: dict) -> dict[str, Any]:
        """ツール呼び出しを処理します。

        Args:
            name: ツール名
            arguments: ツール引数

        Returns:
            実行結果の辞書

        Raises:
            ValueError: 未知のツール名の場合
        """
        self.logger.info(f"ツール呼び出し: {name}, 引数: {arguments}")

        if name == "add_document":
            return await self._add_document(**arguments)
        elif name == "list_documents":
            return await self._list_documents(**arguments)
        elif name == "search":
            return await self._search(**arguments)
        elif name == "search_images":
            return await self._search_images(**arguments)
        elif name == "remove_document":
            return await self._remove_document(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _add_document(
        self,
        file_path: str,
        caption: str | None = None,
        tags: list[str] | None = None
    ) -> dict[str, Any]:
        """ドキュメントまたは画像追加の実装。

        Args:
            file_path: 追加するファイルまたはディレクトリのパス
            caption: 画像の場合のキャプション（オプション、画像ファイルのみ）
            tags: 画像に付与するタグのリスト（オプション、画像ファイルのみ）

        Returns:
            追加結果とメタデータを含む辞書
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
            if _is_image_file(file_path):
                return await self._add_image_file(file_path, caption, tags)
            else:
                return await self._add_document_file(file_path)

        except Exception as e:
            error_msg = f"ドキュメント追加に失敗しました: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "message": error_msg,
                "error": str(e)
            }

    async def _add_document_file(self, file_path: str) -> dict[str, Any]:
        """テキストドキュメントファイルを追加

        Args:
            file_path: 追加するファイルのパス

        Returns:
            追加結果
        """
        try:
            self.logger.info(f"テキストドキュメントを追加中: {file_path}")

            # ドキュメントの処理
            document, chunks = self.document_processor.process_document(file_path)

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

        except UnsupportedFileTypeError as e:
            error_msg = f"サポートされていないファイル形式です: {str(e)}"
            self.logger.warning(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "error": "UnsupportedFileType"
            }

        except DocumentProcessorError as e:
            error_msg = f"ドキュメント処理エラー: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "message": error_msg,
                "error": "DocumentProcessorError"
            }

        except Exception as e:
            error_msg = f"テキストドキュメント追加に失敗しました: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "message": error_msg,
                "error": str(e)
            }

    async def _add_image_file(
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
            追加結果
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

        except ImageProcessorError as e:
            error_msg = f"画像処理エラー: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "message": error_msg,
                "error": "ImageProcessorError"
            }

        except VisionEmbeddingError as e:
            error_msg = f"ビジョン埋め込みエラー: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "message": error_msg,
                "error": "VisionEmbeddingError",
                "hint": "Ollamaが起動していること、ビジョンモデル（llava等）がインストールされていることを確認してください"
            }

        except Exception as e:
            error_msg = f"画像追加に失敗しました: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "message": error_msg,
                "error": str(e)
            }

    async def _list_documents(
        self,
        limit: int | None = None,
        include_images: bool = True
    ) -> dict[str, Any]:
        """ドキュメント一覧取得の実装。

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
                result["message"] = f"合計 {total}件のドキュメントを取得しました（テキスト: {len(result['documents'])}件, 画像: {len(result['images'])}件）"

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

    async def _search(
        self,
        query: str,
        top_k: int = 5
    ) -> dict[str, Any]:
        """ドキュメント検索の実装。

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

    async def _search_images(
        self,
        query: str,
        top_k: int = 5
    ) -> dict[str, Any]:
        """画像検索の実装。

        テキストクエリを使用して、意味的に類似した画像を検索します。
        画像のキャプション（説明文）に基づいて検索を行います。

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

    async def _remove_document(
        self,
        item_id: str,
        item_type: str = "auto"
    ) -> dict[str, Any]:
        """ドキュメントまたは画像削除の実装。

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


class ResourceHandler:
    """MCPリソース読み取りハンドラー。

    MCPクライアントからのリソース読み取り要求を処理します。
    """

    def __init__(self):
        """初期化。

        アプリケーション設定を読み込みます。
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # VectorStoreの初期化（ドキュメント用）
        self.doc_vector_store = VectorStore(
            config=self.config,
            collection_name="documents"
        )
        self.doc_vector_store.initialize()

        # VectorStoreの初期化（画像用）
        self.img_vector_store = VectorStore(
            config=self.config,
            collection_name="images"
        )
        self.img_vector_store.initialize()

    async def handle_resource_read(self, uri: str) -> dict[str, Any]:
        """リソース読み取りを処理します。

        Args:
            uri: リソースURI

        Returns:
            リソースの内容

        Raises:
            ValueError: 未知のリソースURIの場合
        """
        self.logger.info(f"リソース読み取り: {uri} (型: {type(uri).__name__})")

        # URIを文字列に変換（AnyUrl型の可能性があるため）
        uri_str = str(uri)
        self.logger.debug(f"URIを文字列化: '{uri_str}'")

        if uri_str == "resource://documents/list":
            return await self._get_documents_list()
        else:
            raise ValueError(f"Unknown resource: {uri_str}")

    async def _get_documents_list(self) -> dict[str, Any]:
        """ドキュメント一覧取得の実装。

        テキストドキュメントと画像の両方を取得します。

        Returns:
            ドキュメント一覧（テキストと画像を含む）
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
                text_docs = self.doc_vector_store.list_documents()
                result["documents"] = text_docs
                self.logger.info(f"テキストドキュメント: {len(text_docs)}件")
            except VectorStoreError as e:
                self.logger.warning(f"テキストドキュメント取得エラー: {e}")
                result["documents"] = []

            # 画像ドキュメント取得
            try:
                image_docs = self.img_vector_store.list_images()
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
            error_msg = f"ドキュメント一覧の取得に失敗しました: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "documents": [],
                "images": [],
                "total_count": 0,
                "message": error_msg,
                "error": str(e)
            }
