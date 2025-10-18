"""MCPリクエストハンドラー。

MCPツール・リソースの呼び出しを実際のRAGコアロジックに橋渡しします。
"""

import json
import logging
from typing import Any

from ..rag.vector_store import VectorStore, VectorStoreError
from ..utils.config import get_config

logger = logging.getLogger(__name__)


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

        if name == "list_documents":
            return await self._list_documents(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

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
                    image_docs = self.img_vector_store.list_documents(limit=limit)
                    result["images"] = image_docs
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

        # VectorStoreの初期化
        self.doc_vector_store = VectorStore(
            config=self.config,
            collection_name="documents"
        )
        self.doc_vector_store.initialize()

    async def handle_resource_read(self, uri: str) -> dict[str, Any]:
        """リソース読み取りを処理します。

        Args:
            uri: リソースURI

        Returns:
            リソースの内容

        Raises:
            ValueError: 未知のリソースURIの場合
        """
        self.logger.info(f"リソース読み取り: {uri}")

        if uri == "resource://documents/list":
            return await self._get_documents_list()
        else:
            raise ValueError(f"Unknown resource: {uri}")

    async def _get_documents_list(self) -> dict[str, Any]:
        """ドキュメント一覧取得の実装。

        Returns:
            ドキュメント一覧
        """
        try:
            documents = self.doc_vector_store.list_documents()
            return {
                "success": True,
                "documents": documents,
                "count": len(documents)
            }
        except Exception as e:
            error_msg = f"ドキュメント一覧の取得に失敗しました: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "documents": [],
                "count": 0,
                "error": str(e)
            }
