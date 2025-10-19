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

        # 埋め込み生成器の初期化（検索用）
        from ..rag.embeddings import EmbeddingGenerator
        self.embedding_generator = EmbeddingGenerator(self.config)

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
        elif name == "search":
            return await self._search(**arguments)
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
