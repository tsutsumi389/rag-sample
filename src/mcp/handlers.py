"""MCPリクエストハンドラー。

MCPツール・リソースの呼び出しを実際のRAGコアロジックに橋渡しします。
"""

import logging
from typing import Any

from ..services.document_service import DocumentService
from ..utils.config import get_config

logger = logging.getLogger(__name__)


class ToolHandler:
    """MCPツール呼び出しハンドラー。

    MCPクライアントからのツール呼び出しを受け取り、
    RAGコアの機能を実行して結果を返します。
    """

    def __init__(self):
        """初期化。

        アプリケーション設定を読み込み、ドキュメントサービスを初期化します。
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # ドキュメントサービスの初期化（すべてのビジネスロジックを提供）
        self.document_service = DocumentService(self.config)

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
        elif name == "clear_documents":
            return await self._clear_documents(**arguments)
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
        # DocumentServiceに処理を委譲
        return self.document_service.add_file(
            file_path=file_path,
            caption=caption,
            tags=tags
        )


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
        # DocumentServiceに処理を委譲
        return self.document_service.list_documents(
            limit=limit,
            include_images=include_images
        )

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
        # DocumentServiceに処理を委譲
        return self.document_service.search_documents(
            query=query,
            top_k=top_k
        )

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
        # DocumentServiceに処理を委譲
        return self.document_service.search_images(
            query=query,
            top_k=top_k
        )

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
        # DocumentServiceに処理を委譲
        return self.document_service.remove_document(
            item_id=item_id,
            item_type=item_type
        )

    async def _clear_documents(
        self,
        clear_text: bool = True,
        clear_images: bool = True
    ) -> dict[str, Any]:
        """すべてのドキュメントと画像を削除する実装。

        警告: この操作は取り消せません。

        Args:
            clear_text: テキストドキュメントを削除するか（デフォルト: True）
            clear_images: 画像を削除するか（デフォルト: True）

        Returns:
            削除結果とメタデータを含む辞書
        """
        # DocumentServiceに処理を委譲
        return self.document_service.clear_documents(
            clear_text=clear_text,
            clear_images=clear_images
        )


class ResourceHandler:
    """MCPリソース読み取りハンドラー。

    MCPクライアントからのリソース読み取り要求を処理します。
    """

    def __init__(self):
        """初期化。

        アプリケーション設定を読み込み、ドキュメントサービスを初期化します。
        """
        self.config = get_config()
        self.logger = logging.getLogger(__name__)

        # ドキュメントサービスの初期化
        self.document_service = DocumentService(self.config)

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
        elif uri_str.startswith("resource://documents/"):
            # resource://documents/{id} の形式
            document_id = uri_str.replace("resource://documents/", "")
            return await self._get_document_by_id(document_id)
        else:
            raise ValueError(f"Unknown resource: {uri_str}")

    async def _get_documents_list(self) -> dict[str, Any]:
        """ドキュメント一覧取得の実装。

        テキストドキュメントと画像の両方を取得します。

        Returns:
            ドキュメント一覧（テキストと画像を含む）
        """
        # DocumentServiceに処理を委譲
        return self.document_service.list_documents(
            limit=None,
            include_images=True
        )

    async def _get_document_by_id(self, document_id: str) -> dict[str, Any]:
        """ドキュメントID指定での取得実装。

        Args:
            document_id: ドキュメントID

        Returns:
            ドキュメント詳細情報
        """
        # DocumentServiceに処理を委譲
        return self.document_service.get_document_by_id(document_id)
