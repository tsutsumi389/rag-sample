"""MCPリソースの定義。

RAG CLIのデータをMCPリソースとして公開します。
"""

import json
from typing import Any

from mcp.server import Server
from mcp.types import Resource

from .handlers import ResourceHandler

logger = None  # サーバー初期化時に設定


def register_resources(server: Server, handler: ResourceHandler):
    """MCPリソースをサーバーに登録します。

    Args:
        server: MCPサーバーインスタンス
        handler: リソースハンドラーインスタンス
    """

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """利用可能なリソース一覧を返します。

        Returns:
            リソース定義のリスト
        """
        return [
            Resource(
                uri="resource://documents/list",
                name="ドキュメント一覧",
                description="RAGシステムに登録されているすべてのドキュメント（テキストと画像）の一覧を取得します",
                mimeType="application/json"
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """リソースの内容を読み取ります。

        Args:
            uri: リソースURI

        Returns:
            リソースの内容（JSON文字列）
        """
        import logging
        logger = logging.getLogger(__name__)

        try:
            logger.info(f"read_resource called with URI: {uri} (type: {type(uri).__name__})")

            # ハンドラーでリソースを読み取り
            result = await handler.handle_resource_read(uri)

            # 結果をJSON形式で返す
            result_text = json.dumps(result, ensure_ascii=False, indent=2)

            logger.info(f"Successfully read resource: {uri}")
            return result_text

        except Exception as e:
            logger.error(f"Failed to read resource '{uri}': {e}", exc_info=True)
            error_result = {
                "success": False,
                "error": str(e),
                "message": f"リソース '{uri}' の読み取り中にエラーが発生しました"
            }
            error_text = json.dumps(error_result, ensure_ascii=False, indent=2)

            return error_text
