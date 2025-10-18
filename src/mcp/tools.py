"""MCPツールの定義。

RAG CLIの主要機能をMCPツールとして公開します。
"""

import json
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from .handlers import ToolHandler

logger = None  # サーバー初期化時に設定


def register_tools(server: Server, handler: ToolHandler):
    """MCPツールをサーバーに登録します。

    Args:
        server: MCPサーバーインスタンス
        handler: ツールハンドラーインスタンス
    """

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """利用可能なツール一覧を返します。

        Returns:
            ツール定義のリスト
        """
        return [
            Tool(
                name="list_documents",
                description="RAGシステムに登録されているドキュメント一覧を取得します。テキストドキュメントと画像の両方を含みます。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "返すドキュメント数の上限（省略時は全件取得）",
                            "minimum": 1
                        },
                        "include_images": {
                            "type": "boolean",
                            "description": "画像も含めるかどうか（デフォルト: true）",
                            "default": True
                        }
                    }
                }
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """ツール呼び出しを処理します。

        Args:
            name: ツール名
            arguments: ツール引数

        Returns:
            実行結果のテキストコンテンツリスト
        """
        try:
            # ハンドラーでツールを実行
            result = await handler.handle_tool_call(name, arguments)

            # 結果をJSON形式で返す
            result_text = json.dumps(result, ensure_ascii=False, indent=2)

            return [TextContent(
                type="text",
                text=result_text
            )]

        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "message": f"ツール '{name}' の実行中にエラーが発生しました"
            }
            error_text = json.dumps(error_result, ensure_ascii=False, indent=2)

            return [TextContent(
                type="text",
                text=error_text
            )]
