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
                name="add_document",
                description="テキストまたは画像ドキュメントをRAGシステムに追加します。ファイルパスを指定して、テキストドキュメントまたは画像を登録できます。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "追加するファイルまたはディレクトリのパス"
                        },
                        "caption": {
                            "type": "string",
                            "description": "画像の場合のキャプション（オプション、画像ファイルのみ）"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "画像に付与するタグのリスト（オプション、画像ファイルのみ）"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
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
            Tool(
                name="search",
                description="キーワードでドキュメントを検索します。指定されたクエリに類似したドキュメントチャンクを検索し、類似度スコアと共に返します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "検索クエリ文字列"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返す検索結果の最大数（デフォルト: 5）",
                            "minimum": 1,
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            Tool(
                name="remove_document",
                description="ドキュメントまたは画像をIDで削除します。テキストドキュメントIDまたは画像IDを指定して、RAGシステムから削除できます。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "item_id": {
                            "type": "string",
                            "description": "削除するドキュメントIDまたは画像ID"
                        },
                        "item_type": {
                            "type": "string",
                            "description": "削除するアイテムのタイプ（'document' または 'image'）。省略時は自動判定します。",
                            "enum": ["document", "image", "auto"],
                            "default": "auto"
                        }
                    },
                    "required": ["item_id"]
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
