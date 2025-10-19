"""MCPサーバーのメインエントリーポイント（stdio トランスポート）。

このモジュールは、RAG CLIアプリケーションのMCPサーバー実装を提供します。
stdioプロトコル経由でMCPクライアントと通信します。
Claude DesktopなどのローカルMCPクライアントから、RAG機能をツールとして利用できます。

Note: リモート接続が必要な場合は、server_sse.py（SSEトランスポート）を使用してください。
"""

import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .tools import register_tools
from .resources import register_resources
from .handlers import ToolHandler, ResourceHandler


async def main():
    """MCPサーバーを起動します。

    stdio経由でMCPクライアントと通信し、
    RAG機能をツールとして提供します。
    """
    # ロガー設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("mcp_server")
    logger.info("RAG MCP Serverを起動中...")

    try:
        # MCPサーバー初期化
        server = Server("rag-mcp-server")
        logger.info("MCPサーバーを初期化しました")

        # ツールハンドラー初期化
        tool_handler = ToolHandler()
        logger.info("ツールハンドラーを初期化しました")

        # リソースハンドラー初期化
        resource_handler = ResourceHandler()
        logger.info("リソースハンドラーを初期化しました")

        # ツールを登録
        register_tools(server, tool_handler)
        logger.info("ツールを登録しました")

        # リソースを登録
        register_resources(server, resource_handler)
        logger.info("リソースを登録しました")

        # サーバー起動（stdio経由で通信）
        logger.info("RAG MCP Server is ready - stdio通信を開始します")

        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )

    except Exception as e:
        logger.error(f"MCPサーバーの起動に失敗しました: {e}", exc_info=True)
        raise


def run():
    """同期的にMCPサーバーを起動します（CLIエントリーポイント用）。

    pyproject.tomlのscriptsセクションから呼び出されます。
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nMCPサーバーを停止しました")
    except Exception as e:
        print(f"エラー: {e}")
        raise


if __name__ == "__main__":
    run()
