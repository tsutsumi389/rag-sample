"""SSE（Server-Sent Events）トランスポートを使用したMCPサーバー。

このモジュールは、HTTP/SSEプロトコル経由でMCPクライアントと通信するサーバーを提供します。
Claude DesktopなどのリモートMCPクライアントから、RAG機能をツールとして利用できます。
"""

import asyncio
import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from mcp.server import Server
from mcp.types import JSONRPCMessage

from .tools import register_tools
from .resources import register_resources
from .handlers import ToolHandler, ResourceHandler


# ロガー設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_sse_server")


# FastAPIアプリケーション
app = FastAPI(title="RAG MCP Server (SSE)")


# MCPサーバーインスタンス
mcp_server: Server | None = None
tool_handler: ToolHandler | None = None
resource_handler: ResourceHandler | None = None


# セッション管理（簡易実装）
sessions: dict[str, dict[str, Any]] = {}


def initialize_mcp_server() -> Server:
    """MCPサーバーを初期化します。

    Returns:
        Server: 初期化されたMCPサーバーインスタンス
    """
    global mcp_server, tool_handler, resource_handler

    if mcp_server is not None:
        return mcp_server

    logger.info("MCPサーバーを初期化中...")

    # MCPサーバー初期化
    mcp_server = Server("rag-mcp-server")
    logger.info("MCPサーバーを初期化しました")

    # ツールハンドラー初期化
    tool_handler = ToolHandler()
    logger.info("ツールハンドラーを初期化しました")

    # リソースハンドラー初期化
    resource_handler = ResourceHandler()
    logger.info("リソースハンドラーを初期化しました")

    # ツールを登録
    register_tools(mcp_server, tool_handler)
    logger.info("ツールを登録しました")

    # リソースを登録
    register_resources(mcp_server, resource_handler)
    logger.info("リソースを登録しました")

    return mcp_server


@app.on_event("startup")
async def startup_event():
    """アプリケーション起動時の処理。"""
    logger.info("RAG MCP Server (SSE) を起動中...")
    initialize_mcp_server()
    logger.info("RAG MCP Server (SSE) is ready")


@app.get("/")
async def root():
    """ルートエンドポイント。

    Returns:
        dict: サーバー情報
    """
    return {
        "name": "RAG MCP Server",
        "version": "0.1.0",
        "transport": "SSE",
        "endpoints": {
            "sse": "/sse",
            "messages": "/messages"
        }
    }


@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント。

    Returns:
        dict: サーバーの状態
    """
    return {
        "status": "healthy",
        "server": "rag-mcp-server",
        "transport": "sse"
    }


@app.get("/sse")
async def sse_endpoint(request: Request):
    """SSE接続エンドポイント。

    クライアントはこのエンドポイントに接続し、サーバーからのイベントを受信します。
    初回接続時にセッションIDを発行し、メッセージエンドポイントのURLを返します。

    Args:
        request: FastAPIリクエストオブジェクト

    Returns:
        EventSourceResponse: SSEレスポンス
    """
    # セッションID生成
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "created_at": asyncio.get_event_loop().time(),
        "active": True
    }

    logger.info(f"新しいSSE接続: session_id={session_id}")

    async def event_generator():
        """SSEイベントを生成するジェネレーター。"""
        try:
            # 初回接続時にメッセージエンドポイントのURLを送信
            yield {
                "event": "endpoint",
                "data": f"/messages?session_id={session_id}"
            }

            # 接続維持のためのpingイベント（30秒ごと）
            while sessions.get(session_id, {}).get("active", False):
                await asyncio.sleep(30)

                # クライアントの切断を検出
                if await request.is_disconnected():
                    logger.info(f"クライアントが切断されました: session_id={session_id}")
                    break

                yield {
                    "event": "ping",
                    "data": "keepalive"
                }

        except asyncio.CancelledError:
            logger.info(f"SSE接続がキャンセルされました: session_id={session_id}")

        finally:
            # セッションをクリーンアップ
            if session_id in sessions:
                sessions[session_id]["active"] = False
                logger.info(f"セッションを終了: session_id={session_id}")

    return EventSourceResponse(event_generator())


@app.post("/messages")
async def messages_endpoint(request: Request):
    """メッセージ処理エンドポイント。

    クライアントからのJSON-RPCメッセージを受信し、MCPサーバーで処理します。

    Args:
        request: FastAPIリクエストオブジェクト

    Returns:
        JSONResponse: JSON-RPCレスポンス
    """
    try:
        # リクエストボディを取得
        body = await request.json()
        logger.info(f"受信メッセージ: {body}")

        # セッションIDの検証（オプション）
        session_id = request.query_params.get("session_id")
        if session_id and session_id not in sessions:
            logger.warning(f"無効なセッションID: {session_id}")
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid session"
                    },
                    "id": body.get("id")
                }
            )

        # MCPサーバーでメッセージを処理
        if mcp_server is None:
            initialize_mcp_server()

        # JSON-RPCメッセージとして処理
        # Note: この部分は実際のMCP SDKの実装に応じて調整が必要
        response = await handle_jsonrpc_message(body)

        logger.info(f"送信レスポンス: {response}")
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"メッセージ処理エラー: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                },
                "id": body.get("id") if "body" in locals() else None
            }
        )


async def handle_jsonrpc_message(message: dict[str, Any]) -> dict[str, Any]:
    """JSON-RPCメッセージを処理します。

    Args:
        message: JSON-RPCメッセージ

    Returns:
        dict: JSON-RPCレスポンス
    """
    method = message.get("method")
    params = message.get("params", {})
    msg_id = message.get("id")

    if method == "initialize":
        # 初期化リクエスト
        return {
            "jsonrpc": "2.0",
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "rag-mcp-server",
                    "version": "0.1.0"
                },
                "capabilities": {
                    "tools": {},
                    "resources": {}
                }
            },
            "id": msg_id
        }

    elif method == "tools/list":
        # ツール一覧リクエスト
        if tool_handler is None:
            return create_error_response(msg_id, -32603, "Tool handler not initialized")

        # 実際のツール一覧を取得（handlers.pyのToolHandlerから）
        tools_list = [
            {
                "name": "query_documents",
                "description": "ドキュメントに対して質問を投げ、関連する情報を検索して回答を生成します",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "検索する質問文"
                        }
                    },
                    "required": ["question"]
                }
            },
            {
                "name": "search_documents",
                "description": "ドキュメントから関連する情報を検索します（回答生成なし）",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "検索クエリ"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "取得する結果の最大数",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "add_documents",
                "description": "指定されたパスからドキュメントをベクトルデータベースに追加します",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "追加するドキュメントのファイルパスまたはディレクトリパス"
                        }
                    },
                    "required": ["path"]
                }
            },
            {
                "name": "list_documents",
                "description": "ベクトルデータベースに保存されているドキュメントの一覧を取得します",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]

        return {
            "jsonrpc": "2.0",
            "result": {
                "tools": tools_list
            },
            "id": msg_id
        }

    elif method == "tools/call":
        # ツール呼び出しリクエスト
        if tool_handler is None:
            return create_error_response(msg_id, -32603, "Tool handler not initialized")

        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        try:
            # ツールハンドラーで実行
            result = await tool_handler.handle_tool_call(tool_name, tool_args)

            return {
                "jsonrpc": "2.0",
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": str(result)
                        }
                    ]
                },
                "id": msg_id
            }

        except Exception as e:
            logger.error(f"ツール実行エラー: {e}", exc_info=True)
            return create_error_response(msg_id, -32603, f"Tool execution failed: {str(e)}")

    elif method == "resources/list":
        # リソース一覧リクエスト
        return {
            "jsonrpc": "2.0",
            "result": {
                "resources": []
            },
            "id": msg_id
        }

    else:
        # 未知のメソッド
        return create_error_response(msg_id, -32601, f"Method not found: {method}")


def create_error_response(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    """JSON-RPCエラーレスポンスを作成します。

    Args:
        msg_id: メッセージID
        code: エラーコード
        message: エラーメッセージ

    Returns:
        dict: JSON-RPCエラーレスポンス
    """
    return {
        "jsonrpc": "2.0",
        "error": {
            "code": code,
            "message": message
        },
        "id": msg_id
    }


def run():
    """SSE MCPサーバーを起動します（CLIエントリーポイント用）。

    デフォルトでは localhost:8000 でリッスンします。
    環境変数で設定可能:
    - MCP_SSE_HOST: ホスト名（デフォルト: 127.0.0.1）
    - MCP_SSE_PORT: ポート番号（デフォルト: 8000）
    """
    import os
    import uvicorn

    host = os.getenv("MCP_SSE_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_SSE_PORT", "8000"))

    logger.info(f"SSE MCPサーバーを起動: http://{host}:{port}")

    try:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info"
        )
    except KeyboardInterrupt:
        logger.info("MCPサーバーを停止しました")
    except Exception as e:
        logger.error(f"エラー: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    run()
