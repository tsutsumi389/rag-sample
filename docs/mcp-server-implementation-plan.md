# MCP（Model Context Protocol）サーバー対応 実装計画

## 概要

このドキュメントは、RAG CLIアプリケーションをMCPサーバーに対応させるための実装計画です。MCPサーバー化により、Claude DesktopなどのMCPクライアントから、RAGシステムの機能をツールとして呼び出せるようになります。

## MCP（Model Context Protocol）とは

MCPは、LLMアプリケーション（Claude Desktop等）がローカルまたはリモートのデータソースやツールと連携するためのオープンプロトコルです。

**主な特徴:**
- LLMに外部コンテキスト（ローカルファイル、データベース等）を提供
- ツール（関数）をLLMから呼び出し可能にする
- プロンプトテンプレートの提供
- リソース（ドキュメント等）の動的取得

**参考:**
- [MCP公式ドキュメント](https://modelcontextprotocol.io/)
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk)

## 実装目標

### Phase 1: 基本的なMCPサーバー機能（必須）

1. **ツール（Tools）の提供**
   - RAG CLI の主要機能をMCPツールとして公開
   - Claude Desktopから各種操作を実行可能に

2. **リソース（Resources）の提供**
   - 登録済みドキュメントの一覧取得
   - 個別ドキュメントの内容取得

### Phase 2: 高度な機能（オプション）

1. **プロンプト（Prompts）の提供**
   - RAG検索用の定型プロンプトテンプレート
   - マルチモーダル検索用プロンプト

2. **サンプリング（Sampling）機能**
   - サーバー側でLLM推論を実行
   - 結果をクライアントに返す

## アーキテクチャ設計

### ディレクトリ構造（追加分）

```
rag-sample/
├── src/
│   ├── mcp/                         # MCP関連モジュール（新規）
│   │   ├── __init__.py
│   │   ├── server.py                # MCPサーバーのメインエントリーポイント
│   │   ├── tools.py                 # MCPツールの定義と実装
│   │   ├── resources.py             # MCPリソースの定義と実装
│   │   ├── prompts.py               # MCPプロンプトの定義（Phase 2）
│   │   └── handlers.py              # リクエストハンドラー（ビジネスロジック呼び出し）
│   ├── cli.py                       # 既存CLI（変更なし）
│   ├── commands/                    # 既存コマンド（変更なし）
│   ├── rag/                         # 既存RAGコア（変更なし）
│   ├── models/                      # 既存モデル（変更なし）
│   └── utils/                       # 既存ユーティリティ（変更なし）
├── tests/
│   └── mcp/                         # MCPテスト（新規）
│       ├── test_server.py
│       ├── test_tools.py
│       ├── test_resources.py
│       └── test_handlers.py
├── docs/
│   └── mcp-server-implementation-plan.md  # このドキュメント
├── .env.sample                      # 環境変数にMCP設定追加
└── pyproject.toml                   # MCPサーバー用スクリプト追加
```

### 技術スタック

**追加する依存関係:**
- `mcp>=1.1.0` - MCP Python SDK
- `httpx>=0.27.0` - 非同期HTTPクライアント（MCP SDK依存）
- `pydantic>=2.0.0` - データバリデーション（MCP SDK依存）

**既存技術スタックとの統合:**
- 既存のRAGコア（`src/rag/`）をそのまま利用
- CLIとMCPサーバーは独立して動作（コード共有）
- ChromaDB、Ollama、LangChainは変更なし

### MCPサーバーとCLIの関係

```
┌─────────────────────────────────────────────────────────┐
│                    RAG Application                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐              ┌──────────────┐        │
│  │  CLI Layer   │              │  MCP Server  │        │
│  │  (src/cli.py)│              │ (src/mcp/)   │        │
│  └──────┬───────┘              └──────┬───────┘        │
│         │                              │                │
│         │        ┌─────────────────────┘                │
│         │        │                                      │
│         ▼        ▼                                      │
│  ┌──────────────────────┐                              │
│  │   RAG Core Layer     │                              │
│  │   (src/rag/)         │                              │
│  │   - vector_store     │                              │
│  │   - engine           │                              │
│  │   - embeddings       │                              │
│  │   - processors       │                              │
│  └──────────────────────┘                              │
│                                                         │
└─────────────────────────────────────────────────────────┘

両方とも同じRAGコアを利用
→ ビジネスロジックの重複なし
→ 保守性・一貫性の確保
```

## Phase 1 詳細設計

### 1. MCPツール（Tools）定義

MCPツールとして以下の機能を公開します：

#### 1.1 ドキュメント管理ツール

| ツール名 | 説明 | パラメータ | 戻り値 |
|---------|------|-----------|-------|
| `add_document` | テキストまたは画像ドキュメントを追加 | `file_path: str`<br>`caption: str \| None`<br>`tags: list[str] \| None` | `{"success": bool, "document_id": str, "message": str}` |
| `remove_document` | ドキュメントを削除 | `document_id: str` | `{"success": bool, "message": str}` |
| `list_documents` | 登録済みドキュメント一覧を取得 | なし | `{"documents": list[dict]}` |
| `clear_documents` | 全ドキュメントをクリア | `confirm: bool` | `{"success": bool, "message": str}` |

#### 1.2 検索・質問ツール

| ツール名 | 説明 | パラメータ | 戻り値 |
|---------|------|-----------|-------|
| `query` | RAGで質問に回答 | `question: str`<br>`chat_history: list[dict] \| None` | `{"answer": str, "sources": list[dict]}` |
| `search` | キーワードでドキュメント検索 | `keyword: str`<br>`top_k: int` | `{"results": list[dict]}` |
| `search_images` | テキストクエリで画像検索 | `query: str`<br>`top_k: int` | `{"results": list[dict]}` |
| `search_multimodal` | マルチモーダル検索 | `query: str`<br>`top_k: int`<br>`text_weight: float`<br>`image_weight: float` | `{"results": list[dict]}` |

#### 1.3 設定・管理ツール

| ツール名 | 説明 | パラメータ | 戻り値 |
|---------|------|-----------|-------|
| `get_status` | システムステータス確認 | なし | `{"ollama_status": dict, "chroma_status": dict, "document_count": int}` |
| `get_config` | 現在の設定を取得 | なし | `{"config": dict}` |

### 2. MCPリソース（Resources）定義

リソースは `resource://` URIスキームで公開します：

| リソースURI | 説明 | MIME Type | 内容 |
|-----------|------|-----------|------|
| `resource://documents/list` | 全ドキュメント一覧 | `application/json` | ドキュメントメタデータの配列 |
| `resource://documents/{id}` | 特定ドキュメントの内容 | `text/plain` or `application/json` | ドキュメントの全文 |
| `resource://images/list` | 全画像一覧 | `application/json` | 画像メタデータの配列 |
| `resource://images/{id}` | 特定画像の情報 | `application/json` | 画像パス、キャプション、タグ |
| `resource://system/config` | システム設定 | `application/json` | 現在の環境設定 |
| `resource://system/status` | システムステータス | `application/json` | 接続状態、ドキュメント数 |

### 3. 実装詳細

#### 3.1 MCPサーバーのメインエントリーポイント（`src/mcp/server.py`）

```python
"""MCPサーバーのメインエントリーポイント。

このモジュールは、RAG CLIアプリケーションのMCPサーバー実装を提供します。
Claude DesktopなどのMCPクライアントから、RAG機能をツールとして利用できます。
"""

import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, Resource

from .tools import register_tools
from .resources import register_resources
from .handlers import ToolHandler, ResourceHandler
from ..utils.config import load_config
from ..utils.logger import setup_logger


async def main():
    """MCPサーバーを起動します。"""
    # ロガー設定
    logger = setup_logger(__name__)
    logger.info("Starting RAG MCP Server...")

    # 設定読み込み
    config = load_config()

    # MCPサーバー初期化
    server = Server("rag-mcp-server")

    # ツールとリソースを登録
    tool_handler = ToolHandler(config)
    resource_handler = ResourceHandler(config)

    register_tools(server, tool_handler)
    register_resources(server, resource_handler)

    # サーバー起動（stdio経由で通信）
    logger.info("RAG MCP Server is ready")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def run():
    """同期的にMCPサーバーを起動します（CLIエントリーポイント用）。"""
    asyncio.run(main())


if __name__ == "__main__":
    run()
```

#### 3.2 ツール定義（`src/mcp/tools.py`）

```python
"""MCPツールの定義。

RAG CLIの主要機能をMCPツールとして公開します。
"""

from mcp.server import Server
from mcp.types import Tool, TextContent

from .handlers import ToolHandler


def register_tools(server: Server, handler: ToolHandler):
    """MCPツールをサーバーに登録します。

    Args:
        server: MCPサーバーインスタンス
        handler: ツールハンドラーインスタンス
    """

    # ドキュメント追加ツール
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="add_document",
                description="テキストまたは画像ドキュメントをRAGシステムに追加します",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "追加するファイルまたはディレクトリのパス"
                        },
                        "caption": {
                            "type": "string",
                            "description": "画像の場合のキャプション（オプション）"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "画像に付与するタグ（オプション）"
                        }
                    },
                    "required": ["file_path"]
                }
            ),
            Tool(
                name="query",
                description="RAGを使って質問に回答します",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "質問内容"
                        },
                        "chat_history": {
                            "type": "array",
                            "description": "会話履歴（オプション）",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": "string"}
                                }
                            }
                        }
                    },
                    "required": ["question"]
                }
            ),
            # 他のツールも同様に定義...
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """ツール呼び出しを処理します。"""
        result = await handler.handle_tool_call(name, arguments)
        return [TextContent(type="text", text=str(result))]
```

#### 3.3 リソース定義（`src/mcp/resources.py`）

```python
"""MCPリソースの定義。

登録済みドキュメントやシステム情報をリソースとして公開します。
"""

import json
from mcp.server import Server
from mcp.types import Resource, TextContent

from .handlers import ResourceHandler


def register_resources(server: Server, handler: ResourceHandler):
    """MCPリソースをサーバーに登録します。

    Args:
        server: MCPサーバーインスタンス
        handler: リソースハンドラーインスタンス
    """

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(
                uri="resource://documents/list",
                name="ドキュメント一覧",
                description="登録済みドキュメントの一覧",
                mimeType="application/json"
            ),
            Resource(
                uri="resource://system/status",
                name="システムステータス",
                description="RAGシステムの現在の状態",
                mimeType="application/json"
            ),
            # 他のリソースも同様に定義...
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> TextContent:
        """リソース読み取りを処理します。"""
        content = await handler.handle_resource_read(uri)
        return TextContent(
            type="text",
            text=json.dumps(content, ensure_ascii=False, indent=2)
        )
```

#### 3.4 ハンドラー実装（`src/mcp/handlers.py`）

```python
"""MCPリクエストハンドラー。

MCPツール・リソースの呼び出しを実際のRAGコアロジックに橋渡しします。
"""

import logging
from typing import Any

from ..rag.vector_store import VectorStore
from ..rag.engine import RAGEngine
from ..rag.embeddings import OllamaEmbeddings
from ..rag.document_processor import DocumentProcessor
from ..rag.image_processor import ImageProcessor
from ..utils.config import Config


class ToolHandler:
    """MCPツール呼び出しハンドラー。"""

    def __init__(self, config: Config):
        """初期化。

        Args:
            config: アプリケーション設定
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # RAGコンポーネント初期化
        self.vector_store = VectorStore(config.chroma_persist_directory)
        self.embeddings = OllamaEmbeddings(
            base_url=config.ollama_base_url,
            model=config.ollama_embedding_model
        )
        self.engine = RAGEngine(self.vector_store, self.embeddings, config)
        self.doc_processor = DocumentProcessor(config)
        self.img_processor = ImageProcessor(config)

    async def handle_tool_call(self, name: str, arguments: dict) -> dict[str, Any]:
        """ツール呼び出しを処理します。

        Args:
            name: ツール名
            arguments: ツール引数

        Returns:
            実行結果
        """
        if name == "add_document":
            return await self._add_document(**arguments)
        elif name == "query":
            return await self._query(**arguments)
        elif name == "search":
            return await self._search(**arguments)
        # 他のツールも同様に処理...
        else:
            raise ValueError(f"Unknown tool: {name}")

    async def _add_document(
        self,
        file_path: str,
        caption: str | None = None,
        tags: list[str] | None = None
    ) -> dict[str, Any]:
        """ドキュメント追加の実装。"""
        # 既存のRAGコアロジックを呼び出す
        # ...実装...
        pass

    async def _query(
        self,
        question: str,
        chat_history: list[dict] | None = None
    ) -> dict[str, Any]:
        """質問応答の実装。"""
        # 既存のRAGEngineを使用
        # ...実装...
        pass


class ResourceHandler:
    """MCPリソース読み取りハンドラー。"""

    def __init__(self, config: Config):
        """初期化。

        Args:
            config: アプリケーション設定
        """
        self.config = config
        self.vector_store = VectorStore(config.chroma_persist_directory)

    async def handle_resource_read(self, uri: str) -> dict[str, Any]:
        """リソース読み取りを処理します。

        Args:
            uri: リソースURI

        Returns:
            リソースの内容
        """
        if uri == "resource://documents/list":
            return await self._get_documents_list()
        elif uri.startswith("resource://documents/"):
            doc_id = uri.split("/")[-1]
            return await self._get_document(doc_id)
        # 他のリソースも同様に処理...
        else:
            raise ValueError(f"Unknown resource: {uri}")

    async def _get_documents_list(self) -> dict[str, Any]:
        """ドキュメント一覧取得の実装。"""
        # ...実装...
        pass
```

### 4. 設定ファイル更新

#### 4.1 `pyproject.toml` にMCPサーバー起動スクリプト追加

```toml
[project.scripts]
rag = "src.cli:main"
rag-mcp-server = "src.mcp.server:run"  # 追加
```

#### 4.2 依存関係追加

```toml
dependencies = [
    # 既存の依存関係...
    "mcp>=1.1.0",
    "httpx>=0.27.0",
    "pydantic>=2.0.0",
]
```

#### 4.3 `.env.sample` にMCP設定追加

```bash
# MCP Server設定
MCP_SERVER_NAME=rag-mcp-server
MCP_SERVER_VERSION=0.1.0
MCP_LOG_LEVEL=INFO
```

### 5. Claude Desktop設定

MCPサーバーをClaude Desktopで使用するには、`claude_desktop_config.json` に設定を追加：

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-mcp-server": {
      "command": "uv",
      "args": [
        "run",
        "rag-mcp-server"
      ],
      "cwd": "/path/to/rag-sample",
      "env": {
        "OLLAMA_BASE_URL": "http://localhost:11434"
      }
    }
  }
}
```

## Phase 2 詳細設計（オプション）

### 1. プロンプト（Prompts）機能

MCPプロンプトとして、RAG検索用の定型プロンプトテンプレートを提供：

| プロンプト名 | 説明 | 引数 |
|-----------|------|-----|
| `rag_query` | RAG検索質問用テンプレート | `question: str` |
| `multimodal_search` | マルチモーダル検索用テンプレート | `query: str`, `preferences: str` |
| `document_summary` | ドキュメント要約用テンプレート | `document_id: str` |

実装例（`src/mcp/prompts.py`）:

```python
"""MCPプロンプトの定義。"""

from mcp.server import Server
from mcp.types import Prompt, PromptMessage

def register_prompts(server: Server, handler):
    """MCPプロンプトをサーバーに登録します。"""

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return [
            Prompt(
                name="rag_query",
                description="RAGシステムで質問に回答するためのプロンプト",
                arguments=[
                    {
                        "name": "question",
                        "description": "質問内容",
                        "required": True
                    }
                ]
            ),
            # 他のプロンプトも定義...
        ]

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict) -> PromptMessage:
        """プロンプトを取得します。"""
        if name == "rag_query":
            question = arguments["question"]
            return PromptMessage(
                role="user",
                content=f"以下の質問に、登録されたドキュメントを参照して回答してください：\n\n{question}"
            )
        # 他のプロンプト処理...
```

### 2. サンプリング（Sampling）機能

サーバー側でLLM推論を実行し、結果をクライアントに返す機能。

**メリット:**
- クライアント側でLLMトークンを消費しない
- サーバー側で最適化されたプロンプトを使用可能

**実装:**
- `@server.create_message()` デコレーターで実装
- Ollama APIを直接呼び出してLLM推論を実行

## 実装スケジュール

### Phase 1: 基本機能（2-3週間）

| タスク | 期間 | 備考 |
|-------|------|-----|
| 1. 依存関係追加・環境構築 | 1日 | MCP SDK、Pydantic等のインストール |
| 2. MCPサーバー骨格実装 | 2日 | `server.py`、基本的なハンドラー |
| 3. ツール実装（ドキュメント管理） | 3日 | add, remove, list, clear |
| 4. ツール実装（検索・質問） | 4日 | query, search, search_images, search_multimodal |
| 5. リソース実装 | 3日 | documents, images, system |
| 6. テスト実装 | 4日 | ユニットテスト、統合テスト |
| 7. ドキュメント作成 | 2日 | README、使い方ガイド |
| 8. Claude Desktop連携テスト | 2日 | 実際の動作確認、バグ修正 |

### Phase 2: 高度な機能（1-2週間、オプション）

| タスク | 期間 | 備考 |
|-------|------|-----|
| 1. プロンプト機能実装 | 3日 | テンプレート定義、動的生成 |
| 2. サンプリング機能実装 | 4日 | LLM推論、結果返却 |
| 3. テスト・ドキュメント更新 | 2日 | 追加機能のテスト、ドキュメント |

## テスト計画

### 1. ユニットテスト（`tests/mcp/`）

- **`test_server.py`**: MCPサーバー初期化、起動テスト
- **`test_tools.py`**: 各ツールの入力検証、実行ロジック
- **`test_resources.py`**: リソースURI解析、コンテンツ取得
- **`test_handlers.py`**: ハンドラーのビジネスロジック

### 2. 統合テスト

- **`test_mcp_integration.py`**: 実際のMCPクライアントとの通信テスト
- **`test_mcp_tools_e2e.py`**: 各ツールのエンドツーエンドテスト
- **`test_mcp_resources_e2e.py`**: リソース取得のエンドツーエンドテスト

### 3. Claude Desktop連携テスト

- 実際にClaude Desktopから接続
- 各ツールの動作確認
- エラーハンドリングの確認

## セキュリティ考慮事項

1. **ファイルパスの検証**
   - パストラバーサル攻撃防止
   - 許可されたディレクトリ内のみアクセス可能

2. **入力検証**
   - すべてのツール引数をPydanticで検証
   - 不正な入力の拒否

3. **認証・認可**
   - Phase 1では未実装（ローカル実行前提）
   - 将来的にAPIキー認証を検討

4. **ログ記録**
   - すべてのツール呼び出しをログに記録
   - セキュリティイベントの監視

## 今後の拡張可能性

1. **HTTP/WebSocket対応**
   - stdio以外のトランスポート層サポート
   - リモートアクセス対応

2. **マルチユーザー対応**
   - ユーザーごとのドキュメントコレクション分離
   - 権限管理

3. **ストリーミングレスポンス**
   - LLM生成結果のストリーミング返却
   - リアルタイムフィードバック

4. **他のMCPクライアント対応**
   - VSCode拡張機能
   - カスタムアプリケーション

## 参考資料

- [MCP公式ドキュメント](https://modelcontextprotocol.io/)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Claude Desktop MCP設定ガイド](https://docs.anthropic.com/claude/docs/model-context-protocol)

## まとめ

このMCPサーバー実装により、RAG CLIアプリケーションの機能をClaude Desktopから直接利用できるようになります。

**主なメリット:**
- LLM対話中にローカルドキュメントを検索可能
- 会話の流れの中で自然にRAG機能を活用
- CLIとMCPサーバーの両方で同じRAGコアを共有

**実装のポイント:**
- 既存のRAGコアロジックを再利用
- MCPサーバーは薄いラッパー層として実装
- テスト、ドキュメント、セキュリティを重視
