# WEB UI チャットシステム実装計画

## 概要

既存のRAG CLIアプリケーションにWebベースのチャットUIを追加する実装計画書です。
既存の[src/rag/engine.py](../src/rag/engine.py)のチャット機能を活用し、FastAPIベースのWebアプリケーションとして提供します。

**目標:**
- ブラウザから簡単にRAGシステムとチャットできるようにする
- 既存のCLIチャット機能（`rag chat`）と同等の機能をWeb UIで提供
- マルチユーザー対応（セッションベース）
- 既存のRAGコアロジックを変更せずに実装

## 技術スタック

### バックエンド

| 技術 | バージョン | 用途 |
|------|-----------|------|
| **FastAPI** | >=0.115.0 | Webフレームワーク |
| **Uvicorn** | >=0.32.0 | ASGIサーバー |
| **Pydantic** | >=2.0.0 | データバリデーション（既存） |
| **WebSockets** | >=13.0 | リアルタイム通信（Phase 2） |
| **python-multipart** | >=0.0.10 | ファイルアップロード |

### フロントエンド

**推奨: オプション A（シンプル実装）**
- HTML5 + CSS3
- Vanilla JavaScript（ES6+）
- Marked.js（Markdownレンダリング）
- シンタックスハイライト用: Prism.js または Highlight.js

**代替: オプション B（モダン実装）**
- React 18+ with TypeScript
- Vite（ビルドツール）
- TailwindCSS（スタイリング）

> **推奨理由**: オプションAは依存関係が少なく、既存のPythonプロジェクトと統合しやすい。MVPの実装速度が速い。

## アーキテクチャ設計

### システム構成図

```
┌─────────────────────────────────────┐
│         Web Browser (Client)        │
│  ┌──────────────────────────────┐   │
│  │   HTML/CSS/JavaScript UI     │   │
│  │   - チャットインターフェース  │   │
│  │   - Markdownレンダリング      │   │
│  │   - セッション管理UI          │   │
│  └──────────┬───────────────────┘   │
└─────────────┼───────────────────────┘
              │ HTTP/HTTPS
              │ REST API (JSON)
┌─────────────▼───────────────────────┐
│      FastAPI Application Server      │
│  ┌────────────────────────────────┐  │
│  │      API Routers               │  │
│  │  ┌──────────────────────────┐  │  │
│  │  │ /api/chat/*              │  │  │
│  │  │ - セッション作成          │  │  │
│  │  │ - メッセージ送信          │  │  │
│  │  │ - 履歴取得               │  │  │
│  │  └──────────────────────────┘  │  │
│  │  ┌──────────────────────────┐  │  │
│  │  │ /api/query/*             │  │  │
│  │  │ - 単発クエリ              │  │  │
│  │  │ - ドキュメント検索        │  │  │
│  │  └──────────────────────────┘  │  │
│  │  ┌──────────────────────────┐  │  │
│  │  │ /api/documents/*         │  │  │
│  │  │ - 一覧取得               │  │  │
│  │  │ - 追加・削除             │  │  │
│  │  └──────────────────────────┘  │  │
│  └────────────┬───────────────────┘  │
│               │                      │
│  ┌────────────▼───────────────────┐  │
│  │   Chat Session Manager         │  │
│  │  - セッションID管理             │  │
│  │  - RAGEngineインスタンス管理    │  │
│  │  - タイムアウト処理             │  │
│  └────────────┬───────────────────┘  │
└───────────────┼──────────────────────┘
                │
┌───────────────▼──────────────────────┐
│      既存 RAG Core (変更なし)         │
│  ┌────────────────────────────────┐  │
│  │    RAGEngine                   │  │
│  │    (src/rag/engine.py)         │  │
│  │  - chat()                      │  │
│  │  - query()                     │  │
│  │  - retrieve()                  │  │
│  │  - ChatHistory管理             │  │
│  └────────────┬───────────────────┘  │
│               │                      │
│  ┌────────────▼───────────────────┐  │
│  │    VectorStore                 │  │
│  │    (ChromaDB)                  │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │    Embeddings                  │  │
│  │    (Ollama)                    │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │    LLM (Ollama)                │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

### ディレクトリ構成

```
rag-sample/
├── src/
│   ├── web/                           # 新規: Web UI関連
│   │   ├── __init__.py
│   │   ├── app.py                    # FastAPIアプリケーションエントリポイント
│   │   │
│   │   ├── routers/                  # APIルーター
│   │   │   ├── __init__.py
│   │   │   ├── chat.py              # チャットAPI
│   │   │   ├── query.py             # クエリAPI
│   │   │   └── documents.py         # ドキュメント管理API
│   │   │
│   │   ├── models/                   # API用データモデル（Pydantic）
│   │   │   ├── __init__.py
│   │   │   ├── request.py           # リクエストスキーマ
│   │   │   └── response.py          # レスポンススキーマ
│   │   │
│   │   ├── services/                 # ビジネスロジック層
│   │   │   ├── __init__.py
│   │   │   ├── chat_session.py      # セッション管理サービス
│   │   │   └── rag_adapter.py       # RAGエンジンアダプター
│   │   │
│   │   ├── static/                   # 静的ファイル（HTML/CSS/JS）
│   │   │   ├── index.html           # メインUI
│   │   │   ├── css/
│   │   │   │   └── style.css        # スタイルシート
│   │   │   └── js/
│   │   │       ├── chat.js          # チャットロジック
│   │   │       └── api-client.js    # APIクライアント
│   │   │
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── error_handler.py     # エラーハンドリング
│   │       └── cors.py              # CORS設定
│   │
│   ├── rag/                          # 既存のRAGエンジン（変更なし）
│   │   ├── engine.py                # ← chat()メソッドを使用
│   │   ├── vector_store.py
│   │   ├── embeddings.py
│   │   └── ...
│   │
│   ├── cli.py                        # 既存のCLI（変更なし）
│   └── ...
│
├── tests/
│   └── web/                          # Web UI用テスト
│       ├── __init__.py
│       ├── test_chat_router.py
│       ├── test_session_manager.py
│       └── test_api_integration.py
│
├── docs/
│   ├── web-ui-implementation-plan.md # このドキュメント
│   ├── web-ui-api-reference.md       # API仕様書（実装後作成）
│   └── ...
│
├── pyproject.toml                    # 依存関係追加
└── README.md                         # 使用方法追加
```

## データモデル設計

### API リクエスト/レスポンススキーマ

#### `src/web/models/request.py`

```python
"""APIリクエストモデル"""

from typing import Optional
from pydantic import BaseModel, Field


class CreateSessionRequest(BaseModel):
    """チャットセッション作成リクエスト"""
    session_name: Optional[str] = Field(None, description="セッション名（任意）")
    max_history: int = Field(10, ge=1, le=100, description="最大履歴メッセージ数")


class ChatMessageRequest(BaseModel):
    """チャットメッセージ送信リクエスト"""
    message: str = Field(..., min_length=1, max_length=10000, description="ユーザーメッセージ")
    n_results: int = Field(3, ge=1, le=10, description="検索するコンテキスト数")
    include_sources: bool = Field(True, description="情報源を含めるか")


class QueryRequest(BaseModel):
    """単発クエリリクエスト"""
    question: str = Field(..., min_length=1, max_length=10000, description="質問文")
    n_results: int = Field(5, ge=1, le=20, description="検索結果数")
    include_sources: bool = Field(True, description="情報源を含めるか")


class DocumentUploadRequest(BaseModel):
    """ドキュメントアップロードリクエスト（メタデータ）"""
    file_path: str = Field(..., description="ファイルパス")
    document_type: Optional[str] = Field(None, description="ドキュメントタイプ")
```

#### `src/web/models/response.py`

```python
"""APIレスポンスモデル"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    """情報源情報"""
    name: str
    source: str
    score: float


class ChatMessageResponse(BaseModel):
    """チャットメッセージレスポンス"""
    session_id: str
    message_id: str
    answer: str
    sources: Optional[list[SourceInfo]] = None
    context_count: int
    history_length: int
    timestamp: datetime


class SessionInfo(BaseModel):
    """セッション情報"""
    session_id: str
    session_name: Optional[str] = None
    created_at: datetime
    last_activity: datetime
    message_count: int
    is_active: bool


class QueryResponse(BaseModel):
    """クエリレスポンス"""
    answer: str
    sources: Optional[list[SourceInfo]] = None
    context_count: int


class DocumentInfo(BaseModel):
    """ドキュメント情報"""
    document_id: str
    name: str
    source: str
    chunk_count: int
    created_at: datetime


class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
```

## API エンドポイント設計

### チャット API (`/api/chat`)

| メソッド | エンドポイント | 説明 | リクエスト | レスポンス |
|---------|---------------|------|-----------|-----------|
| POST | `/api/chat/sessions` | 新規セッション作成 | `CreateSessionRequest` | `SessionInfo` |
| GET | `/api/chat/sessions` | セッション一覧取得 | - | `list[SessionInfo]` |
| GET | `/api/chat/sessions/{session_id}` | セッション情報取得 | - | `SessionInfo` |
| POST | `/api/chat/sessions/{session_id}/messages` | メッセージ送信 | `ChatMessageRequest` | `ChatMessageResponse` |
| GET | `/api/chat/sessions/{session_id}/history` | 履歴取得 | - | `list[dict]` |
| DELETE | `/api/chat/sessions/{session_id}` | セッション削除 | - | `{"status": "deleted"}` |
| POST | `/api/chat/sessions/{session_id}/clear` | 履歴クリア | - | `{"status": "cleared"}` |

### クエリ API (`/api/query`)

| メソッド | エンドポイント | 説明 | リクエスト | レスポンス |
|---------|---------------|------|-----------|-----------|
| POST | `/api/query` | 単発質問（履歴なし） | `QueryRequest` | `QueryResponse` |
| POST | `/api/search` | ドキュメント検索 | `{"query": str, "n_results": int}` | `list[SearchResult]` |

### ドキュメント API (`/api/documents`)

| メソッド | エンドポイント | 説明 | リクエスト | レスポンス |
|---------|---------------|------|-----------|-----------|
| GET | `/api/documents` | ドキュメント一覧 | - | `list[DocumentInfo]` |
| POST | `/api/documents/upload` | ドキュメント追加 | `multipart/form-data` | `DocumentInfo` |
| DELETE | `/api/documents/{document_id}` | ドキュメント削除 | - | `{"status": "deleted"}` |

### ヘルスチェック API

| メソッド | エンドポイント | 説明 |
|---------|---------------|------|
| GET | `/health` | ヘルスチェック |
| GET | `/api/status` | システムステータス |

## 実装フェーズ

### Phase 1: 基盤構築（最優先）

**目標**: 最小限のMVPを動かす

#### 1.1 FastAPI アプリケーション基盤

**実装ファイル**: `src/web/app.py`

```python
"""FastAPI Webアプリケーション"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .routers import chat, query, documents

app = FastAPI(
    title="RAG Chat Web UI",
    description="RAGシステムのWebチャットインターフェース",
    version="0.1.0"
)

# CORS設定（開発環境用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番環境では制限すること
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーターの登録
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])

# 静的ファイルのサーブ
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {"message": "RAG Chat Web UI", "status": "running"}

@app.get("/health")
async def health_check():
    """ヘルスチェック"""
    return {"status": "healthy"}
```

**タスク**:
- [x] FastAPIプロジェクト構造作成
- [ ] 基本的なルーティング設定
- [ ] CORS設定
- [ ] 静的ファイルサーブ設定
- [ ] ヘルスチェックエンドポイント

#### 1.2 セッション管理サービス

**実装ファイル**: `src/web/services/chat_session.py`

```python
"""チャットセッション管理サービス"""

import uuid
from datetime import datetime, timedelta
from typing import Optional
from threading import Lock

from ...rag.engine import RAGEngine, create_rag_engine
from ...utils.config import get_config


class ChatSession:
    """個別チャットセッション"""

    def __init__(self, session_id: str, session_name: Optional[str] = None):
        self.session_id = session_id
        self.session_name = session_name or f"Session {session_id[:8]}"
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.rag_engine = create_rag_engine()
        self.rag_engine.initialize()
        self.message_count = 0

    def chat(self, message: str, n_results: int = 3, include_sources: bool = True) -> dict:
        """チャットメッセージを送信"""
        self.last_activity = datetime.now()
        result = self.rag_engine.chat(
            message=message,
            n_results=n_results,
            include_sources=include_sources
        )
        self.message_count += 1
        return result

    def get_history(self) -> list[dict]:
        """履歴を取得"""
        return self.rag_engine.get_chat_history()

    def clear_history(self) -> None:
        """履歴をクリア"""
        self.rag_engine.clear_chat_history()
        self.message_count = 0


class ChatSessionManager:
    """セッション管理マネージャー（シングルトン）"""

    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.sessions: dict[str, ChatSession] = {}
        self.session_timeout = timedelta(hours=1)
        self._initialized = True

    def create_session(self, session_name: Optional[str] = None) -> ChatSession:
        """新規セッションを作成"""
        session_id = str(uuid.uuid4())
        session = ChatSession(session_id, session_name)
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """セッションを取得"""
        session = self.sessions.get(session_id)
        if session and self._is_session_expired(session):
            self.delete_session(session_id)
            return None
        return session

    def delete_session(self, session_id: str) -> bool:
        """セッションを削除"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(self) -> list[ChatSession]:
        """全セッション一覧"""
        # 有効なセッションのみ返す
        valid_sessions = []
        expired_ids = []

        for session_id, session in self.sessions.items():
            if self._is_session_expired(session):
                expired_ids.append(session_id)
            else:
                valid_sessions.append(session)

        # 期限切れセッションを削除
        for session_id in expired_ids:
            self.delete_session(session_id)

        return valid_sessions

    def _is_session_expired(self, session: ChatSession) -> bool:
        """セッションが期限切れかチェック"""
        return datetime.now() - session.last_activity > self.session_timeout
```

**タスク**:
- [ ] ChatSessionクラス実装
- [ ] ChatSessionManagerクラス実装（シングルトン）
- [ ] セッションタイムアウト機能
- [ ] スレッドセーフな実装

#### 1.3 チャット API ルーター

**実装ファイル**: `src/web/routers/chat.py`

```python
"""チャットAPIルーター"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime

from ..models.request import CreateSessionRequest, ChatMessageRequest
from ..models.response import (
    SessionInfo, ChatMessageResponse, SourceInfo, ErrorResponse
)
from ..services.chat_session import ChatSessionManager

router = APIRouter()
session_manager = ChatSessionManager()


@router.post("/sessions", response_model=SessionInfo, status_code=status.HTTP_201_CREATED)
async def create_session(request: CreateSessionRequest):
    """新規チャットセッションを作成"""
    session = session_manager.create_session(request.session_name)

    return SessionInfo(
        session_id=session.session_id,
        session_name=session.session_name,
        created_at=session.created_at,
        last_activity=session.last_activity,
        message_count=0,
        is_active=True
    )


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """全セッション一覧を取得"""
    sessions = session_manager.list_sessions()

    return [
        SessionInfo(
            session_id=s.session_id,
            session_name=s.session_name,
            created_at=s.created_at,
            last_activity=s.last_activity,
            message_count=s.message_count,
            is_active=True
        )
        for s in sessions
    ]


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageResponse)
async def send_message(session_id: str, request: ChatMessageRequest):
    """チャットメッセージを送信"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    try:
        result = session.chat(
            message=request.message,
            n_results=request.n_results,
            include_sources=request.include_sources
        )

        # レスポンス構築
        sources = None
        if "sources" in result:
            sources = [
                SourceInfo(
                    name=s["name"],
                    source=s["source"],
                    score=s["score"]
                )
                for s in result["sources"]
            ]

        return ChatMessageResponse(
            session_id=session_id,
            message_id=str(datetime.now().timestamp()),
            answer=result["answer"],
            sources=sources,
            context_count=result["context_count"],
            history_length=result["history_length"],
            timestamp=datetime.now()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )


@router.get("/sessions/{session_id}/history")
async def get_history(session_id: str):
    """チャット履歴を取得"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    return {"history": session.get_history()}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """セッションを削除"""
    if not session_manager.delete_session(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    return {"status": "deleted", "session_id": session_id}


@router.post("/sessions/{session_id}/clear")
async def clear_history(session_id: str):
    """チャット履歴をクリア"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found"
        )

    session.clear_history()
    return {"status": "cleared", "session_id": session_id}
```

**タスク**:
- [ ] セッション作成エンドポイント
- [ ] メッセージ送信エンドポイント
- [ ] 履歴取得エンドポイント
- [ ] セッション削除エンドポイント
- [ ] エラーハンドリング

#### 1.4 基本的な HTML UI

**実装ファイル**: `src/web/static/index.html`

シンプルなチャットインターフェースを実装（詳細はPhase 2で）

**タスク**:
- [ ] HTML構造作成
- [ ] 基本的なスタイリング
- [ ] JavaScriptでAPI連携
- [ ] メッセージ送受信機能

### Phase 2: UI/UX 改善（高優先度）

#### 2.1 フロントエンド機能拡張

**実装ファイル**: `src/web/static/js/chat.js`

- Markdownレンダリング（marked.js使用）
- コードのシンタックスハイライト
- ローディングインジケーター
- エラーメッセージ表示
- セッション管理UI

#### 2.2 スタイリング改善

**実装ファイル**: `src/web/static/css/style.css`

- モダンなチャットUI
- レスポンシブデザイン
- ダークモード対応
- アニメーション

**タスク**:
- [ ] Markdownレンダリング実装
- [ ] UIコンポーネント改善
- [ ] レスポンシブ対応
- [ ] ダークモード実装

### Phase 3: 拡張機能（中優先度）

#### 3.1 クエリ・検索 API

**実装ファイル**: `src/web/routers/query.py`

- 単発クエリエンドポイント
- ドキュメント検索エンドポイント
- 画像検索エンドポイント

#### 3.2 ドキュメント管理 API

**実装ファイル**: `src/web/routers/documents.py`

- ドキュメント一覧取得
- ドキュメントアップロード
- ドキュメント削除

**タスク**:
- [ ] クエリAPIルーター実装
- [ ] ドキュメント管理APIルーター実装
- [ ] ファイルアップロード機能
- [ ] ドキュメント管理UI

### Phase 4: 高度な機能（低優先度）

#### 4.1 WebSocket対応（ストリーミング回答）

リアルタイムでLLMの回答をストリーミング表示

```python
@router.websocket("/ws/chat/{session_id}")
async def chat_websocket(websocket: WebSocket, session_id: str):
    """WebSocketでチャット"""
    await websocket.accept()
    # ストリーミング実装
```

#### 4.2 認証・セキュリティ

- JWT認証
- レート制限
- セッションベースアクセス制御

#### 4.3 高度なUI機能

- チャット履歴エクスポート
- パラメータ調整UI（temperature、max_tokens等）
- マルチモーダル対応（画像検索結果表示）

**タスク**:
- [ ] WebSocketエンドポイント実装
- [ ] 認証機能実装
- [ ] レート制限実装
- [ ] 高度なUI機能

## 依存関係の追加

### `pyproject.toml` への追加

```toml
[project.dependencies]
# 既存の依存関係...
"fastapi>=0.115.0"
"uvicorn[standard]>=0.32.0"
"websockets>=13.0"
"python-multipart>=0.0.10"

[project.scripts]
rag = "src.cli:main"
rag-mcp-server = "src.mcp.server:run"
rag-web = "src.web.app:run"  # 新規追加

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.1.0",
    "pytest-mock>=3.12.0",
    "pytest-asyncio>=0.23.0",
    "httpx>=0.27.0",  # FastAPIテスト用
]
```

### インストールコマンド

```bash
# 依存関係のインストール
uv sync

# 開発サーバー起動
uv run uvicorn src.web.app:app --reload --port 8000

# または
uv run rag-web
```

## テスト戦略

### 単体テスト

**ファイル**: `tests/web/test_session_manager.py`

```python
"""セッション管理のテスト"""

import pytest
from src.web.services.chat_session import ChatSessionManager, ChatSession


def test_create_session():
    """セッション作成テスト"""
    manager = ChatSessionManager()
    session = manager.create_session("Test Session")

    assert session.session_id is not None
    assert session.session_name == "Test Session"
    assert session.message_count == 0


def test_get_session():
    """セッション取得テスト"""
    manager = ChatSessionManager()
    session = manager.create_session()

    retrieved = manager.get_session(session.session_id)
    assert retrieved is not None
    assert retrieved.session_id == session.session_id
```

### 統合テスト

**ファイル**: `tests/web/test_api_integration.py`

```python
"""API統合テスト"""

import pytest
from fastapi.testclient import TestClient
from src.web.app import app

client = TestClient(app)


def test_create_session_and_chat():
    """セッション作成〜チャットまでのフロー"""
    # セッション作成
    response = client.post("/api/chat/sessions", json={"session_name": "Test"})
    assert response.status_code == 201
    session_id = response.json()["session_id"]

    # メッセージ送信
    response = client.post(
        f"/api/chat/sessions/{session_id}/messages",
        json={"message": "こんにちは", "n_results": 3}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
```

## セキュリティ考慮事項

### 1. 入力バリデーション
- Pydanticモデルで全入力を検証
- 文字列長の制限（max_length）
- XSS対策（フロントエンドでサニタイズ）

### 2. CORS設定
```python
# 本番環境では制限
allow_origins=["https://yourdomain.com"]
```

### 3. レート制限
```python
from slowapi import Limiter

limiter = Limiter(key_func=get_remote_address)

@router.post("/sessions/{session_id}/messages")
@limiter.limit("10/minute")  # 1分に10リクエストまで
async def send_message(...):
    ...
```

### 4. セッション管理
- タイムアウト処理（デフォルト1時間）
- メモリリーク防止
- セッション数の上限設定

### 5. ファイルアップロード
- MIMEタイプチェック
- ファイルサイズ制限
- ウイルススキャン（本番環境）

## デプロイメント

### 開発環境

```bash
# 開発サーバー起動（ホットリロード有効）
uv run uvicorn src.web.app:app --reload --host 0.0.0.0 --port 8000

# ブラウザでアクセス
open http://localhost:8000/static/index.html
```

### 本番環境（例）

#### Docker化

**Dockerfile**:
```dockerfile
FROM python:3.13-slim

WORKDIR /app

# uvのインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# 依存関係のインストール
COPY pyproject.toml .
RUN uv sync --no-dev

# アプリケーションのコピー
COPY . .

# Ollamaへの接続設定（環境変数）
ENV OLLAMA_BASE_URL=http://ollama:11434

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama

  rag-web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - CHROMA_PERSIST_DIRECTORY=/data/chroma_db
    volumes:
      - ./chroma_db:/data/chroma_db
    depends_on:
      - ollama

volumes:
  ollama-data:
```

## 実装チェックリスト

### Phase 1: MVP（必須）

- [ ] FastAPIアプリケーション基盤構築
  - [ ] `src/web/app.py` 作成
  - [ ] ルーター設定
  - [ ] CORS設定
  - [ ] 静的ファイルサーブ
- [ ] セッション管理サービス
  - [ ] `ChatSession` クラス実装
  - [ ] `ChatSessionManager` クラス実装
  - [ ] タイムアウト機能
- [ ] チャットAPIルーター
  - [ ] セッション作成エンドポイント
  - [ ] メッセージ送信エンドポイント
  - [ ] 履歴取得エンドポイント
  - [ ] セッション削除エンドポイント
- [ ] データモデル
  - [ ] リクエストモデル作成
  - [ ] レスポンスモデル作成
- [ ] 基本的なHTML UI
  - [ ] `index.html` 作成
  - [ ] `chat.js` 基本実装
  - [ ] `style.css` 基本スタイル
- [ ] 依存関係追加
  - [ ] `pyproject.toml` 更新
  - [ ] `uv sync` 実行
- [ ] テスト
  - [ ] セッション管理テスト
  - [ ] APIエンドポイントテスト
- [ ] ドキュメント
  - [ ] README更新（起動方法）
  - [ ] API仕様書作成

### Phase 2: UI改善（推奨）

- [ ] Markdownレンダリング実装
- [ ] シンタックスハイライト
- [ ] レスポンシブデザイン
- [ ] ダークモード
- [ ] エラーハンドリングUI

### Phase 3: 拡張機能（オプション）

- [ ] クエリAPI実装
- [ ] ドキュメント管理API実装
- [ ] ファイルアップロード機能

### Phase 4: 高度な機能（オプション）

- [ ] WebSocket対応
- [ ] 認証機能
- [ ] レート制限
- [ ] チャット履歴エクスポート

## まとめ

この実装計画では、既存のRAGエンジンを活用しながら、段階的にWeb UIを構築します。

**推奨実装順序:**
1. **Phase 1（MVP）**: 基本的なチャット機能を動かす（1-2週間）
2. **Phase 2（UI改善）**: ユーザー体験を向上させる（1週間）
3. **Phase 3（拡張機能）**: 必要に応じて追加機能を実装（1-2週間）
4. **Phase 4（高度な機能）**: 本番運用に向けた機能追加（適宜）

**次のステップ:**
1. `pyproject.toml` に依存関係を追加
2. `src/web/` ディレクトリ構造を作成
3. Phase 1のタスクから実装開始

実装開始の準備が整いましたら、具体的なコードを作成いたします。
