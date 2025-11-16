# WEB UI チャットシステム実装計画（React + TypeScript版）

## 概要

既存のRAG CLIアプリケーションにWebベースのチャットUIを追加する実装計画書です。
既存の[src/rag/engine.py](../src/rag/engine.py)のチャット機能を活用し、**React + TypeScript + FastAPI**のモダンなスタックで実装します。

**目標:**
- ブラウザから簡単にRAGシステムとチャットできるようにする
- 既存のCLIチャット機能（`rag chat`）と同等の機能をWeb UIで提供
- マルチユーザー対応（セッションベース）
- モダンで保守性の高いフロントエンド
- 既存のRAGコアロジックを変更せずに実装

## 技術スタック

### バックエンド

| 技術 | バージョン | 用途 |
|------|-----------|------|
| **FastAPI** | >=0.115.0 | Webフレームワーク |
| **Uvicorn** | >=0.32.0 | ASGIサーバー |
| **Pydantic** | >=2.0.0 | データバリデーション（既存） |
| **WebSockets** | >=13.0 | リアルタイム通信 |
| **python-multipart** | >=0.0.10 | ファイルアップロード |

### フロントエンド（React + TypeScript）

| 技術 | バージョン | 用途 |
|------|-----------|------|
| **React** | 18+ | UIフレームワーク |
| **TypeScript** | 5+ | 型安全性 |
| **Vite** | 5+ | ビルドツール（高速） |
| **TailwindCSS** | 3+ | ユーティリティファーストCSS |
| **React Router** | 6+ | ルーティング |
| **React Query (TanStack Query)** | 5+ | サーバーステート管理 |
| **Zustand** | 4+ | クライアントステート管理（軽量） |
| **react-markdown** | 9+ | Markdownレンダリング |
| **react-syntax-highlighter** | 15+ | コードハイライト |
| **Axios** | 1+ | HTTPクライアント |
| **React Hot Toast** | 2+ | 通知UI |

### 開発ツール

| 技術 | 用途 |
|------|------|
| **ESLint** | コード品質チェック |
| **Prettier** | コードフォーマッター |
| **TypeScript ESLint** | TypeScript用Lint |
| **Vitest** | ユニットテスト |
| **React Testing Library** | コンポーネントテスト |
| **Playwright** | E2Eテスト（オプション） |

## アーキテクチャ設計

### システム構成図

```
┌─────────────────────────────────────────────┐
│         Web Browser (Client)                │
│  ┌───────────────────────────────────────┐  │
│  │   React Application (SPA)             │  │
│  │  ┌─────────────────────────────────┐  │  │
│  │  │  Pages (React Router)           │  │  │
│  │  │  - /        (ChatPage)          │  │  │
│  │  │  - /search  (SearchPage)        │  │  │
│  │  │  - /docs    (DocumentsPage)     │  │  │
│  │  └─────────────┬───────────────────┘  │  │
│  │                │                       │  │
│  │  ┌─────────────▼───────────────────┐  │  │
│  │  │  Components                     │  │  │
│  │  │  - ChatInterface                │  │  │
│  │  │  - MessageList                  │  │  │
│  │  │  - MessageInput                 │  │  │
│  │  │  - SessionSidebar               │  │  │
│  │  │  - MarkdownRenderer             │  │  │
│  │  └─────────────┬───────────────────┘  │  │
│  │                │                       │  │
│  │  ┌─────────────▼───────────────────┐  │  │
│  │  │  State Management               │  │  │
│  │  │  - React Query (server state)   │  │  │
│  │  │  - Zustand (client state)       │  │  │
│  │  └─────────────┬───────────────────┘  │  │
│  │                │                       │  │
│  │  ┌─────────────▼───────────────────┐  │  │
│  │  │  API Client (Axios)             │  │  │
│  │  │  - ChatAPI                      │  │  │
│  │  │  - QueryAPI                     │  │  │
│  │  │  - DocumentAPI                  │  │  │
│  │  └─────────────┬───────────────────┘  │  │
│  └────────────────┼─────────────────────┘  │
└───────────────────┼─────────────────────────┘
                    │ HTTP/HTTPS (REST API)
                    │ WebSocket (リアルタイム)
┌───────────────────▼─────────────────────────┐
│      FastAPI Application Server              │
│  ┌────────────────────────────────────────┐  │
│  │      API Routers                       │  │
│  │  - /api/chat/*    (チャット)           │  │
│  │  - /api/query/*   (検索)               │  │
│  │  - /api/documents/* (ドキュメント管理) │  │
│  │  - /ws/chat/{session_id} (WebSocket)   │  │
│  └────────────┬───────────────────────────┘  │
│               │                               │
│  ┌────────────▼───────────────────────────┐  │
│  │   Chat Session Manager                 │  │
│  │  - セッションID管理                     │  │
│  │  - RAGEngineインスタンス管理            │  │
│  │  - タイムアウト処理                     │  │
│  └────────────┬───────────────────────────┘  │
└───────────────┼─────────────────────────────┘
                │
┌───────────────▼─────────────────────────────┐
│      既存 RAG Core (変更なし)                │
│  - RAGEngine (src/rag/engine.py)            │
│  - VectorStore (ChromaDB)                   │
│  - Embeddings (Ollama)                      │
└─────────────────────────────────────────────┘
```

### ディレクトリ構成

```
rag-sample/
├── backend/                          # バックエンド（既存のsrcを移動）
│   ├── src/
│   │   ├── web/                     # 新規: FastAPI Web API
│   │   │   ├── __init__.py
│   │   │   ├── app.py              # FastAPIアプリ
│   │   │   ├── routers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── chat.py
│   │   │   │   ├── query.py
│   │   │   │   └── documents.py
│   │   │   ├── models/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── request.py
│   │   │   │   └── response.py
│   │   │   ├── services/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── chat_session.py
│   │   │   │   └── rag_adapter.py
│   │   │   └── middleware/
│   │   │       ├── __init__.py
│   │   │       └── cors.py
│   │   ├── rag/                    # 既存のRAGエンジン
│   │   ├── cli.py                  # 既存のCLI
│   │   └── ...
│   ├── tests/
│   └── pyproject.toml
│
├── frontend/                         # 新規: React フロントエンド
│   ├── src/
│   │   ├── main.tsx                # エントリポイント
│   │   ├── App.tsx                 # ルートコンポーネント
│   │   │
│   │   ├── pages/                  # ページコンポーネント
│   │   │   ├── ChatPage.tsx       # チャット画面
│   │   │   ├── SearchPage.tsx     # 検索画面
│   │   │   └── DocumentsPage.tsx  # ドキュメント管理画面
│   │   │
│   │   ├── components/             # 再利用可能なコンポーネント
│   │   │   ├── chat/
│   │   │   │   ├── ChatInterface.tsx
│   │   │   │   ├── MessageList.tsx
│   │   │   │   ├── MessageItem.tsx
│   │   │   │   ├── MessageInput.tsx
│   │   │   │   └── SessionSidebar.tsx
│   │   │   ├── common/
│   │   │   │   ├── Button.tsx
│   │   │   │   ├── Input.tsx
│   │   │   │   ├── Loading.tsx
│   │   │   │   └── ErrorMessage.tsx
│   │   │   └── markdown/
│   │   │       └── MarkdownRenderer.tsx
│   │   │
│   │   ├── api/                    # API クライアント
│   │   │   ├── client.ts          # Axios インスタンス
│   │   │   ├── chat.ts            # チャットAPI
│   │   │   ├── query.ts           # クエリAPI
│   │   │   └── documents.ts       # ドキュメントAPI
│   │   │
│   │   ├── hooks/                  # カスタムフック
│   │   │   ├── useChat.ts         # チャット機能
│   │   │   ├── useSessions.ts     # セッション管理
│   │   │   ├── useQuery.ts        # クエリ機能
│   │   │   └── useDocuments.ts    # ドキュメント管理
│   │   │
│   │   ├── store/                  # Zustand ストア
│   │   │   ├── sessionStore.ts    # セッションステート
│   │   │   └── uiStore.ts         # UIステート（サイドバー等）
│   │   │
│   │   ├── types/                  # TypeScript型定義
│   │   │   ├── chat.ts
│   │   │   ├── session.ts
│   │   │   ├── document.ts
│   │   │   └── api.ts
│   │   │
│   │   ├── utils/                  # ユーティリティ関数
│   │   │   ├── formatters.ts
│   │   │   └── validators.ts
│   │   │
│   │   └── styles/                 # グローバルスタイル
│   │       └── index.css
│   │
│   ├── public/                      # 静的ファイル
│   │   └── favicon.ico
│   │
│   ├── index.html                   # HTMLエントリ
│   ├── package.json                 # npm依存関係
│   ├── tsconfig.json               # TypeScript設定
│   ├── vite.config.ts              # Vite設定
│   ├── tailwind.config.js          # Tailwind設定
│   ├── postcss.config.js           # PostCSS設定
│   └── .eslintrc.js                # ESLint設定
│
├── docs/
│   ├── web-ui-implementation-plan-react.md  # このドキュメント
│   └── ...
│
└── README.md
```

## TypeScript 型定義

### `frontend/src/types/chat.ts`

```typescript
/**
 * チャット関連の型定義
 */

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: Source[];
  contextCount?: number;
}

export interface Source {
  name: string;
  source: string;
  score: number;
}

export interface ChatMessageRequest {
  message: string;
  n_results?: number;
  include_sources?: boolean;
}

export interface ChatMessageResponse {
  session_id: string;
  message_id: string;
  answer: string;
  sources?: Source[];
  context_count: number;
  history_length: number;
  timestamp: string;
}
```

### `frontend/src/types/session.ts`

```typescript
/**
 * セッション関連の型定義
 */

export interface Session {
  session_id: string;
  session_name: string | null;
  created_at: string;
  last_activity: string;
  message_count: number;
  is_active: boolean;
}

export interface CreateSessionRequest {
  session_name?: string;
  max_history?: number;
}
```

### `frontend/src/types/document.ts`

```typescript
/**
 * ドキュメント関連の型定義
 */

export interface Document {
  document_id: string;
  name: string;
  source: string;
  chunk_count: number;
  created_at: string;
}

export interface SearchResult {
  document_name: string;
  chunk: {
    content: string;
    metadata: Record<string, any>;
  };
  score: number;
  document_source: string;
}
```

## API クライアント実装例

### `frontend/src/api/client.ts`

```typescript
/**
 * Axios クライアント設定
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30秒
});

// リクエストインターセプター（認証トークン追加等）
apiClient.interceptors.request.use(
  (config) => {
    // 必要に応じてトークンを追加
    // const token = localStorage.getItem('token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => Promise.reject(error)
);

// レスポンスインターセプター（エラーハンドリング）
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // 認証エラー処理
      console.error('Unauthorized');
    }
    return Promise.reject(error);
  }
);
```

### `frontend/src/api/chat.ts`

```typescript
/**
 * チャットAPI クライアント
 */

import { apiClient } from './client';
import type {
  Session,
  CreateSessionRequest,
  ChatMessageRequest,
  ChatMessageResponse,
} from '../types';

export const chatAPI = {
  // セッション作成
  createSession: async (data: CreateSessionRequest): Promise<Session> => {
    const response = await apiClient.post<Session>('/api/chat/sessions', data);
    return response.data;
  },

  // セッション一覧取得
  getSessions: async (): Promise<Session[]> => {
    const response = await apiClient.get<Session[]>('/api/chat/sessions');
    return response.data;
  },

  // セッション情報取得
  getSession: async (sessionId: string): Promise<Session> => {
    const response = await apiClient.get<Session>(`/api/chat/sessions/${sessionId}`);
    return response.data;
  },

  // メッセージ送信
  sendMessage: async (
    sessionId: string,
    data: ChatMessageRequest
  ): Promise<ChatMessageResponse> => {
    const response = await apiClient.post<ChatMessageResponse>(
      `/api/chat/sessions/${sessionId}/messages`,
      data
    );
    return response.data;
  },

  // 履歴取得
  getHistory: async (sessionId: string): Promise<any> => {
    const response = await apiClient.get(`/api/chat/sessions/${sessionId}/history`);
    return response.data;
  },

  // セッション削除
  deleteSession: async (sessionId: string): Promise<void> => {
    await apiClient.delete(`/api/chat/sessions/${sessionId}`);
  },

  // 履歴クリア
  clearHistory: async (sessionId: string): Promise<void> => {
    await apiClient.post(`/api/chat/sessions/${sessionId}/clear`);
  },
};
```

## カスタムフック実装例

### `frontend/src/hooks/useChat.ts`

```typescript
/**
 * チャット機能のカスタムフック
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { chatAPI } from '../api/chat';
import type { ChatMessageRequest, ChatMessageResponse } from '../types';

export const useChat = (sessionId: string | null) => {
  const queryClient = useQueryClient();

  // メッセージ送信
  const sendMessage = useMutation({
    mutationFn: (data: ChatMessageRequest) => {
      if (!sessionId) throw new Error('No active session');
      return chatAPI.sendMessage(sessionId, data);
    },
    onSuccess: () => {
      // 履歴を再取得
      queryClient.invalidateQueries({ queryKey: ['chatHistory', sessionId] });
    },
  });

  // 履歴取得
  const { data: history, isLoading: isLoadingHistory } = useQuery({
    queryKey: ['chatHistory', sessionId],
    queryFn: () => {
      if (!sessionId) throw new Error('No active session');
      return chatAPI.getHistory(sessionId);
    },
    enabled: !!sessionId,
  });

  return {
    sendMessage,
    history,
    isLoadingHistory,
  };
};
```

### `frontend/src/hooks/useSessions.ts`

```typescript
/**
 * セッション管理のカスタムフック
 */

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { chatAPI } from '../api/chat';
import type { CreateSessionRequest } from '../types';

export const useSessions = () => {
  const queryClient = useQueryClient();

  // セッション一覧取得
  const { data: sessions, isLoading } = useQuery({
    queryKey: ['sessions'],
    queryFn: chatAPI.getSessions,
  });

  // セッション作成
  const createSession = useMutation({
    mutationFn: (data: CreateSessionRequest) => chatAPI.createSession(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });

  // セッション削除
  const deleteSession = useMutation({
    mutationFn: (sessionId: string) => chatAPI.deleteSession(sessionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sessions'] });
    },
  });

  return {
    sessions,
    isLoading,
    createSession,
    deleteSession,
  };
};
```

## コンポーネント実装例

### `frontend/src/components/chat/ChatInterface.tsx`

```typescript
/**
 * チャットインターフェース メインコンポーネント
 */

import React, { useState } from 'react';
import { MessageList } from './MessageList';
import { MessageInput } from './MessageInput';
import { useChat } from '../../hooks/useChat';
import { Loading } from '../common/Loading';
import { ErrorMessage } from '../common/ErrorMessage';

interface ChatInterfaceProps {
  sessionId: string;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ sessionId }) => {
  const { sendMessage, history, isLoadingHistory } = useChat(sessionId);

  const handleSendMessage = async (message: string) => {
    try {
      await sendMessage.mutateAsync({
        message,
        n_results: 3,
        include_sources: true,
      });
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  if (isLoadingHistory) {
    return <Loading />;
  }

  return (
    <div className="flex flex-col h-full bg-gray-50 dark:bg-gray-900">
      {/* ヘッダー */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
        <h1 className="text-xl font-semibold text-gray-900 dark:text-white">
          RAG Chat
        </h1>
      </div>

      {/* メッセージリスト */}
      <div className="flex-1 overflow-y-auto">
        <MessageList messages={history?.history || []} />
      </div>

      {/* エラー表示 */}
      {sendMessage.isError && (
        <ErrorMessage message="メッセージの送信に失敗しました" />
      )}

      {/* 入力欄 */}
      <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <MessageInput
          onSendMessage={handleSendMessage}
          isLoading={sendMessage.isPending}
        />
      </div>
    </div>
  );
};
```

### `frontend/src/components/chat/MessageItem.tsx`

```typescript
/**
 * 個別メッセージアイテム
 */

import React from 'react';
import { MarkdownRenderer } from '../markdown/MarkdownRenderer';
import type { Message } from '../../types';

interface MessageItemProps {
  message: Message;
}

export const MessageItem: React.FC<MessageItemProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div
      className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 px-6`}
    >
      <div
        className={`max-w-3xl rounded-lg px-4 py-3 ${
          isUser
            ? 'bg-blue-600 text-white'
            : 'bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 border border-gray-200 dark:border-gray-700'
        }`}
      >
        {/* メッセージ内容 */}
        {isUser ? (
          <p className="whitespace-pre-wrap">{message.content}</p>
        ) : (
          <MarkdownRenderer content={message.content} />
        )}

        {/* 情報源（アシスタントのみ） */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-700">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-2">
              参照元:
            </p>
            <div className="space-y-1">
              {message.sources.map((source, idx) => (
                <div
                  key={idx}
                  className="text-xs text-gray-600 dark:text-gray-400"
                >
                  {source.name} (類似度: {source.score.toFixed(3)})
                </div>
              ))}
            </div>
          </div>
        )}

        {/* タイムスタンプ */}
        <p className="text-xs mt-2 opacity-70">
          {new Date(message.timestamp).toLocaleTimeString()}
        </p>
      </div>
    </div>
  );
};
```

### `frontend/src/components/markdown/MarkdownRenderer.tsx`

```typescript
/**
 * Markdownレンダリングコンポーネント
 */

import React from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface MarkdownRendererProps {
  content: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
}) => {
  return (
    <ReactMarkdown
      className="prose dark:prose-invert max-w-none"
      components={{
        code({ node, inline, className, children, ...props }) {
          const match = /language-(\w+)/.exec(className || '');
          return !inline && match ? (
            <SyntaxHighlighter
              style={vscDarkPlus}
              language={match[1]}
              PreTag="div"
              {...props}
            >
              {String(children).replace(/\n$/, '')}
            </SyntaxHighlighter>
          ) : (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
};
```

## Zustand ストア実装例

### `frontend/src/store/sessionStore.ts`

```typescript
/**
 * セッションステート管理（Zustand）
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface SessionState {
  activeSessionId: string | null;
  setActiveSessionId: (sessionId: string | null) => void;
}

export const useSessionStore = create<SessionState>()(
  persist(
    (set) => ({
      activeSessionId: null,
      setActiveSessionId: (sessionId) => set({ activeSessionId: sessionId }),
    }),
    {
      name: 'session-storage', // localStorage key
    }
  )
);
```

## 実装フェーズ

### Phase 1: プロジェクトセットアップ（1-2日）

**タスク:**
- [ ] Reactプロジェクト初期化（Vite）
- [ ] TypeScript + ESLint + Prettier 設定
- [ ] TailwindCSS セットアップ
- [ ] React Router セットアップ
- [ ] React Query セットアップ
- [ ] ディレクトリ構造作成
- [ ] 型定義ファイル作成

**コマンド:**
```bash
# Reactプロジェクト作成
cd rag-sample
npm create vite@latest frontend -- --template react-ts

cd frontend
npm install

# 依存関係インストール
npm install \
  react-router-dom \
  @tanstack/react-query \
  zustand \
  axios \
  react-markdown \
  react-syntax-highlighter \
  react-hot-toast

# TailwindCSS セットアップ
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# 開発ツール
npm install -D \
  @types/react-syntax-highlighter \
  eslint \
  prettier \
  @typescript-eslint/eslint-plugin \
  @typescript-eslint/parser
```

### Phase 2: バックエンドAPI実装（3-5日）

**タスク:**
- [ ] FastAPI アプリケーション基盤
- [ ] セッション管理サービス
- [ ] チャットAPIルーター
- [ ] クエリAPIルーター
- [ ] ドキュメントAPIルーター
- [ ] エラーハンドリング
- [ ] CORS設定
- [ ] APIドキュメント（自動生成）

### Phase 3: フロントエンド基本実装（5-7日）

**タスク:**
- [ ] APIクライアント実装
- [ ] カスタムフック実装
- [ ] Zustandストア実装
- [ ] 基本コンポーネント実装
  - [ ] Button, Input, Loading, ErrorMessage
- [ ] チャットページ実装
  - [ ] ChatInterface
  - [ ] MessageList
  - [ ] MessageItem
  - [ ] MessageInput
  - [ ] SessionSidebar
- [ ] Markdownレンダリング
- [ ] ルーティング設定

### Phase 4: UI/UX改善（3-5日）

**タスク:**
- [ ] レスポンシブデザイン
- [ ] ダークモード実装
- [ ] アニメーション追加
- [ ] ローディングステート改善
- [ ] エラーハンドリングUI
- [ ] 通知システム（Toast）
- [ ] アクセシビリティ対応

### Phase 5: 拡張機能（5-7日）

**タスク:**
- [ ] 検索ページ実装
- [ ] ドキュメント管理ページ実装
- [ ] ファイルアップロードUI
- [ ] WebSocket対応（ストリーミング）
- [ ] セッション永続化
- [ ] パラメータ調整UI

### Phase 6: テスト・最適化（3-5日）

**タスク:**
- [ ] ユニットテスト（Vitest）
- [ ] コンポーネントテスト（React Testing Library）
- [ ] E2Eテスト（Playwright）
- [ ] パフォーマンス最適化
- [ ] バンドルサイズ最適化
- [ ] SEO対応

## 設定ファイル例

### `frontend/vite.config.ts`

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
});
```

### `frontend/tailwind.config.js`

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          100: '#dbeafe',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
      },
    },
  },
  plugins: [require('@tailwindcss/typography')],
};
```

### `frontend/package.json`

```json
{
  "name": "rag-chat-frontend",
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview",
    "lint": "eslint . --ext ts,tsx --report-unused-disable-directives --max-warnings 0",
    "format": "prettier --write \"src/**/*.{ts,tsx}\"",
    "test": "vitest",
    "test:ui": "vitest --ui"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.24.0",
    "@tanstack/react-query": "^5.45.0",
    "zustand": "^4.5.2",
    "axios": "^1.7.2",
    "react-markdown": "^9.0.1",
    "react-syntax-highlighter": "^15.5.0",
    "react-hot-toast": "^2.4.1"
  },
  "devDependencies": {
    "@types/react": "^18.3.3",
    "@types/react-dom": "^18.3.0",
    "@types/react-syntax-highlighter": "^15.5.13",
    "@vitejs/plugin-react": "^4.3.1",
    "typescript": "^5.5.2",
    "vite": "^5.3.1",
    "tailwindcss": "^3.4.4",
    "autoprefixer": "^10.4.19",
    "postcss": "^8.4.38",
    "@tailwindcss/typography": "^0.5.13",
    "eslint": "^8.57.0",
    "prettier": "^3.3.2",
    "@typescript-eslint/eslint-plugin": "^7.13.0",
    "@typescript-eslint/parser": "^7.13.0",
    "vitest": "^1.6.0",
    "@testing-library/react": "^15.0.7"
  }
}
```

## 開発・実行コマンド

### 開発環境

```bash
# バックエンド起動
cd rag-sample
uv run uvicorn src.web.app:app --reload --port 8000

# フロントエンド起動（別ターミナル）
cd frontend
npm run dev

# ブラウザで開く
open http://localhost:3000
```

### 本番ビルド

```bash
# フロントエンドビルド
cd frontend
npm run build

# バックエンドでビルド済みファイルをサーブ
# FastAPIで dist/ ディレクトリを静的ファイルとして配信
```

## Docker構成

### `docker-compose.yml`

```yaml
version: '3.8'

services:
  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama-data:/root/.ollama

  backend:
    build:
      context: .
      dockerfile: backend/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - CHROMA_PERSIST_DIRECTORY=/data/chroma_db
    volumes:
      - ./chroma_db:/data/chroma_db
    depends_on:
      - ollama

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:80"
    depends_on:
      - backend

volumes:
  ollama-data:
```

### `frontend/Dockerfile`

```dockerfile
# ビルドステージ
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# 本番ステージ
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

## まとめ

### React + TypeScript 実装のメリット

1. **型安全性** - TypeScriptによるバグの早期発見
2. **保守性** - コンポーネントベースの構造で保守しやすい
3. **開発体験** - 優れたIDEサポート、自動補完
4. **エコシステム** - 豊富なライブラリとツール
5. **スケーラビリティ** - 大規模アプリへの拡張が容易

### 推奨実装順序

**合計開発期間: 3-4週間**

1. **Week 1**: プロジェクトセットアップ + バックエンドAPI
2. **Week 2**: フロントエンド基本実装
3. **Week 3**: UI/UX改善 + 拡張機能
4. **Week 4**: テスト・最適化・デプロイ

### 次のステップ

実装を開始する場合は、以下の順序で進めます:

1. ✅ Reactプロジェクト初期化
2. ✅ 依存関係インストール
3. ✅ ディレクトリ構造作成
4. ✅ 型定義ファイル作成
5. ✅ APIクライアント実装
6. ✅ カスタムフック実装
7. ✅ コンポーネント実装

**実装を開始しますか？** Phase 1のプロジェクトセットアップから進めることができます。
