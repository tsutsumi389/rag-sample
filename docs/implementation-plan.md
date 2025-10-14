# RAGアプリケーション実装計画

## プロジェクト概要
Python + Ollama + LangChain + ChromaDBを使用したRAG（Retrieval-Augmented Generation）CLIアプリケーション

## 実装タスク

### フェーズ1: 基礎構造
#### タスク1: プロジェクト構造（ディレクトリ・ファイル）を作成
- **目的**: overview.mdに基づいた基本的なプロジェクト構造を構築
- **成果物**:
  ```
  src/
  ├── cli.py              # CLIエントリーポイント
  ├── commands/           # コマンド実装
  │   ├── __init__.py
  │   ├── document.py     # add, remove, list, clear
  │   ├── query.py        # query, search, chat
  │   └── config.py       # config, status, init
  ├── rag/               # RAGコア機能
  │   ├── __init__.py
  │   ├── vector_store.py # ChromaDB操作
  │   ├── embeddings.py   # 埋め込み生成
  │   ├── document_processor.py # ドキュメント処理
  │   └── engine.py       # RAGエンジン
  ├── utils/             # ユーティリティ
  │   ├── __init__.py
  │   ├── file_handler.py # ファイル操作
  │   └── logger.py       # ログ出力
  └── models/            # データモデル
      ├── __init__.py
      └── document.py     # Document, Chunk等
  ```
- **所要時間**: 10分

#### タスク2: 設定管理モジュール（config.py）を実装
- **目的**: アプリケーション全体で使用する設定を一元管理
- **実装内容**:
  - 環境変数の読み込み（python-dotenv使用）
  - デフォルト設定の定義
  - 設定値のバリデーション
- **主要な設定項目**:
  - Ollamaのモデル名（LLM、Embedding）
  - ChromaDBの保存パス
  - チャンク設定（サイズ、オーバーラップ）
  - ログレベル
- **所要時間**: 15分

#### タスク3: データモデル（models/）を実装
- **目的**: アプリケーション内で使用するデータ構造を定義
- **実装クラス**:
  - `Document`: ドキュメントの基本情報
  - `Chunk`: 分割されたテキストチャンク
  - `SearchResult`: 検索結果
  - `ChatMessage`: チャット履歴
- **使用ライブラリ**: dataclasses または pydantic
- **所要時間**: 20分

### フェーズ2: RAGコア機能
#### タスク4: ChromaDB接続・初期化モジュール（rag/vector_store.py）を実装
- **目的**: ベクトルデータベースの管理・操作
- **実装内容**:
  - ChromaDBクライアントの初期化（PersistentClient）
  - コレクションの作成・取得
  - ドキュメントの追加・削除・検索
  - メタデータフィルタリング
- **主要メソッド**:
  - `initialize()`: DB初期化
  - `add_documents()`: ドキュメント追加
  - `search()`: 類似度検索
  - `delete()`: ドキュメント削除
  - `list_documents()`: ドキュメント一覧取得
- **所要時間**: 30分

#### タスク5: ドキュメント処理モジュール（rag/document_processor.py）を実装
- **目的**: ファイルの読み込みとテキスト分割
- **実装内容**:
  - ファイル読み込み（TXT, PDF, MD等）
  - テキスト分割（RecursiveCharacterTextSplitter）
  - メタデータ付与（ファイル名、パス、タイムスタンプ等）
- **対応フォーマット**:
  - テキストファイル (.txt, .md)
  - PDF (.pdf) - PyPDF2またはpdfplumber
  - 将来的に: Word, Excel等
- **主要メソッド**:
  - `load_document()`: ファイル読み込み
  - `split_text()`: テキスト分割
  - `create_chunks()`: チャンク作成
- **所要時間**: 30分

#### タスク6: 埋め込み生成モジュール（rag/embeddings.py）を実装
- **目的**: Ollamaを使用したembedding生成
- **実装内容**:
  - OllamaEmbeddingsの初期化
  - テキストのベクトル化
  - バッチ処理対応
- **使用モデル**: nomic-embed-text（デフォルト）
- **主要メソッド**:
  - `embed_documents()`: 複数ドキュメントの埋め込み
  - `embed_query()`: クエリの埋め込み
- **所要時間**: 20分

#### タスク7: RAGエンジン（rag/engine.py）を実装
- **目的**: 検索・質問応答のコアロジック
- **実装内容**:
  - コンテキスト検索（類似度ベース）
  - プロンプト生成
  - LLMによる回答生成（Ollama経由）
  - チャット履歴管理
- **主要メソッド**:
  - `retrieve()`: 関連ドキュメント検索
  - `generate_answer()`: 回答生成
  - `chat()`: 会話形式の質問応答
- **使用モデル**: llama3.2（デフォルト）
- **所要時間**: 40分

### フェーズ3: CLIインターフェース
#### タスク8: CLIコマンド群（commands/）を実装
- **目的**: ユーザーが実行する各コマンドの実装

##### 8-1: ドキュメント管理コマンド（commands/document.py）
- `add`: ドキュメント追加
- `remove`: ドキュメント削除
- `list`: ドキュメント一覧表示
- `clear`: 全ドキュメント削除

##### 8-2: 検索・質問コマンド（commands/query.py）
- `query`: 質問応答
- `search`: ドキュメント検索
- `chat`: 対話モード

##### 8-3: 設定・管理コマンド（commands/config.py）
- `init`: 初期化
- `status`: ステータス表示
- `config`: 設定変更

- **所要時間**: 60分

#### タスク9: CLIエントリーポイント（cli.py）を実装
- **目的**: メインのCLIアプリケーション
- **実装内容**:
  - Clickグループの設定
  - 各コマンドの登録
  - グローバルオプション（--verbose等）
  - Rich使用による見やすい出力
- **所要時間**: 30分

### フェーズ4: 仕上げ
#### タスク10: 環境設定ファイル（.env.example）を作成
- **目的**: 環境変数のテンプレート提供
- **内容**:
  ```
  # Ollama設定
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_LLM_MODEL=llama3.2
  OLLAMA_EMBEDDING_MODEL=nomic-embed-text

  # ChromaDB設定
  CHROMA_PERSIST_DIRECTORY=./chroma_db

  # チャンク設定
  CHUNK_SIZE=1000
  CHUNK_OVERLAP=200

  # ログ設定
  LOG_LEVEL=INFO
  ```
- **所要時間**: 5分

#### タスク11: 基本的な動作テストを実施
- **目的**: 各機能の動作確認
- **テスト内容**:
  1. `rag init`: 初期化テスト
  2. `rag add sample.txt`: ドキュメント追加テスト
  3. `rag list`: 一覧表示テスト
  4. `rag query "質問"`: 質問応答テスト
  5. `rag search "キーワード"`: 検索テスト
  6. `rag chat`: 対話モードテスト
- **所要時間**: 30分

## 総所要時間
約4.5時間（実装のみ、テスト・デバッグ除く）

## 前提条件
- [x] Python 3.13環境構築完了
- [x] 必要なライブラリインストール完了
  - langchain, langchain-community, langchain-ollama
  - chromadb
  - click, rich, python-dotenv
- [ ] Ollamaのインストール・起動
- [ ] 必要なモデルのダウンロード
  - `ollama pull llama3.2`
  - `ollama pull nomic-embed-text`

## 実装の優先順位
1. **最優先**: フェーズ1（基礎構造）
2. **高優先**: フェーズ2（RAGコア機能）
3. **中優先**: フェーズ3（CLIインターフェース）
4. **低優先**: フェーズ4（仕上げ）

## 次のアクション
タスク1「プロジェクト構造の作成」から開始
