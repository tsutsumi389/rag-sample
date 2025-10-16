# RAG CLI アプリケーション 概要

## プロジェクト概要

Python + Ollama + LangChain + ChromaDBを使用したRAG（Retrieval-Augmented Generation）システムのCLIアプリケーションです。
ローカルのドキュメントをベクトルデータベースに格納し、自然言語での質問に対して関連情報を検索・生成して回答します。

## 技術スタック

- **Python 3.13+** - メイン開発言語（uvパッケージマネージャー使用）
- **Ollama** - ローカルLLMの実行環境（gpt-oss / nomic-embed-text）
- **LangChain** - LLMアプリケーション開発フレームワーク（langchain, langchain-community, langchain-ollama）
- **ChromaDB** - ベクトルデータベース（組み込みモード、別サーバー不要）
- **Click** - CLIフレームワーク
- **Rich** - ターミナルUI整形

## 主要機能

### 1. ドキュメント管理系

#### `add <file_or_directory>`
- ファイルまたはディレクトリをベクトルDBに追加
- 対応フォーマット: テキスト、PDF、Markdown など
- 自動的にチャンク分割して埋め込みを生成

#### `remove <document_id>`
- 特定のドキュメントをベクトルDBから削除
- ドキュメントIDで指定

#### `list`
- 登録済みドキュメントの一覧を表示
- ドキュメント名、ID、追加日時などを表示

#### `clear`
- すべてのドキュメントをクリア
- ベクトルDBを初期化

### 2. 検索・質問系

#### `query <question>`
- RAGを使って質問に回答（メインコマンド）
- 関連ドキュメントを検索し、LLMで回答を生成
- 参照元のドキュメントも表示

#### `search <keyword>`
- キーワードで類似ドキュメントを検索
- RAG実行なし、関連チャンクのみを表示

#### `chat`
- 対話モードに入る
- 連続して質問可能
- `exit` または `quit` で終了

### 3. 設定・管理系

#### `init`
- 初期セットアップ
- ChromaDBディレクトリの作成
- 環境変数の確認

#### `status`
- システムの状態確認
- Ollama接続状態
- ChromaDB接続状態
- 登録ドキュメント数など

#### `config`
- 現在の環境変数設定を表示
- Ollamaモデル名
- チャンクサイズ・オーバーラップ
- ChromaDBパスなど

### 4. その他機能

#### `--version`
- バージョン情報を表示
- 使用技術スタックの表示

#### `--verbose` / `-v`
- 詳細ログ出力モード
- デバッグ情報の表示
- 各コマンドで使用可能

**注意**: 現在の実装では `export`, `import`, `stats`, `config set` は未実装です。設定は `.env` ファイルで管理します。

## 使用例

### 環境構築

```bash
# 1. 依存関係のインストール
uv sync

# 2. Ollamaモデルの取得
ollama pull gpt-oss
ollama pull nomic-embed-text

# 3. 環境変数の設定（オプション）
cp .env.sample .env
# .envファイルを編集して設定をカスタマイズ

# 4. システムの初期化
uv run rag init
```

### 基本的な使い方

```bash
# ドキュメント追加
uv run rag add ./docs
uv run rag add manual.pdf

# 登録確認
uv run rag list

# 質問
uv run rag query "このシステムの使い方は？"

# キーワード検索
uv run rag search "インストール方法"

# ドキュメント削除
uv run rag remove <document_id>

# 全ドキュメントクリア
uv run rag clear
```

### 対話モード

```bash
uv run rag chat

> このドキュメントには何が書かれていますか？
[回答が表示される]

> もっと詳しく教えてください
[回答が表示される]

> exit
```

### システム管理

```bash
# 現在の設定を確認
uv run rag config

# システムステータス確認
uv run rag status

# バージョン情報表示
uv run rag --version

# 詳細ログ出力
uv run rag --verbose query "質問内容"
```

## アーキテクチャ

### ディレクトリ構造

```
rag-sample/
├── src/
│   ├── cli.py                    # CLIエントリーポイント（Click）
│   ├── commands/                 # 各コマンドの実装
│   │   ├── __init__.py
│   │   ├── document.py           # add, remove, list, clear
│   │   ├── query.py              # query, search, chat
│   │   └── config.py             # init, status, config
│   ├── rag/                      # RAGコア機能
│   │   ├── __init__.py
│   │   ├── vector_store.py       # ChromaDB操作（PersistentClient）
│   │   ├── embeddings.py         # Ollama埋め込み生成
│   │   ├── document_processor.py # ファイル読み込み・チャンク分割
│   │   └── engine.py             # RAGオーケストレーション（検索+生成）
│   ├── utils/                    # ユーティリティ
│   │   ├── __init__.py
│   │   ├── config.py             # 設定管理（python-dotenv）
│   │   ├── file_handler.py       # ファイル操作
│   │   └── logger.py             # ロギング設定
│   └── models/                   # データモデル
│       ├── __init__.py
│       └── document.py           # Document, Chunk, SearchResult, ChatMessage
├── tests/                        # テストコード
│   ├── conftest.py               # pytest共通フィクスチャ
│   ├── unit/                     # ユニットテスト
│   │   ├── test_embeddings.py
│   │   ├── test_vector_store.py
│   │   ├── test_document_processor.py
│   │   ├── test_engine.py
│   │   └── test_config.py
│   └── integration/              # 統合テスト
│       ├── test_ollama_integration.py
│       └── test_full_rag_flow.py
├── docs/                         # ドキュメント
│   ├── overview.md               # 機能仕様
│   └── implementation-plan.md    # 実装計画
├── chroma_db/                    # ChromaDBデータ（.gitignore）
├── .env.sample                   # 環境変数サンプル
├── .env                          # 環境変数（.gitignore）
├── pyproject.toml                # プロジェクト設定・依存関係
├── uv.lock                       # 依存関係ロックファイル
└── README.md
```

### レイヤー構造

アプリケーションは以下のレイヤーに分離されています：

1. **CLIレイヤー** ([src/cli.py](src/cli.py), [src/commands/](src/commands/))
   - ユーザーインターフェース
   - コマンド処理
   - Rich による整形出力

2. **RAGコアレイヤー** ([src/rag/](src/rag/))
   - ビジネスロジック
   - ドキュメント処理
   - 検索・生成機能

3. **データモデルレイヤー** ([src/models/](src/models/))
   - 共有データ構造
   - 型定義

4. **ユーティリティレイヤー** ([src/utils/](src/utils/))
   - 横断的関心事
   - 設定管理、ログ、ファイル処理

## ワークフロー

### 1. ドキュメント登録フロー

```
ファイル入力 → ロード（document_processor）
           ↓
    チャンク分割（RecursiveCharacterTextSplitter）
           ↓
    埋め込み生成（embeddings / Ollama）
           ↓
    ChromaDBに保存（vector_store / PersistentClient）
```

### 2. 質問応答フロー

```
ユーザー質問 → クエリ埋め込み（embeddings）
            ↓
    類似チャンク検索（vector_store）
            ↓
    コンテキスト取得 → LLM回答生成（engine）
            ↓
         回答返却
```

### 3. 対話セッションフロー

```
chat開始 → 会話履歴初期化
        ↓
  ユーザー入力 → RAG処理 → 回答 → 履歴保存
        ↓                            ↑
    exit/quit ←──────────────────────┘
        ↓
      終了
```

## 環境変数設定

アプリケーションは `.env` ファイルで設定を管理します（`.env.sample` を参照）：

```bash
# Ollama設定
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=gpt-oss               # 回答生成モデル
OLLAMA_EMBEDDING_MODEL=nomic-embed-text # 埋め込みモデル

# ChromaDB設定
CHROMA_PERSIST_DIRECTORY=./chroma_db

# ドキュメント処理設定
CHUNK_SIZE=1000           # チャンクサイズ（100-10000）
CHUNK_OVERLAP=200         # チャンク間オーバーラップ

# ログ設定
LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### ChromaDBの使用パターン

- `PersistentClient` を使用（`Client` ではない）
- デフォルト保存先: `./chroma_db`
- 別サーバープロセス不要（組み込みモード）
- コレクションはドキュメントタイプやプロジェクトごとに整理

## 実装状況

### ✅ Phase 1-3: コア実装完了

**実装済みコンポーネント:**

#### CLIレイヤー
- [src/cli.py](../src/cli.py) - Clickによるメインエントリーポイント
- [src/commands/document.py](../src/commands/document.py) - ドキュメント管理（add, remove, list, clear）
- [src/commands/query.py](../src/commands/query.py) - クエリ操作（query, search, chat）
- [src/commands/config.py](../src/commands/config.py) - 設定コマンド（init, status, config）

#### RAGコア
- [src/rag/vector_store.py](../src/rag/vector_store.py) - ChromaDBベクトル操作
- [src/rag/embeddings.py](../src/rag/embeddings.py) - Ollama埋め込み生成
- [src/rag/document_processor.py](../src/rag/document_processor.py) - ドキュメント読み込み・チャンク分割
- [src/rag/engine.py](../src/rag/engine.py) - RAGオーケストレーション（会話履歴サポート付き）

#### データモデル
- [src/models/document.py](../src/models/document.py) - Document, Chunk, SearchResult, ChatMessage

#### ユーティリティ
- [src/utils/config.py](../src/utils/config.py) - .envサポート付き設定管理

**テストカバレッジ:**
- 全コアモジュールのユニットテスト（[tests/unit/](../tests/unit/)）
- Ollama統合テスト（[tests/integration/test_ollama_integration.py](../tests/integration/test_ollama_integration.py)）
- 完全RAGフローE2Eテスト（[tests/integration/test_full_rag_flow.py](../tests/integration/test_full_rag_flow.py)）
- pytest共通フィクスチャ（[tests/conftest.py](../tests/conftest.py)）

### 今後の拡張案

- [ ] パフォーマンス最適化と本番環境対応
- [ ] バッチ操作（複数ファイル一括処理）
- [ ] エクスポート/インポート機能
- [ ] 統計情報表示（stats コマンド）
- [ ] Web URLからのドキュメント取得
- [ ] 複数コレクションのサポート
- [ ] 会話履歴の永続化
- [ ] より多様なファイル形式のサポート（Word, Excel等）
- [ ] GUI版の開発
- [ ] クラウドベクトルDBへの対応

## 開発ガイドライン

### テスト実行

```bash
# 全テスト実行
uv run pytest

# ユニットテストのみ
uv run pytest tests/unit/

# 統合テストのみ
uv run pytest tests/integration/

# カバレッジレポート付き
uv run pytest --cov=src --cov-report=term-missing

# 特定のテストファイル
uv run pytest tests/unit/test_engine.py -v
```

### コーディング規約

- **コメントとdocstringは日本語で記述**
- Python 3.13+ の型ヒント使用（例: `dict[str, Any]`）
- Google スタイルのdocstring（日本語）
- モジュールdocstringで目的と主要コンポーネントを説明
- 依存性注入パターンを使用
- テストカバレッジ80%以上を目標

## ライセンス

MIT License
