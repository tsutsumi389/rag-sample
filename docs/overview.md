# RAG CLI アプリケーション 概要

## プロジェクト概要

Python + Ollama + LangChain + ChromaDBを使用した**マルチモーダルRAG（Retrieval-Augmented Generation）システム**のCLIアプリケーションです。

ローカルのテキストドキュメント**と画像**をベクトルデータベースに格納し、自然言語での質問に対して関連情報を検索・生成して回答します。

**主な特徴:**
- テキストドキュメント（PDF, Markdown, TXTなど）の処理
- **画像ファイル（JPEG, PNG, GIF, WebPなど）の処理**
- **ビジョンモデル（llava）による画像キャプション自動生成**
- **マルチモーダル検索（テキスト+画像の統合検索）**
- 対話形式での質問応答
- 完全ローカル実行（Ollama使用）

## 技術スタック

- **Python 3.13+** - メイン開発言語（uvパッケージマネージャー使用）
- **Ollama** - ローカルLLMの実行環境
  - テキストLLM: `gpt-oss`
  - テキスト埋め込み: `nomic-embed-text`
  - ビジョンモデル: `llava` (画像埋め込み・キャプション生成)
  - マルチモーダルLLM: `gemma3` (テキスト+画像対応)
- **LangChain** - LLMアプリケーション開発フレームワーク（langchain, langchain-community, langchain-ollama）
- **ChromaDB** - ベクトルデータベース（組み込みモード、別サーバー不要）
- **Click** - CLIフレームワーク
- **Rich** - ターミナルUI整形
- **Pillow** - 画像処理ライブラリ

## 主要機能

### 1. ドキュメント/画像管理系

#### `add <file_or_directory>`
- ファイルまたはディレクトリをベクトルDBに追加
- **対応フォーマット**:
  - テキスト: `.txt`, `.md`, `.pdf` など
  - 画像: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp`
- 拡張子から自動判定（テキスト/画像）
- テキスト: 自動的にチャンク分割して埋め込みを生成
- 画像: ビジョンモデルでキャプション自動生成、埋め込み生成
- オプション:
  - `--caption <text>`: 画像に手動でキャプションを指定
  - `--tags <tag1,tag2>`: 画像にタグを付与

#### `remove <document_id>`
- 特定のドキュメントをベクトルDBから削除
- ドキュメントIDで指定

#### `list`
- 登録済みドキュメントの一覧を表示
- ドキュメント名、ID、追加日時などを表示

#### `clear`
- すべてのドキュメントをクリア
- ベクトルDBを初期化

#### `clear-images`
- 画像コレクションのみをクリア
- テキストドキュメントは保持

### 2. 検索・質問系

#### `query <question>`
- RAGを使って質問に回答（メインコマンド）
- 関連ドキュメントを検索し、LLMで回答を生成
- 参照元のドキュメントも表示

#### `search <keyword>`
- キーワードで類似テキストドキュメントを検索
- RAG実行なし、関連チャンクのみを表示

#### `search-images <query>`
- テキストクエリで画像を検索
- キャプションベースの類似画像検索
- 画像ファイル名、キャプション、類似度スコアを表示
- オプション:
  - `-k, --top-k <number>`: 検索結果数（デフォルト: 5）
  - `--show-path`: 画像のフルパスを表示
  - `-v, --verbose`: 詳細情報を表示

#### `search-multimodal <query>`
- テキストと画像を統合したマルチモーダル検索
- テキストコレクションと画像コレクションの両方を検索
- 重み付けしたスコアで統合結果を返す
- オプション:
  - `-k, --top-k <number>`: 検索結果数（デフォルト: 10）
  - `--text-weight <0.0-1.0>`: テキスト検索の重み（デフォルト: 0.5）
  - `--image-weight <0.0-1.0>`: 画像検索の重み（デフォルト: 0.5）
  - `--show-content`: 検索結果の内容を表示
  - `-v, --verbose`: 詳細情報を表示

#### `chat`
- 対話モードに入る
- 連続して質問可能
- `exit` または `quit` で終了
- `clear` で履歴をクリア

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
# テキストRAG用モデル
ollama pull gpt-oss
ollama pull nomic-embed-text

# マルチモーダルRAG用モデル（画像機能を使う場合）
ollama pull llava         # ビジョンモデル（画像埋め込み・キャプション生成）
ollama pull gemma3        # マルチモーダルLLM（画像を含む質問応答）

# 3. 環境変数の設定（オプション）
cp .env.sample .env
# .envファイルを編集して設定をカスタマイズ

# 4. システムの初期化
uv run rag init
```

### 基本的な使い方

```bash
# テキストドキュメント追加
uv run rag add ./docs
uv run rag add manual.pdf

# 画像追加（拡張子から自動判定）
uv run rag add image.jpg
uv run rag add ./images              # ディレクトリ内の画像を一括追加

# 画像追加（手動でキャプション指定）
uv run rag add photo.png --caption "プロジェクトの概要図" --tags "図解,概要"

# 登録確認
uv run rag list

# 質問
uv run rag query "このシステムの使い方は？"

# キーワード検索
uv run rag search "インストール方法"

# 画像検索
uv run rag search-images "犬の写真"

# マルチモーダル検索（テキスト+画像）
uv run rag search-multimodal "Pythonのコード例"
uv run rag search-multimodal "機械学習" --text-weight 0.7 --image-weight 0.3

# ドキュメント削除
uv run rag remove <document_id>

# 全ドキュメントクリア
uv run rag clear

# 画像のみクリア
uv run rag clear-images
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
│   │   ├── document.py           # add, remove, list, clear, clear-images
│   │   ├── query.py              # query, search, search-images, search-multimodal, chat
│   │   └── config.py             # init, status, config
│   ├── rag/                      # RAGコア機能
│   │   ├── __init__.py
│   │   ├── vector_store.py       # ChromaDB操作（PersistentClient、マルチモーダル対応）
│   │   ├── embeddings.py         # Ollamaテキスト埋め込み生成
│   │   ├── vision_embeddings.py  # Ollamaビジョン埋め込み生成（画像）
│   │   ├── document_processor.py # テキストファイル読み込み・チャンク分割
│   │   ├── image_processor.py    # 画像ファイル処理
│   │   ├── engine.py             # テキストRAGオーケストレーション
│   │   └── multimodal_engine.py  # マルチモーダルRAGエンジン（テキスト+画像）
│   ├── utils/                    # ユーティリティ
│   │   ├── __init__.py
│   │   ├── config.py             # 設定管理（python-dotenv）
│   │   ├── file_handler.py       # ファイル操作
│   │   └── logger.py             # ロギング設定
│   └── models/                   # データモデル
│       ├── __init__.py
│       └── document.py           # Document, Chunk, SearchResult, ImageDocument, ChatMessage
├── tests/                        # テストコード
│   ├── conftest.py               # pytest共通フィクスチャ
│   ├── unit/                     # ユニットテスト
│   │   ├── test_embeddings.py
│   │   ├── test_vision_embeddings.py
│   │   ├── test_vector_store.py
│   │   ├── test_document_processor.py
│   │   ├── test_image_processor.py
│   │   ├── test_engine.py
│   │   ├── test_multimodal_engine.py
│   │   └── test_config.py
│   └── integration/              # 統合テスト
│       ├── test_ollama_integration.py
│       ├── test_full_rag_flow.py
│       ├── test_image_search.py
│       ├── test_multimodal_rag.py
│       └── test_multimodal_search.py
├── docs/                         # ドキュメント
│   ├── overview.md               # 機能仕様
│   ├── implementation-plan.md    # 実装計画
│   └── multimodal-rag-implementation-plan.md  # マルチモーダルRAG実装計画
├── chroma_db/                    # ChromaDBデータ（.gitignore）
│   ├── documents/                # テキストドキュメントコレクション
│   └── images/                   # 画像コレクション
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

### 1. テキストドキュメント登録フロー

```
テキストファイル入力 → ロード（document_processor）
                   ↓
            チャンク分割（RecursiveCharacterTextSplitter）
                   ↓
            埋め込み生成（embeddings / Ollama nomic-embed-text）
                   ↓
            ChromaDB documentsコレクションに保存（vector_store）
```

### 2. 画像登録フロー

```
画像ファイル入力 → バリデーション（image_processor）
              ↓
        キャプション生成（vision_embeddings / Ollama llava）
              ↓
        埋め込み生成（キャプションベース / embeddings）
              ↓
        ChromaDB imagesコレクションに保存（vector_store）
```

### 3. テキスト質問応答フロー

```
ユーザー質問 → クエリ埋め込み（embeddings）
            ↓
    類似チャンク検索（vector_store / documentsコレクション）
            ↓
    コンテキスト取得 → LLM回答生成（engine / gpt-oss）
            ↓
         回答返却
```

### 4. マルチモーダル検索フロー

```
検索クエリ → クエリ埋め込み（embeddings）
          ↓
    ┌─────┴─────┐
    │           │
テキスト検索  画像検索
(documents) (images)
    │           │
    └─────┬─────┘
          ↓
    重み付けスコア計算
          ↓
    統合結果をソート
          ↓
    Top-K件を返却
```

### 5. 対話セッションフロー

```
chat開始 → 会話履歴初期化
        ↓
  ユーザー入力 → RAG処理 → 回答 → 履歴保存
        ↓                            ↑
    exit/quit ←──────────────────────┘
    または clear（履歴クリア）
        ↓
      終了
```

## 環境変数設定

アプリケーションは `.env` ファイルで設定を管理します（`.env.sample` を参照）：

```bash
# Ollama設定
OLLAMA_BASE_URL=http://localhost:11434

# テキストRAG用モデル
OLLAMA_LLM_MODEL=gpt-oss               # 回答生成モデル
OLLAMA_EMBEDDING_MODEL=nomic-embed-text # テキスト埋め込みモデル

# マルチモーダルRAG用モデル（画像機能を使う場合）
OLLAMA_VISION_MODEL=llava              # ビジョンモデル（画像埋め込み・キャプション生成）
OLLAMA_MULTIMODAL_LLM_MODEL=gemma3     # マルチモーダルLLM（画像を含む質問応答）

# ChromaDB設定
CHROMA_PERSIST_DIRECTORY=./chroma_db

# ドキュメント処理設定
CHUNK_SIZE=1000           # チャンクサイズ（100-10000）
CHUNK_OVERLAP=200         # チャンク間オーバーラップ

# 画像処理設定
IMAGE_CAPTION_AUTO_GENERATE=true      # 画像追加時に自動でキャプション生成
MAX_IMAGE_SIZE_MB=10                  # 画像ファイルサイズ上限
IMAGE_RESIZE_ENABLED=false            # 画像リサイズ有効化（大容量対策）
IMAGE_RESIZE_MAX_WIDTH=1024           # リサイズ後の最大幅
IMAGE_RESIZE_MAX_HEIGHT=1024          # リサイズ後の最大高さ

# マルチモーダル検索設定
MULTIMODAL_SEARCH_TEXT_WEIGHT=0.5     # テキスト検索の重み
MULTIMODAL_SEARCH_IMAGE_WEIGHT=0.5    # 画像検索の重み

# ログ設定
LOG_LEVEL=INFO           # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### ChromaDBの使用パターン

- `PersistentClient` を使用（`Client` ではない）
- デフォルト保存先: `./chroma_db`
- 別サーバープロセス不要（組み込みモード）
- **コレクション構成**:
  - `documents`: テキストドキュメント用コレクション
  - `images`: 画像用コレクション（キャプション、メタデータ、埋め込み）
- コレクションごとに独立したベクトル空間を管理

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
