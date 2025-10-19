# RAG CLI アプリケーション

Python + Ollama + LangChain + ChromaDBを使用したRAG（Retrieval-Augmented Generation）システムのCLIアプリケーションです。
ローカルのドキュメントをベクトルデータベースに格納し、自然言語での質問に対して関連情報を検索・生成して回答します。

[![Python Version](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 特徴

- **完全ローカル実行** - インターネット接続不要、データは外部に送信されません
- **柔軟なドキュメント管理** - TXT、PDF、Markdownなど多様なファイル形式に対応
- **高速な検索** - ChromaDBによる効率的なベクトル検索
- **対話的なチャット** - 会話履歴を考慮した継続的な対話が可能
- **カスタマイズ可能** - チャンクサイズ、モデル、検索パラメータなど細かく調整可能

## 技術スタック

- **言語**: Python 3.13+
- **パッケージ管理**: uv
- **LLM**: Ollama (gpt-oss)
- **埋め込みモデル**: nomic-embed-text
- **ベクトルDB**: ChromaDB (組み込み型、サーバー不要)
- **ライブラリ**: LangChain, Click, Rich

## 目次

- [セットアップ](#セットアップ)
  - [前提条件](#前提条件)
  - [インストール手順](#インストール手順)
- [使い方](#使い方)
  - [基本的な流れ](#基本的な流れ)
  - [コマンド一覧](#コマンド一覧)
- [詳細ガイド](#詳細ガイド)
  - [初期化](#初期化)
  - [ドキュメント管理](#ドキュメント管理)
  - [質問と検索](#質問と検索)
  - [システム設定](#システム設定)
- [設定のカスタマイズ](#設定のカスタマイズ)
- [MCPサーバー](#mcpサーバー)
  - [MCPとは](#mcpとは)
  - [MCPサーバーのセットアップ](#mcpサーバーのセットアップ)
  - [提供機能](#提供機能)
- [開発者向け情報](#開発者向け情報)

## セットアップ

### 前提条件

このアプリケーションを実行するには、以下が必要です：

1. **Python 3.13以上**
   ```bash
   python --version  # Python 3.13+ であることを確認
   ```

2. **uv パッケージマネージャ**
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # インストール確認
   uv --version
   ```

3. **Ollama**
   - [公式サイト](https://ollama.ai/)からインストール
   - または Homebrew (macOS):
     ```bash
     brew install ollama
     ```

### インストール手順

#### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd rag-sample
```

#### 2. Ollamaのセットアップ

Ollamaを起動し、必要なモデルをダウンロードします：

```bash
# Ollamaサービスの起動
ollama serve  # バックグラウンドで実行されます

# 別のターミナルで以下を実行
# LLMモデル（回答生成用）
ollama pull gpt-oss

# 埋め込みモデル（ベクトル化用）
ollama pull nomic-embed-text
```

**モデルの確認:**
```bash
ollama list
# gpt-oss と nomic-embed-text が表示されることを確認
```

#### 3. 依存パッケージのインストール

```bash
# 依存パッケージをインストール
uv sync

# または開発環境用（推奨）
uv sync --all-extras
```

#### 4. アプリケーションの初期化

```bash
# CLIアプリケーションの初期化
uv run rag init
```

これにより以下が作成されます：
- `.env` - 環境設定ファイル
- `chroma_db/` - ベクトルデータベースディレクトリ

#### 5. 動作確認

```bash
# ステータス確認
uv run rag status

# 以下のような出力が表示されればOK:
# ✓ Ollama接続: 正常
# ✓ ベクトルDB: 正常
# ✓ モデル: gpt-oss, nomic-embed-text
```

**トラブルシューティング:**
- Ollama接続エラーの場合: `ollama serve` が実行中か確認
- モデルが見つからない場合: `ollama list` でモデルがダウンロード済みか確認

### セットアップ完了！

これでRAG CLIアプリケーションを使用する準備が整いました。[使い方](#使い方)セクションに進んでください。

---

## 使い方

### 基本的な流れ

```bash
# 1. ドキュメントを追加
uv run rag add ./docs/sample.txt

# 2. 追加されたドキュメントを確認
uv run rag list

# 3. 質問してみる
uv run rag query "このドキュメントの内容は？"
```

### コマンド一覧

#### ドキュメント管理

| コマンド | 説明 | 使用例 |
|---------|------|--------|
| `rag add <path>` | ファイルまたはディレクトリを追加 | `uv run rag add ./docs` |
| `rag list` | 登録済みドキュメント一覧を表示 | `uv run rag list` |
| `rag remove <id>` | ドキュメントを削除 | `uv run rag remove doc_123` |
| `rag clear` | すべてのドキュメントを削除 | `uv run rag clear` |

#### 質問と検索

| コマンド | 説明 | 使用例 |
|---------|------|--------|
| `rag query <question>` | 質問に対して回答を生成 | `uv run rag query "RAGとは？"` |
| `rag search <keyword>` | キーワードで関連ドキュメントを検索 | `uv run rag search "機械学習"` |
| `rag chat` | 対話モードを開始 | `uv run rag chat` |

#### システム管理

| コマンド | 説明 | 使用例 |
|---------|------|--------|
| `rag init` | システムを初期化 | `uv run rag init` |
| `rag status` | システムの状態を確認 | `uv run rag status` |
| `rag config` | 現在の設定を表示 | `uv run rag config` |
| `rag config set <key> <value>` | 設定を変更 | `uv run rag config set chunk_size 500` |

#### その他オプション

```bash
# バージョン情報を表示
uv run rag --version

# 詳細ログを表示
uv run rag --verbose <command>

# ヘルプを表示
uv run rag --help
uv run rag <command> --help
```

---

## 詳細ガイド

### 初期化

初めて使用する前に、システムを初期化します：

```bash
uv run rag init
```

これにより以下が作成されます：
- `.env` ファイル（設定）
- `chroma_db/` ディレクトリ（ベクトルDB）

**カスタム設定での初期化:**
```bash
# カスタムデータベースパスを指定
uv run rag init --db-path ./my_data/chroma

# デフォルト設定を上書き
uv run rag init --force
```

### ドキュメント管理

#### ドキュメントの追加

**単一ファイルを追加:**
```bash
uv run rag add document.txt
uv run rag add manual.pdf
```

**ディレクトリを再帰的に追加:**
```bash
uv run rag add ./docs
```

**特定の拡張子のみを追加:**
```bash
# .txt と .md ファイルのみ
uv run rag add ./docs --include "*.txt" --include "*.md"
```

#### ドキュメントの一覧表示

```bash
uv run rag list
```

出力例：
```
┌──────────┬────────────────┬──────────┬─────────────────────┐
│ ID       │ ファイル名      │ チャンク数│ 追加日時            │
├──────────┼────────────────┼──────────┼─────────────────────┤
│ doc_001  │ sample.txt     │ 15       │ 2025-01-15 10:30:00 │
│ doc_002  │ manual.pdf     │ 42       │ 2025-01-15 10:35:00 │
└──────────┴────────────────┴──────────┴─────────────────────┘
```

#### ドキュメントの削除

**特定のドキュメントを削除:**
```bash
uv run rag remove doc_001
```

**すべてのドキュメントを削除:**
```bash
uv run rag clear
```

### 質問と検索

#### 質問応答（query）

最も一般的な使い方です。質問に対してLLMが回答を生成します：

```bash
uv run rag query "RAGシステムの利点は何ですか？"
```

出力例：
```
質問: RAGシステムの利点は何ですか？

回答:
RAGシステムの主な利点は以下の通りです：
1. 最新情報の活用: 外部ドキュメントから最新の情報を取得できます
2. 信頼性の向上: 参照元を明示することで回答の根拠が明確になります
3. 幻覚の抑制: 実際のドキュメントに基づく回答により、誤情報が減少します

参照元:
  • document.txt (類似度: 0.92)
  • manual.pdf (類似度: 0.87)
```

**オプション:**
```bash
# 参照するチャンク数を指定（デフォルト: 3）
uv run rag query "質問内容" --top-k 5

# 回答のみ表示（参照元を非表示）
uv run rag query "質問内容" --no-sources
```

#### キーワード検索（search）

LLMを使わず、関連するドキュメントチャンクのみを検索：

```bash
uv run rag search "機械学習"
```

出力例：
```
検索結果: "機械学習"

1. [類似度: 0.95] document.txt
   機械学習は、コンピュータがデータから学習し、
   明示的にプログラムされることなくタスクを実行...

2. [類似度: 0.88] manual.pdf
   深層学習は機械学習の一分野であり...
```

#### 対話モード（chat）

連続して質問できる対話モードに入ります：

```bash
uv run rag chat
```

使用例：
```
対話モードを開始します。終了するには 'exit' または 'quit' と入力してください。

> このシステムについて教えてください
[AIが回答]

> もっと詳しく
[AIが前の会話を考慮して回答]

> exit
対話を終了しました。
```

### システム設定

#### 現在の設定を確認

```bash
uv run rag config
```

出力例：
```
現在の設定:

[Ollama設定]
  • ベースURL: http://localhost:11434
  • LLMモデル: gpt-oss
  • 埋め込みモデル: nomic-embed-text

[チャンク設定]
  • チャンクサイズ: 1000
  • オーバーラップ: 200

[データベース設定]
  • 保存先: ./chroma_db
```

#### 設定の変更

```bash
# チャンクサイズを変更
uv run rag config set chunk_size 500

# オーバーラップを変更
uv run rag config set chunk_overlap 100

# 使用モデルを変更
uv run rag config set llm_model llama3.1

# Ollama URLを変更
uv run rag config set ollama_base_url http://localhost:11434
```

#### システムステータスの確認

```bash
uv run rag status
```

出力例：
```
システムステータス:

✓ Ollama接続: 正常
✓ ベクトルDB: 正常
✓ LLMモデル (gpt-oss): 利用可能
✓ 埋め込みモデル (nomic-embed-text): 利用可能

統計情報:
  • ドキュメント数: 12
  • 総チャンク数: 342
  • データベースサイズ: 15.3 MB
```

---

## 設定のカスタマイズ

`.env` ファイルを編集することで、より詳細な設定が可能です：

```bash
# .env ファイルを編集
nano .env
```

主な設定項目：

```env
# Ollama設定
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=gpt-oss
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

# ChromaDB設定
CHROMA_PERSIST_DIRECTORY=./chroma_db

# チャンク設定
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# 検索設定
TOP_K=3

# ログレベル
LOG_LEVEL=INFO
```

設定変更後、再起動は不要です。次回実行時に自動的に反映されます。

---

## MCPサーバー

### MCPとは

MCP (Model Context Protocol) は、LLMアプリケーション（Claude Desktopなど）がローカルまたはリモートのデータソースやツールと連携するためのオープンプロトコルです。

このRAGアプリケーションはMCPサーバーとして動作し、Claude Desktopから直接RAG機能を利用できます。

**主な特徴:**
- Claude Desktopの対話中にローカルドキュメントを検索
- ツール呼び出しによるドキュメント管理
- リソースとして登録済みドキュメント一覧を提供

**参考:**
- [MCP公式ドキュメント](https://modelcontextprotocol.io/)
- [Claude Desktop MCP設定ガイド](https://docs.anthropic.com/claude/docs/model-context-protocol)

### MCPサーバーのセットアップ

#### 1. MCPサーバーの起動確認

MCPサーバーは通常、Claude Desktopから自動的に起動されますが、手動で動作確認することもできます：

```bash
# MCPサーバーを起動（stdio経由で通信）
uv run rag-mcp-server
```

**注意:** MCPサーバーはstdio経由で通信するため、手動起動時は対話的に操作できません。Claude Desktopなどのクライアントから接続する必要があります。

#### 2. Claude Desktop設定

Claude DesktopでこのMCPサーバーを使用するには、設定ファイルを編集します：

**macOS:**
```bash
# 設定ファイルを開く
nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

**Windows:**
```bash
# 設定ファイルを開く
notepad %APPDATA%\Claude\claude_desktop_config.json
```

**設定例:**
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

**重要な設定項目:**
- `cwd`: このRAGプロジェクトのディレクトリパスを指定（絶対パス）
- `env.OLLAMA_BASE_URL`: Ollamaが別ポートで動作している場合は変更

#### 3. Claude Desktopの再起動

設定ファイルを保存したら、Claude Desktopを再起動します。正常に接続されると、ツール一覧にRAG関連のツールが表示されます。

#### 4. 動作確認

Claude Desktopで以下のように質問してみてください：

```
登録されているドキュメント一覧を取得してください
```

MCPサーバーが正常に動作していれば、`list_documents` ツールが呼び出され、登録済みドキュメントの一覧が表示されます。

### 提供機能

MCPサーバーは以下のツールとリソースを提供します：

#### ツール（Tools）

| ツール名 | 説明 | 主なパラメータ |
|---------|------|--------------|
| `add_document` | テキストまたは画像ドキュメントを追加 | `file_path` (必須)<br>`caption` (画像の場合)<br>`tags` (画像の場合) |
| `list_documents` | 登録済みドキュメント一覧を取得 | `limit` (件数制限)<br>`include_images` (画像を含むか) |
| `search` | キーワードでドキュメントを検索 | `query` (必須)<br>`top_k` (結果数) |
| `remove_document` | ドキュメントまたは画像を削除 | `item_id` (必須)<br>`item_type` (document/image/auto) |

**使用例（Claude Desktop上）:**

```
# ドキュメントを追加
./docs/sample.txtをRAGシステムに追加してください

# ドキュメントを検索
「機械学習」に関連するドキュメントを検索してください

# ドキュメントを削除
doc_001を削除してください
```

#### リソース（Resources）

| リソースURI | 説明 | 内容 |
|-----------|------|------|
| `resource://documents/list` | 全ドキュメント一覧 | テキストドキュメントと画像のメタデータ配列（JSON形式） |

**使用例（Claude Desktop上）:**

```
登録されているドキュメントのリソースを確認してください
```

#### MCPサーバーの動作フロー

```
┌─────────────────┐
│ Claude Desktop  │
│   (MCPクライアント) │
└────────┬────────┘
         │ stdio通信
         ▼
┌─────────────────────┐
│  RAG MCP Server     │
│  (src/mcp/server.py)│
├─────────────────────┤
│ ツールハンドラー      │
│ リソースハンドラー    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  RAG Core           │
│  (src/rag/)         │
│  - vector_store     │
│  - embeddings       │
│  - engine           │
└─────────────────────┘
```

**特徴:**
- Claude DesktopとMCPサーバーはstdio（標準入出力）で通信
- MCPサーバーは既存のRAGコア機能を呼び出すラッパー層
- CLIとMCPサーバーは同じRAGコアを共有

### トラブルシューティング

**Q: Claude DesktopでMCPサーバーが認識されない**
```bash
# 1. 設定ファイルのパスが正しいか確認
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json

# 2. uvコマンドが利用可能か確認
which uv

# 3. プロジェクトディレクトリで手動起動を試す
cd /path/to/rag-sample
uv run rag-mcp-server
```

**Q: ツール呼び出しがエラーになる**
- Ollamaが起動しているか確認: `ollama list`
- `.env` ファイルが存在し、正しい設定が含まれているか確認
- `uv run rag status` でシステムステータスを確認

**Q: 画像ドキュメントが追加できない**
- 画像ファイルの拡張子が `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.webp` のいずれかか確認
- ファイルパスが正しいか確認（絶対パスを推奨）

**Q: MCPサーバーのログを確認したい**
MCPサーバーは標準エラー出力にログを出力します。Claude Desktopのログは以下で確認できます：

**macOS:**
```bash
# Claude Desktopのログを確認
tail -f ~/Library/Logs/Claude/mcp*.log
```

---

## 開発者向け情報

### プロジェクト構成

```
rag-sample/
├── src/
│   ├── cli.py                  # CLIエントリーポイント
│   ├── commands/               # コマンド実装
│   │   ├── document.py        # ドキュメント管理
│   │   ├── query.py           # 質問・検索
│   │   └── config.py          # 設定管理
│   ├── rag/                    # RAGコア機能
│   │   ├── vector_store.py    # ChromaDB操作
│   │   ├── embeddings.py      # 埋め込み生成
│   │   ├── document_processor.py  # ドキュメント処理
│   │   └── engine.py          # RAGエンジン
│   ├── models/                 # データモデル
│   │   └── document.py
│   └── utils/                  # ユーティリティ
│       └── config.py
├── tests/
│   ├── unit/                   # ユニットテスト
│   └── integration/            # 統合テスト
├── docs/                       # ドキュメント
├── CLAUDE.md                   # Claude Code用プロジェクト指示
└── pyproject.toml              # プロジェクト設定
```

### 開発環境のセットアップ

```bash
# 開発用依存関係を含めてインストール
uv sync --all-extras

# または個別に
uv add --dev pytest pytest-cov pytest-mock
```

### テストの実行

```bash
# 全テストを実行
uv run pytest

# カバレッジレポート付き
uv run pytest --cov=src --cov-report=html

# ユニットテストのみ
uv run pytest tests/unit/ -v

# 統合テストのみ（Ollamaが必要）
uv run pytest tests/integration/ -v

# 特定のテストファイル
uv run pytest tests/unit/test_engine.py -v
```

**テストの種類:**
- **ユニットテスト** (`tests/unit/`) - 外部依存なし、モックを使用
- **統合テスト** (`tests/integration/`) - 実際のOllamaとChromaDBを使用

### コードスタイル

このプロジェクトでは以下のコーディング規約に従っています:

- **Python 3.13+** の型ヒントを使用 (`dict[str, Any]` など)
- **日本語のコメントとdocstring** - すべてのドキュメントは日本語で記述
- **Google-style docstrings** - パラメータと戻り値を明確に記述
- **依存性注入パターン** - テストしやすい設計

### 新機能の追加

1. **新しいコマンドを追加する場合:**
   - `src/commands/` に実装を追加
   - `src/cli.py` でコマンドを登録
   - `tests/unit/` にテストを追加

2. **RAGコアを変更する場合:**
   - `src/rag/` の該当モジュールを編集
   - ユニットテストを更新
   - 統合テストで動作確認

詳細は [CLAUDE.md](CLAUDE.md) を参照してください。

---

## トラブルシューティング

### よくある問題と解決方法

**Q: `ollama: command not found` エラーが出る**
```bash
# Ollamaをインストール
brew install ollama  # macOS
# または https://ollama.ai/ から手動インストール
```

**Q: モデルのダウンロードが遅い**
```bash
# ダウンロード進捗が確認できます
ollama pull gpt-oss

# 別のターミナルでログを確認
tail -f ~/.ollama/logs/server.log
```

**Q: `uv run rag` で "No module named 'src'" エラー**
```bash
# 依存関係を再インストール
uv sync --reinstall
```

**Q: ドキュメント追加時にメモリエラー**
```bash
# チャンクサイズを小さくする
uv run rag config set chunk_size 500
uv run rag config set chunk_overlap 50
```

**Q: 回答の精度が低い**
- `TOP_K` を増やして参照ドキュメントを増やす
- チャンクサイズを調整（小さすぎると文脈が失われる）
- より高性能なLLMモデルを使用（llama3.1など）

**Q: Ollamaの接続エラー**
```bash
# Ollamaが起動しているか確認
curl http://localhost:11434/api/tags

# 起動していない場合
ollama serve &

# ポートを変更した場合
uv run rag config set ollama_base_url http://localhost:YOUR_PORT
```

### ログの確認

詳細なログを確認するには:
```bash
# 詳細ログを有効化
uv run rag --verbose <command>

# または .env で設定
echo "LOG_LEVEL=DEBUG" >> .env
```

---

## ライセンス

MIT License - 詳細は [LICENSE](LICENSE) を参照してください。
