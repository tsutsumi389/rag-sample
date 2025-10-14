# RAG CLI アプリケーション

Python + Ollama + LangChain + ChromaDBを使用したRAG（Retrieval-Augmented Generation）システムのCLIアプリケーションです。
ローカルのドキュメントをベクトルデータベースに格納し、自然言語での質問に対して関連情報を検索・生成して回答します。

## 技術スタック

- **言語**: Python 3.13+
- **LLM**: Ollama (llama3.2)
- **埋め込みモデル**: nomic-embed-text
- **ベクトルDB**: ChromaDB
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
- [トラブルシューティング](#トラブルシューティング)

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
ollama pull llama3.2

# 埋め込みモデル（ベクトル化用）
ollama pull nomic-embed-text
```

**モデルの確認:**
```bash
ollama list
# llama3.2 と nomic-embed-text が表示されることを確認
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
# ✓ モデル: llama3.2, nomic-embed-text
```

### セットアップ完了！

これでRAG CLIアプリケーションを使用する準備が整いました。

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
  • LLMモデル: llama3.2
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
✓ LLMモデル (llama3.2): 利用可能
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
OLLAMA_LLM_MODEL=llama3.2
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
