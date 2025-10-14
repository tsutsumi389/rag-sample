# RAG CLI アプリケーション 概要

## プロジェクト概要

Python + Ollama + LangChain + ChromaDBを使用したRAG（Retrieval-Augmented Generation）システムのCLIアプリケーションです。
ローカルのドキュメントをベクトルデータベースに格納し、自然言語での質問に対して関連情報を検索・生成して回答します。

## 技術スタック

- **Python 3.x** - メイン開発言語
- **Ollama** - ローカルLLMの実行環境
- **LangChain** - LLMアプリケーション開発フレームワーク
- **ChromaDB** - ベクトルデータベース

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

#### `config`
- 現在の設定を表示
- モデル名、チャンクサイズ、オーバーラップなど

#### `config set <key> <value>`
- 設定を変更
- 主な設定項目:
  - `model`: 使用するOllamaモデル
  - `chunk_size`: テキストチャンクのサイズ
  - `chunk_overlap`: チャンク間のオーバーラップ
  - `top_k`: 検索時の取得チャンク数

#### `status`
- システムの状態確認
- DB接続状態、モデルの動作状況など

#### `init`
- 初期セットアップ
- DBディレクトリの作成
- 設定ファイルの生成

### 4. 便利機能

#### `export <output_file>`
- ベクトルDBのバックアップ
- 他の環境への移行に使用

#### `import <input_file>`
- ベクトルDBの復元
- バックアップファイルから復元

#### `stats`
- 統計情報の表示
- ドキュメント数、チャンク数、DB容量など

## 使用例

### 基本的な使い方

```bash
# 1. 初期化
rag-cli init

# 2. ドキュメント追加
rag-cli add ./docs
rag-cli add manual.pdf

# 3. 登録確認
rag-cli list

# 4. 質問
rag-cli query "このシステムの使い方は？"

# 5. キーワード検索
rag-cli search "インストール方法"
```

### 対話モード

```bash
rag-cli chat

> このドキュメントには何が書かれていますか？
[回答が表示される]

> もっと詳しく教えてください
[回答が表示される]

> exit
```

### 設定のカスタマイズ

```bash
# 現在の設定を確認
rag-cli config

# チャンクサイズを変更
rag-cli config set chunk_size 500

# 使用モデルを変更
rag-cli config set model llama2

# 検索結果数を変更
rag-cli config set top_k 5
```

### バックアップと復元

```bash
# バックアップ
rag-cli export backup.json

# 復元
rag-cli import backup.json

# 統計確認
rag-cli stats
```

## アーキテクチャ

```
rag-cli/
├── src/
│   ├── cli.py              # CLIエントリーポイント
│   ├── commands/           # 各コマンドの実装
│   │   ├── add.py
│   │   ├── query.py
│   │   ├── chat.py
│   │   └── ...
│   ├── rag/                # RAGコア機能
│   │   ├── embeddings.py   # 埋め込み生成
│   │   ├── retriever.py    # ドキュメント検索
│   │   ├── generator.py    # 回答生成
│   │   └── vectorstore.py  # ベクトルDB操作
│   ├── utils/              # ユーティリティ
│   │   ├── config.py       # 設定管理
│   │   ├── loader.py       # ドキュメントローダー
│   │   └── chunker.py      # チャンク分割
│   └── models/             # データモデル
├── docs/                   # ドキュメント
├── tests/                  # テストコード
├── chroma_db/              # ChromaDBデータ（.gitignore）
├── config.yaml             # 設定ファイル
├── requirements.txt        # 依存パッケージ
└── README.md
```

## ワークフロー

1. **ドキュメント登録フロー**
   ```
   ファイル読み込み → チャンク分割 → 埋め込み生成 → ChromaDBに保存
   ```

2. **質問応答フロー**
   ```
   質問入力 → 質問を埋め込み → 類似チャンク検索 → LLMで回答生成 → 回答表示
   ```

## 設定ファイル例

```yaml
# config.yaml
model:
  name: "llama2"
  temperature: 0.7

chunking:
  chunk_size: 1000
  chunk_overlap: 200

retrieval:
  top_k: 3
  similarity_threshold: 0.7

database:
  path: "./chroma_db"
  collection_name: "documents"
```

## 今後の拡張案

- [ ] Web URLからのドキュメント取得
- [ ] 複数コレクションのサポート
- [ ] 会話履歴の保存・復元
- [ ] より多様なファイル形式のサポート
- [ ] GUI版の開発
- [ ] クラウドベクトルDBへの対応
- [ ] マルチモーダル対応（画像、音声）

## ライセンス

MIT License
