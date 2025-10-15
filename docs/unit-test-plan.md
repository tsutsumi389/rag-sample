# ユニットテスト計画

## 概要

このドキュメントはRAGアプリケーションの包括的なユニットテスト計画を定義します。各モジュールの機能を確実に検証し、高いコードカバレッジを達成することを目標とします。

## テスト戦略

### 基本方針

1. **テスト種別**
   - **ユニットテスト**: 外部依存を持たない単一モジュール・関数のテスト（`@pytest.mark.unit`）
   - **統合テスト**: 外部依存（ChromaDB、Ollama等）を含むテスト（`@pytest.mark.integration`）
   - **モックテスト**: 外部依存をモック化したテスト

2. **テストカバレッジ目標**
   - 全体カバレッジ: 80%以上
   - コアモジュール（models, utils.config）: 90%以上
   - RAGコアモジュール: 80%以上

3. **テストの独立性**
   - 各テストは独立して実行可能
   - テスト間で共有状態を持たない
   - fixtureを活用してセットアップ・クリーンアップを自動化

4. **エラーハンドリングのテスト**
   - 正常系だけでなく異常系のテストを必ず含める
   - カスタム例外が適切にraiseされることを確認

## テスト環境

### 必要なライブラリ

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",           # テストフレームワーク
    "pytest-cov>=4.1.0",       # カバレッジ測定
    "pytest-mock>=3.12.0",     # モッキング機能
    "pytest-asyncio>=0.23.0",  # 非同期テスト対応
]
```

### テストディレクトリ構造

```
tests/
├── __init__.py
├── conftest.py                    # 共通fixture定義
├── unit/                          # ユニットテスト
│   ├── __init__.py
│   ├── test_models.py            # データモデルのテスト
│   ├── test_config.py            # 設定管理のテスト
│   ├── test_embeddings.py        # 埋め込み生成のテスト（モック化）
│   ├── test_document_processor.py # ドキュメント処理のテスト
│   ├── test_vector_store.py      # ベクトルストアのテスト（モック化）
│   └── test_engine.py            # RAGエンジンのテスト（モック化）
├── integration/                   # 統合テスト
│   ├── __init__.py
│   ├── test_full_rag_flow.py     # エンドツーエンドテスト
│   └── test_ollama_integration.py # Ollama統合テスト
└── fixtures/                      # テスト用データファイル
    ├── sample.txt
    ├── sample.md
    └── sample.pdf
```

## モジュール別テスト計画

### 1. データモデル（`src/models/document.py`）

**ファイル**: `tests/unit/test_models.py`

#### テスト項目

##### 1.1 Document クラス
- ✓ 正常なDocumentインスタンスの作成
- ✓ file_pathが自動的にPathオブジェクトに変換されることを確認
- ✓ sizeプロパティが正しい文字数を返すことを確認
- ✓ metadataのデフォルト値が空辞書であることを確認
- ✓ timestampが自動設定されることを確認

##### 1.2 Chunk クラス
- ✓ 正常なChunkインスタンスの作成
- ✓ `__post_init__`でメタデータが正しく追加されることを確認
- ✓ sizeプロパティが正しい文字数を返すことを確認
- ✓ metadataにchunk固有の情報が含まれることを確認

##### 1.3 SearchResult クラス
- ✓ 正常なSearchResultインスタンスの作成
- ✓ scoreが0-1の範囲外の場合にValueErrorがraiseされることを確認
- ✓ 境界値テスト（score=0, score=1）

##### 1.4 ChatMessage クラス
- ✓ 正常なChatMessageインスタンスの作成（user, assistant, system）
- ✓ 無効なroleでValueErrorがraiseされることを確認
- ✓ to_dict()メソッドが正しい辞書を返すことを確認

##### 1.5 ChatHistory クラス
- ✓ 正常なChatHistoryインスタンスの作成
- ✓ add_message()でメッセージが追加されることを確認
- ✓ max_messagesによる履歴制限が機能することを確認
- ✓ to_dicts()で全メッセージが辞書リストに変換されることを確認
- ✓ clear()で履歴がクリアされることを確認
- ✓ `__len__`でメッセージ数が正しく返されることを確認

### 2. 設定管理（`src/utils/config.py`）

**ファイル**: `tests/unit/test_config.py`

#### テスト項目

##### 2.1 Config クラス - 正常系
- ✓ デフォルト値でのConfig作成
- ✓ 環境変数からの設定読み込み
- ✓ カスタム.envファイルからの読み込み
- ✓ to_dict()メソッドが全設定を返すことを確認
- ✓ get_chroma_path()が正しいPathオブジェクトを返すことを確認
- ✓ ensure_chroma_directory()でディレクトリが作成されることを確認

##### 2.2 Config クラス - バリデーション異常系
- ✓ 不正なOLLAMA_BASE_URL（http/https以外）でConfigErrorがraise
- ✓ 空のOLLAMA_LLM_MODELでConfigErrorがraise
- ✓ 空のOLLAMA_EMBEDDING_MODELでConfigErrorがraise
- ✓ CHUNK_SIZEが範囲外（<100, >10000）でConfigErrorがraise
- ✓ CHUNK_OVERLAPが負数でConfigErrorがraise
- ✓ CHUNK_OVERLAP >= CHUNK_SIZEでConfigErrorがraise
- ✓ 不正なLOG_LEVELでConfigErrorがraise
- ✓ CHUNK_SIZEが整数でない場合にConfigErrorがraise

##### 2.3 get_config 関数
- ✓ シングルトンパターンが機能することを確認
- ✓ reload=Trueで設定が再読み込みされることを確認

### 3. ドキュメント処理（`src/rag/document_processor.py`）

**ファイル**: `tests/unit/test_document_processor.py`

#### テスト項目

##### 3.1 DocumentProcessor - ファイル形式サポート
- ✓ is_supported_file()でサポート形式を正しく判定
- ✓ TXTファイルの読み込み
- ✓ MDファイルの読み込み
- ✓ PDFファイルの読み込み（PyPDF2を使用）
- ✓ サポート外ファイルでUnsupportedFileTypeErrorがraise

##### 3.2 DocumentProcessor - エラーハンドリング
- ✓ 存在しないファイルでDocumentProcessorErrorがraise
- ✓ ディレクトリパスでDocumentProcessorErrorがraise
- ✓ 空ファイルでDocumentProcessorErrorがraise
- ✓ 不正なエンコーディングのファイルでエラー処理（UTF-8/Shift_JIS試行）

##### 3.3 DocumentProcessor - テキスト分割
- ✓ split_text()で正しくチャンクに分割
- ✓ chunk_sizeが正しく適用されることを確認
- ✓ chunk_overlapが正しく適用されることを確認
- ✓ 日本語テキストの分割（separators: "。"）
- ✓ 空文字列の分割では空リストが返される
- ✓ 短いテキストは分割されずに1つのチャンクとして返される
- ✓ 改行を含むテキストが正しく分割される

##### 3.4 DocumentProcessor - チャンク作成
- ✓ create_chunks()で正しいChunkリストが作成される
- ✓ chunk_idが正しく生成される（document_id_chunk_XXXX）
- ✓ document_idが省略時に自動生成される
- ✓ start_char/end_charが正しく設定される
- ✓ メタデータがドキュメントから継承される

##### 3.5 DocumentProcessor - 統合メソッド
- ✓ process_document()でDocumentとChunkが一度に取得できる

### 4. 埋め込み生成（`src/rag/embeddings.py`）

**ファイル**: `tests/unit/test_embeddings.py`

#### テスト項目

##### 4.1 EmbeddingGenerator - 初期化
- ✓ デフォルト設定での初期化
- ✓ カスタムmodel_name/base_urlでの初期化
- ✓ Ollama接続失敗時にEmbeddingErrorがraise（モック）

##### 4.2 EmbeddingGenerator - ドキュメント埋め込み
- ✓ embed_documents()で正しいベクトルリストが返される（モック）
- ✓ 空リストでValueErrorがraise
- ✓ 空文字列を含むリストでValueErrorがraise
- ✓ バッチ処理が正しく動作する（モック）

##### 4.3 EmbeddingGenerator - クエリ埋め込み
- ✓ embed_query()で正しいベクトルが返される（モック）
- ✓ 空文字列でValueErrorがraise

##### 4.4 EmbeddingGenerator - 次元数取得
- ✓ get_embedding_dimension()で正しい次元数が返される（モック）

##### 4.5 便利関数
- ✓ create_embedding_generator()で正しくインスタンスが作成される

### 5. ベクトルストア（`src/rag/vector_store.py`）

**ファイル**: `tests/unit/test_vector_store.py`

#### テスト項目

##### 5.1 VectorStore - 初期化
- ✓ VectorStoreインスタンスの作成
- ✓ initialize()でChromaDBクライアントとコレクションが初期化される（モック）
- ✓ 初期化失敗時にVectorStoreErrorがraise（モック）

##### 5.2 VectorStore - ドキュメント追加
- ✅ add_documents()で正しくChunkが追加される（モック）
- ✅ chunksとembeddingsの長さが不一致でVectorStoreErrorがraise
- ✅ 空リストの追加で警告ログが出力される
- ✅ コレクション未初期化でVectorStoreErrorがraise

##### 5.3 VectorStore - 検索
- ✅ search()で正しいSearchResultリストが返される（モック）
- ✅ whereフィルタが正しく適用される（モック）
- ✅ n_resultsパラメータが機能する（モック）
- ✅ 検索結果が空の場合に空リストが返される（モック）
- ✅ スコア計算（距離から類似度への変換）が正しい

##### 5.4 VectorStore - 削除
- ✅ delete()でdocument_id指定による削除（モック）
- ✅ delete()でchunk_ids指定による削除（モック）
- ✅ delete()でwhereフィルタによる削除（モック）
- ✅ 削除条件未指定でVectorStoreErrorがraise
- ✅ 削除件数が正しく返される（モック）

##### 5.5 VectorStore - その他操作
- ✅ list_documents()でドキュメント一覧が取得できる（モック）
- ✅ clear()で全データが削除される（モック）
- ✅ get_document_count()で正しいカウントが返される（モック）
- ✅ get_collection_info()でコレクション情報が取得できる（モック）

##### 5.6 VectorStore - コンテキストマネージャー
- ✅ `with`文で初期化・クローズが自動実行される

### 6. RAGエンジン（`src/rag/engine.py`）

**ファイル**: `tests/unit/test_engine.py`

#### テスト項目

##### 6.1 RAGEngine - 初期化
- ✓ デフォルト設定での初期化（モック）
- ✓ カスタムvector_store/embedding_generatorでの初期化
- ✓ LLM初期化失敗時にRAGEngineErrorがraise（モック）

##### 6.2 RAGEngine - 検索
- ✓ retrieve()で正しいSearchResultリストが返される（モック）
- ✓ 空クエリでRAGEngineErrorがraise
- ✓ n_results/whereパラメータが正しく渡される（モック）

##### 6.3 RAGEngine - 回答生成
- ✓ generate_answer()で正しい回答辞書が返される（モック）
- ✓ 空の質問でRAGEngineErrorがraise
- ✓ コンテキストが空の場合の処理
- ✓ include_sources=Trueで情報源が含まれる
- ✓ プロンプトテンプレートのカスタマイズが機能する

##### 6.4 RAGEngine - 統合クエリ
- ✓ query()で検索と回答生成が一度に実行される（モック）

##### 6.5 RAGEngine - チャット機能
- ✓ chat()でチャット形式の回答が生成される（モック）
- ✓ chat_historyにメッセージが追加される
- ✓ 履歴がプロンプトに含まれる
- ✓ max_chat_historyによる履歴制限が機能する

##### 6.6 RAGEngine - その他機能
- ✓ clear_chat_history()で履歴がクリアされる
- ✓ get_chat_history()で履歴が取得できる
- ✓ get_status()でステータス情報が取得できる
- ✓ initialize()でベクトルストアが初期化される

##### 6.7 便利関数
- ✓ create_rag_engine()で正しくインスタンスが作成される

### 7. 統合テスト（`tests/integration/`）

**ファイル**: `tests/integration/test_full_rag_flow.py`

#### テスト項目

##### 7.1 エンドツーエンドフロー（実際のChromaDB使用）
- ✓ ドキュメント追加 → 検索 → 回答生成の完全フロー
- ✓ 複数ドキュメントの追加と検索
- ✓ ドキュメント削除と再検索

**ファイル**: `tests/integration/test_ollama_integration.py`

##### 7.2 Ollama統合テスト（実際のOllama使用）
- ✓ 埋め込み生成の実行
- ✓ LLMによる回答生成の実行
- ✓ Ollama未起動時のエラーハンドリング

## テストデータ準備

### fixtures/sample.txt
```
これはテスト用のサンプルテキストファイルです。
RAGシステムのテストに使用します。
複数行のテキストを含んでいます。
```

### fixtures/sample.md
```markdown
# サンプルMarkdown

これはテスト用のMarkdownファイルです。

## セクション1
内容1

## セクション2
内容2
```

### fixtures/sample.pdf
- シンプルなPDFファイル（テキスト抽出可能）

## 共通Fixture（conftest.py）

```python
@pytest.fixture
def sample_config():
    """テスト用設定"""
    return Config(env_file=None)

@pytest.fixture
def sample_document():
    """テスト用Documentオブジェクト"""
    ...

@pytest.fixture
def sample_chunks():
    """テスト用Chunkリスト"""
    ...

@pytest.fixture
def mock_embeddings(mocker):
    """モック化された埋め込み生成器"""
    ...

@pytest.fixture
def mock_vector_store(mocker):
    """モック化されたベクトルストア"""
    ...

@pytest.fixture
def temp_chroma_db(tmp_path):
    """一時的なChromaDBディレクトリ"""
    ...
```

## テスト実行コマンド

```bash
# 全テスト実行
uv run pytest

# ユニットテストのみ実行
uv run pytest -m unit

# 統合テストのみ実行
uv run pytest -m integration

# カバレッジ付き実行
uv run pytest --cov=src --cov-report=html

# 特定のファイルのみ実行
uv run pytest tests/unit/test_models.py

# 詳細出力
uv run pytest -v

# 失敗したテストのみ再実行
uv run pytest --lf
```

## 成功基準

1. ✓ 全ユニットテストが通過（グリーン）
2. ✓ コードカバレッジが80%以上
3. ✓ 統合テストが通過（Ollama/ChromaDBが利用可能な環境で）
4. ✓ すべてのエラーハンドリングパスがテストされている
5. ✓ モックが適切に使用され、外部依存なしでユニットテストが実行可能

## 今後の拡張

- パフォーマンステスト（大量ドキュメントの処理速度）
- セキュリティテスト（インジェクション攻撃の防御）
- ストレステスト（同時リクエスト処理）
- CLIコマンドのe2eテスト（click.testing.CliRunner使用）
