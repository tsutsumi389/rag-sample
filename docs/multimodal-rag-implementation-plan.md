# マルチモーダルRAG実装計画

## 概要

現在のテキストベースRAGシステムを拡張し、画像検索・画像を含む質問応答機能を実装します。
OllamaのビジョンモデルとマルチモーダルLLM（Gemma3）を活用したマルチモーダルRAGシステムを構築します。

## 技術スタック追加

### モデル構成

- **マルチモーダルLLM**: `gemma3` (Ollama)
  - テキストと画像の両方を理解できるLLM
  - 画像を含む質問への回答生成

- **ビジョン埋め込みモデル**: `llava` または `bakllava` (Ollama)
  - 画像から特徴ベクトルを抽出
  - テキストと画像を同じベクトル空間にマッピング

- **既存モデル**: `gpt-oss` (テキストLLM), `nomic-embed-text` (テキスト埋め込み)
  - 既存のテキスト処理パイプラインは維持

### 対応画像形式

- JPEG / JPG
- PNG
- GIF
- BMP
- WebP

## アーキテクチャ設計

### 1. データモデル拡張

#### 新規モデル: `ImageDocument`

```python
@dataclass
class ImageDocument:
    """画像ドキュメント"""
    id: str
    file_path: str
    file_name: str
    image_type: str  # 'jpg', 'png', etc.
    caption: str  # ビジョンモデルが生成した説明文
    metadata: dict[str, Any]
    created_at: datetime

    # 画像データ（Base64エンコード）
    image_data: str | None = None
```

#### 拡張モデル: `Document`

```python
@dataclass
class Document:
    # 既存フィールド
    id: str
    content: str
    metadata: dict[str, Any]

    # 新規フィールド
    document_type: str  # 'text' または 'image'
    image_path: str | None = None  # 画像の場合のパス
```

#### 拡張モデル: `SearchResult`

```python
@dataclass
class SearchResult:
    # 既存フィールド
    chunk: Chunk
    score: float

    # 新規フィールド
    result_type: str  # 'text' または 'image'
    image_path: str | None = None
    caption: str | None = None
```

### 2. モジュール構成

#### 新規モジュール

**`src/rag/vision_embeddings.py`** - ビジョン埋め込み生成
```python
class VisionEmbeddings:
    """画像埋め込み生成クラス（Ollama llava/bakllava使用）"""

    def __init__(self, model_name: str, base_url: str)
    def embed_image(self, image_path: str) -> list[float]
    def embed_images(self, image_paths: list[str]) -> list[list[float]]
    def generate_caption(self, image_path: str) -> str
```

**`src/rag/image_processor.py`** - 画像処理
```python
class ImageProcessor:
    """画像ファイルの処理クラス"""

    def __init__(self, vision_embeddings: VisionEmbeddings)
    def load_image(self, file_path: str) -> ImageDocument
    def load_images_from_directory(self, dir_path: str) -> list[ImageDocument]
    def validate_image(self, file_path: str) -> bool
    def encode_image_base64(self, file_path: str) -> str
```

**`src/rag/multimodal_engine.py`** - マルチモーダルRAGエンジン
```python
class MultimodalRAGEngine:
    """テキストと画像を統合的に扱うRAGエンジン"""

    def __init__(
        self,
        vector_store: VectorStore,
        text_embeddings: Embeddings,
        vision_embeddings: VisionEmbeddings,
        llm_model: str,
        ollama_base_url: str
    )

    def query_with_images(
        self,
        query: str,
        image_paths: list[str] | None = None,
        chat_history: list[ChatMessage] | None = None
    ) -> str

    def search_images(self, query: str, top_k: int = 5) -> list[SearchResult]
    def search_multimodal(self, query: str, top_k: int = 5) -> list[SearchResult]
```

#### 既存モジュール拡張

**`src/rag/vector_store.py`** - ベクトルストア拡張
```python
class VectorStore:
    # 既存メソッド
    def add_documents(...)
    def search(...)

    # 新規メソッド
    def add_images(
        self,
        images: list[ImageDocument],
        embeddings: list[list[float]],
        collection_name: str = "images"
    ) -> list[str]

    def search_images(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        collection_name: str = "images"
    ) -> list[SearchResult]

    def search_multimodal(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> list[SearchResult]

    def get_image_by_id(self, image_id: str) -> ImageDocument | None
    def remove_image(self, image_id: str) -> bool
    def list_images(self) -> list[ImageDocument]
```

### 3. ChromaDBコレクション設計

#### コレクション構成

1. **`documents`** - 既存のテキストドキュメント用コレクション
2. **`images`** - 新規の画像用コレクション
3. **`multimodal`** (オプション) - テキストと画像を統合したコレクション

#### 画像コレクションのメタデータ構造

```python
{
    "id": "img_uuid",
    "file_path": "/path/to/image.jpg",
    "file_name": "image.jpg",
    "image_type": "jpg",
    "caption": "ビジョンモデルが生成した画像説明",
    "created_at": "2025-10-17T10:00:00",
    "source": "local",
    "tags": ["tag1", "tag2"],  # オプション
}
```

### 4. コマンド拡張

#### `src/commands/document.py` 拡張

```bash
# 画像追加コマンド
rag add-image <image_path_or_directory>
  --caption <custom_caption>  # オプション: 手動でキャプション指定
  --tags <tag1,tag2>          # オプション: タグ付け

# 画像一覧コマンド
rag list-images
  --format [table|json]       # 出力フォーマット

# 画像削除コマンド
rag remove-image <image_id>

# 画像クリアコマンド
rag clear-images
```

#### `src/commands/query.py` 拡張

```bash
# マルチモーダルクエリ（テキスト+画像で質問）
rag query-multimodal <question>
  --image <image_path>        # オプション: 質問に画像を添付
  --mode [text|image|both]    # 検索対象（デフォルト: both）

# 画像検索コマンド
rag search-images <query>
  --top-k <number>            # 取得件数（デフォルト: 5）
  --show-image                # 画像プレビュー表示（ターミナル対応）

# マルチモーダルチャット
rag chat-multimodal
  - テキストと画像の両方を扱える対話モード
  - `/image <path>` で画像を添付
  - `/search-images <query>` で画像検索
```

#### `src/commands/config.py` 拡張

```bash
# マルチモーダル設定の表示
rag config
  - OLLAMA_VISION_MODEL を追加表示
  - OLLAMA_MULTIMODAL_LLM_MODEL を追加表示

# ステータス確認
rag status
  - ビジョンモデルの接続状態
  - 画像コレクション情報（画像数、容量等）
```

### 5. 環境変数追加

`.env` に以下を追加:

```bash
# マルチモーダルLLM設定
OLLAMA_MULTIMODAL_LLM_MODEL=gemma3  # テキスト+画像対応LLM

# ビジョンモデル設定
OLLAMA_VISION_MODEL=llava            # 画像埋め込み・キャプション生成

# 画像処理設定
IMAGE_CAPTION_AUTO_GENERATE=true     # 画像追加時に自動でキャプション生成
MAX_IMAGE_SIZE_MB=10                 # 画像ファイルサイズ上限
IMAGE_RESIZE_ENABLED=false           # 画像リサイズ有効化（大容量対策）
IMAGE_RESIZE_MAX_WIDTH=1024          # リサイズ後の最大幅
IMAGE_RESIZE_MAX_HEIGHT=1024         # リサイズ後の最大高さ

# マルチモーダル検索設定
MULTIMODAL_SEARCH_TEXT_WEIGHT=0.5    # テキスト検索の重み
MULTIMODAL_SEARCH_IMAGE_WEIGHT=0.5   # 画像検索の重み
```

## 実装フェーズ

### Phase 1: 基盤実装（画像埋め込み・キャプション生成）

**目標**: 画像をベクトル化してChromaDBに保存できるようにする

#### タスク

1. **データモデル実装**
   - [x] `ImageDocument` モデルの作成
   - [x] `Document` モデルに `document_type` フィールド追加
   - [x] `SearchResult` モデル拡張

2. **ビジョン埋め込み実装**
   - [ ] `src/rag/vision_embeddings.py` 作成
   - [ ] Ollama llava/bakllava との統合
   - [ ] 画像埋め込み生成機能
   - [ ] 画像キャプション生成機能

3. **画像処理実装**
   - [ ] `src/rag/image_processor.py` 作成
   - [ ] 画像ファイル読み込み機能
   - [ ] 画像バリデーション（フォーマット、サイズ確認）
   - [ ] Base64エンコード機能

4. **ベクトルストア拡張**
   - [ ] `add_images()` メソッド実装
   - [ ] `search_images()` メソッド実装
   - [ ] 画像コレクション管理機能

5. **環境設定**
   - [ ] `.env.sample` に新しい環境変数を追加
   - [ ] `src/utils/config.py` で新しい設定を読み込み

6. **ユニットテスト**
   - [ ] `tests/unit/test_vision_embeddings.py`
   - [ ] `tests/unit/test_image_processor.py`
   - [ ] `tests/unit/test_vector_store.py` に画像テスト追加

**完了基準**:
- 画像ファイルを読み込み、埋め込みを生成できる
- ChromaDBの画像コレクションに保存できる
- ユニットテストがすべてパスする

---

### Phase 2: 画像検索機能実装

**目標**: テキストクエリで画像を検索できるようにする

#### タスク

1. **コマンド実装**
   - [ ] `add-image` コマンド実装
   - [ ] `list-images` コマンド実装
   - [ ] `remove-image` コマンド実装
   - [ ] `search-images` コマンド実装

2. **検索機能**
   - [ ] テキストクエリ → 画像埋め込み → 類似画像検索
   - [ ] 検索結果の整形（Rich テーブル表示）
   - [ ] 画像パス、キャプション、スコア表示

3. **統合テスト**
   - [ ] `tests/integration/test_image_search.py`
   - [ ] 画像追加 → 検索 → 取得のエンドツーエンドテスト

**完了基準**:
- `rag add-image ./images` で画像を追加できる
- `rag search-images "犬の写真"` で関連画像を検索できる
- 統合テストがすべてパスする

---

### Phase 3: マルチモーダルRAG実装

**目標**: テキストと画像を統合した質問応答システムを構築

#### タスク

1. **マルチモーダルエンジン実装**
   - [ ] `src/rag/multimodal_engine.py` 作成
   - [ ] Gemma3 (マルチモーダルLLM) との統合
   - [ ] テキストと画像の両方を含むコンテキスト構築
   - [ ] 画像を含むプロンプト生成

2. **マルチモーダル検索**
   - [ ] `search_multimodal()` メソッド実装
   - [ ] テキストと画像の検索結果をマージ
   - [ ] 重み付けスコアリング

3. **質問応答機能**
   - [ ] `query_with_images()` メソッド実装
   - [ ] 画像を含む質問への回答生成
   - [ ] 会話履歴のサポート（画像付き）

4. **コマンド実装**
   - [ ] `query-multimodal` コマンド実装
   - [ ] `chat-multimodal` コマンド実装

5. **統合テスト**
   - [ ] `tests/integration/test_multimodal_rag.py`
   - [ ] 画像を含む質問応答のエンドツーエンドテスト

**完了基準**:
- テキストと画像を組み合わせた質問ができる
- `rag query-multimodal "この画像について説明して" --image ./image.jpg` が動作する
- `rag chat-multimodal` で対話的に画像を扱える

---

### Phase 4: 最適化と追加機能

**目標**: パフォーマンス向上とユーザビリティ改善

#### タスク

1. **パフォーマンス最適化**
   - [ ] 画像埋め込み生成のバッチ処理
   - [ ] キャッシュ機構（同じ画像の再処理を避ける）
   - [ ] 画像リサイズ機能（大容量画像対策）

2. **ユーザビリティ向上**
   - [ ] 画像プレビュー表示（ターミナル対応、iTerm2等）
   - [ ] プログレスバー（大量画像追加時）
   - [ ] 詳細なエラーメッセージ

3. **追加機能**
   - [ ] 画像タグ付け機能
   - [ ] 画像メタデータ編集機能
   - [ ] 画像エクスポート機能（検索結果を別ディレクトリにコピー）
   - [ ] 統計情報（画像数、容量、モデル使用状況等）

4. **ドキュメント更新**
   - [ ] README.md にマルチモーダル機能を追記
   - [ ] CLAUDE.md にマルチモーダル開発ガイドライン追加
   - [ ] 使用例とチュートリアル作成

**完了基準**:
- 大量画像の処理が高速化される
- ユーザーフレンドリーなUI
- 完全なドキュメント整備

---

## 技術的考慮事項

### 1. モデル選定

#### ビジョンモデル: `llava` vs `bakllava`

- **llava**:
  - より汎用的なビジョン埋め込み
  - 多言語対応
  - 推奨: 最初のプロトタイプで使用

- **bakllava**:
  - より高精度な画像理解
  - モデルサイズが大きい
  - 推奨: Phase 4で精度向上が必要な場合に検証

#### マルチモーダルLLM: `gemma3`

- テキストと画像の両方を理解
- Ollamaでサポート済み
- 日本語対応（要検証）

### 2. 埋め込み空間の統合

**課題**: テキスト埋め込み (nomic-embed-text) と画像埋め込み (llava) は異なる空間

**解決策**:
- **オプション1**: 別々のコレクションで管理（実装が簡単）
- **オプション2**: 画像キャプションをテキスト埋め込みに変換（既存パイプライン活用）
- **オプション3**: マルチモーダル埋め込みモデル（CLIP系）の導入（要追加モデル）

**推奨**: Phase 1-2ではオプション1、Phase 3でオプション2を組み合わせ

### 3. 画像データの保存

**課題**: ChromaDBに画像バイナリを直接保存するか、パスのみ保存するか

**解決策**:
- **メタデータのみ**: 画像パス、キャプション、埋め込みベクトルのみ保存（推奨）
- **Base64エンコード**: 小さい画像はBase64でメタデータに含める（オプション）
- **ファイルシステム**: 画像ファイルは元のまま保持、ChromaDBにはパスのみ

**推奨**: メタデータ方式で実装、必要に応じてBase64オプション追加

### 4. LLMへの画像入力

Ollamaのマルチモーダルモデル (gemma3) は以下の形式で画像を受け取る:

```python
import ollama

response = ollama.chat(
    model='gemma3',
    messages=[
        {
            'role': 'user',
            'content': 'この画像について説明してください',
            'images': ['/path/to/image.jpg']  # ファイルパスまたはBase64
        }
    ]
)
```

### 5. エラーハンドリング

- Ollamaのビジョンモデルが未インストールの場合
- 画像ファイルが破損している場合
- サポート外の画像フォーマットの場合
- 画像サイズが大きすぎる場合
- モデルのメモリ不足

## テスト戦略

### ユニットテスト

- `VisionEmbeddings`: モックを使った埋め込み生成テスト
- `ImageProcessor`: サンプル画像での処理テスト
- `VectorStore`: 画像コレクション操作のテスト

### 統合テスト

- Ollama llava/bakllava との実際の統合
- ChromaDBへの画像保存・検索
- エンドツーエンドのマルチモーダルRAGフロー

### テストデータ

- `tests/fixtures/images/` ディレクトリに以下を配置:
  - `sample1.jpg` - テスト用画像1
  - `sample2.png` - テスト用画像2
  - `invalid.txt` - 無効なファイル（エラーハンドリングテスト）

## マイルストーン

| Phase | 期間目安 | 完了条件 |
|-------|---------|---------|
| Phase 1 | 3-5日 | 画像埋め込み・キャプション生成が動作 |
| Phase 2 | 2-3日 | 画像検索コマンドが動作 |
| Phase 3 | 4-6日 | マルチモーダルRAGが完全動作 |
| Phase 4 | 3-5日 | 最適化・ドキュメント完成 |

**合計見積もり**: 12-19日

## リスクと対策

| リスク | 影響度 | 対策 |
|-------|-------|------|
| Ollamaのビジョンモデルが期待通り動作しない | 高 | 早期にプロトタイプで検証、代替モデルの検討 |
| 画像埋め込みとテキスト埋め込みの統合が困難 | 中 | 別コレクション管理でフォールバック |
| Gemma3の日本語対応が不十分 | 中 | 英語翻訳レイヤーの追加、または別LLMの検討 |
| 大量画像処理でメモリ不足 | 中 | バッチ処理、リサイズ機能の実装 |
| ChromaDBの容量制限 | 低 | メタデータのみ保存、外部ストレージ連携 |

## 参考資料

### Ollama関連

- Ollama公式ドキュメント: https://github.com/ollama/ollama
- Ollama Vision Models: https://ollama.com/library?capabilities=vision
- Ollama Python SDK: https://github.com/ollama/ollama-python

### LangChain関連

- LangChain Multimodal RAG: https://python.langchain.com/docs/use_cases/multi_modal/
- LangChain Ollama Integration: https://python.langchain.com/docs/integrations/llms/ollama/

### ChromaDB関連

- ChromaDB Multimodal: https://docs.trychroma.com/guides#multimodal
- ChromaDB Collections: https://docs.trychroma.com/usage-guide#using-collections

## 次のステップ

1. **環境準備**
   ```bash
   # 必要なモデルをプル
   ollama pull gemma3
   ollama pull llava
   # または
   ollama pull bakllava
   ```

2. **Phase 1の開始**
   - データモデル実装から着手
   - ビジョン埋め込みのプロトタイプ作成
   - 早期検証でリスクを軽減

3. **継続的なフィードバック**
   - 各フェーズ完了時に動作確認
   - 必要に応じて計画を調整
