# 複数ベクトルDB対応 実装計画書

**バージョン:** 1.0
**作成日:** 2025-10-17
**対象プロジェクト:** RAG CLI Application

---

## 📋 目次

1. [概要](#概要)
2. [目的と背景](#目的と背景)
3. [対応予定のベクトルDB](#対応予定のベクトルdb)
4. [アーキテクチャ設計](#アーキテクチャ設計)
5. [実装タスク詳細](#実装タスク詳細)
6. [Docker環境構築](#docker環境構築)
7. [設定管理](#設定管理)
8. [マイグレーションガイド](#マイグレーションガイド)
9. [テスト戦略](#テスト戦略)
10. [実装スケジュール](#実装スケジュール)
11. [リスクと対策](#リスクと対策)

---

## 概要

現在ChromaDBのみに依存しているRAGアプリケーションを、複数のベクトルデータベースに対応させるための実装計画です。ストラテジーパターンとファクトリーパターンを採用し、拡張性と保守性を確保します。

### 主要な変更点

- **アーキテクチャ**: 単一実装 → ストラテジーパターン
- **対応DB**: ChromaDB → ChromaDB + Qdrant + Milvus + Weaviate (オプション)
- **インフラ**: ローカル → Docker Compose環境
- **設定**: 固定 → 動的切り替え可能

---

## 目的と背景

### 目的

1. **柔軟性の向上**: ユースケースに応じて最適なベクトルDBを選択可能に
2. **スケーラビリティ**: 開発環境から本番環境まで対応
3. **ベンダーロックイン回避**: 特定のDBに依存しない設計
4. **学習機会の提供**: 各ベクトルDBの特性を理解

### 背景

- **現状**: ChromaDBは軽量で開発には最適だが、本番環境では性能不足の可能性
- **需要**: プロダクション向けの高性能ベクトルDBへの移行ニーズ
- **技術トレンド**: Qdrant、Milvusなどの専用ベクトルDBの台頭

---

## 対応予定のベクトルDB

### 1. ChromaDB (現状維持)

**特徴:**
- Python組み込み型、別サーバー不要
- セットアップが簡単
- 開発・小規模データセットに最適

**用途:**
- ローカル開発環境
- プロトタイピング
- 小規模プロジェクト (< 100万ベクトル)

**インストール:**
```bash
uv sync  # 既存の依存関係に含まれる
```

---

### 2. Qdrant (優先度: 高)

**特徴:**
- Rust製の高性能ベクトルDB
- RESTful API + gRPC対応
- メタデータフィルタリングが強力
- クラウド版も利用可能

**用途:**
- 本番環境 (中規模)
- リアルタイム検索が必要な場合
- APIサービスとしての利用

**インストール:**
```bash
# Python クライアント
uv sync --extra qdrant

# Dockerサービス
docker compose --profile qdrant up -d
```

**接続設定:**
```python
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
```

---

### 3. Milvus (優先度: 中)

**特徴:**
- LF AI & Data Foundationのプロジェクト
- 超大規模データに対応 (数十億ベクトル)
- GPU対応、分散処理
- エンタープライズ向け機能が充実

**用途:**
- 大規模本番環境
- 高スループットが必要な場合
- マルチテナント環境

**インストール:**
```bash
# Python クライアント
uv sync --extra milvus

# Dockerサービス (etcd + MinIO + Milvus)
docker compose --profile milvus up -d
```

**接続設定:**
```python
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=  # オプション
MILVUS_PASSWORD=  # オプション
```

---

### 4. Weaviate (優先度: 低・オプション)

**特徴:**
- GraphQLベースのAPI
- セマンティック検索特化
- モジュールシステムで拡張可能
- ベクトル化モジュール内蔵

**用途:**
- セマンティック検索が主目的
- GraphQL APIが必要な場合
- 複雑なスキーマ管理

**インストール:**
```bash
# Python クライアント
uv sync --extra weaviate

# Dockerサービス
docker compose --profile weaviate up -d
```

**接続設定:**
```python
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=  # オプション
```

---

## 比較表

| 項目 | ChromaDB | Qdrant | Milvus | Weaviate |
|------|----------|--------|--------|----------|
| **セットアップ難易度** | ⭐ 簡単 | ⭐⭐ 普通 | ⭐⭐⭐ やや難 | ⭐⭐ 普通 |
| **パフォーマンス** | 中 | 高 | 最高 | 高 |
| **スケーラビリティ** | 低 (単一マシン) | 高 | 最高 (分散) | 高 |
| **メモリ使用量** | 低 (～1GB) | 中 (2-4GB) | 高 (4GB～) | 中 (2-4GB) |
| **データ規模** | ～100万 | ～数千万 | 数億～数十億 | ～数千万 |
| **クラスタリング** | ✗ | ✓ | ✓ | ✓ |
| **GPU対応** | ✗ | 一部 | ✓ | ✗ |
| **メタデータフィルタ** | ✓ (基本) | ✓ (高度) | ✓ (高度) | ✓ (GraphQL) |
| **API方式** | Python SDK | REST + gRPC | gRPC | REST + GraphQL |
| **ライセンス** | Apache 2.0 | Apache 2.0 | Apache 2.0 | BSD-3 |
| **推奨用途** | 開発・検証 | 本番環境 | 大規模本番 | セマンティック検索 |

---

## アーキテクチャ設計

### 設計原則

1. **Open/Closed Principle**: 拡張に開き、修正に閉じる
2. **Dependency Inversion**: 抽象に依存し、具象に依存しない
3. **Strategy Pattern**: 実行時にアルゴリズム(DB実装)を切り替え可能に
4. **Factory Pattern**: オブジェクト生成ロジックを隠蔽

### ディレクトリ構造

```
src/rag/
├── vector_store/              # 新規ディレクトリ
│   ├── __init__.py           # パッケージ初期化 + 公開API
│   ├── base.py               # BaseVectorStore 抽象基底クラス
│   ├── chroma_store.py       # ChromaDB実装
│   ├── qdrant_store.py       # Qdrant実装
│   ├── milvus_store.py       # Milvus実装
│   ├── weaviate_store.py     # Weaviate実装
│   └── factory.py            # create_vector_store ファクトリー
│
├── embeddings.py             # 既存ファイル (変更なし)
├── document_processor.py     # 既存ファイル (変更なし)
└── engine.py                 # 既存ファイル (インポート変更)
```

### クラス図

```
┌─────────────────────────┐
│   BaseVectorStore      │ (抽象基底クラス)
│   (ABC)                │
├─────────────────────────┤
│ + initialize()         │
│ + add_documents()      │
│ + search()             │
│ + delete()             │
│ + list_documents()     │
│ + clear()              │
│ + get_document_count() │
│ + close()              │
└───────────┬─────────────┘
            │
            │ 継承
    ┌───────┴────────┬──────────┬───────────┐
    │                │          │           │
┌───▼────────┐  ┌───▼──────┐ ┌─▼────────┐ ┌─▼──────────┐
│ ChromaVectorStore│ QdrantVectorStore│ MilvusVectorStore│ WeaviateVectorStore│
└────────────┘  └──────────┘ └──────────┘ └────────────┘

                    ▲
                    │ 生成
              ┌─────┴─────┐
              │  Factory  │
              │           │
              └───────────┘
```

### データフロー

```
User Request
    ↓
CLI Command
    ↓
RAG Engine
    ↓
Factory.create_vector_store(config)  ← 設定からDB種別を判定
    ↓
BaseVectorStore (抽象)
    ↓
具体的な実装 (ChromaDB | Qdrant | Milvus | Weaviate)
    ↓
Vector DB (ローカル or Docker)
```

---

## 実装タスク詳細

### Phase 1: アーキテクチャのリファクタリング

#### Task 1.1: 抽象基底クラスの作成

**ファイル:** `src/rag/vector_store/base.py`

**目的:** すべてのベクトルDB実装が従うべきインターフェースを定義

**実装内容:**

```python
"""ベクトルストア抽象基底クラス

すべてのベクトルデータベース実装が継承すべき抽象基底クラスを定義します。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ...models.document import Chunk, SearchResult
from ...utils.config import Config


class VectorStoreError(Exception):
    """ベクトルストア操作のエラー"""
    pass


class BaseVectorStore(ABC):
    """ベクトルストアの抽象基底クラス

    すべてのベクトルデータベース実装が実装すべきメソッドを定義します。

    Attributes:
        config: アプリケーション設定
        collection_name: コレクション名
    """

    def __init__(self, config: Config, collection_name: str = "documents"):
        """初期化

        Args:
            config: アプリケーション設定
            collection_name: コレクション名
        """
        self.config = config
        self.collection_name = collection_name

    @abstractmethod
    def initialize(self) -> None:
        """ベクトルストアの初期化

        クライアント接続、コレクション作成などを行います。

        Raises:
            VectorStoreError: 初期化に失敗した場合
        """
        pass

    @abstractmethod
    def add_documents(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]]
    ) -> None:
        """ドキュメントチャンクをベクトルストアに追加

        Args:
            chunks: 追加するChunkオブジェクトのリスト
            embeddings: 各チャンクの埋め込みベクトルのリスト

        Raises:
            VectorStoreError: 追加に失敗した場合
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        **kwargs
    ) -> list[SearchResult]:
        """埋め込みベクトルを使用して類似ドキュメントを検索

        Args:
            query_embedding: クエリの埋め込みベクトル
            n_results: 返す結果の最大数
            **kwargs: DB固有のフィルタ条件

        Returns:
            SearchResultオブジェクトのリスト（類似度の高い順）

        Raises:
            VectorStoreError: 検索に失敗した場合
        """
        pass

    @abstractmethod
    def delete(
        self,
        document_id: Optional[str] = None,
        chunk_ids: Optional[list[str]] = None,
        **kwargs
    ) -> int:
        """ドキュメントまたはチャンクを削除

        Args:
            document_id: 削除するドキュメントID
            chunk_ids: 削除する特定のチャンクIDのリスト
            **kwargs: DB固有の削除条件

        Returns:
            削除されたチャンク数

        Raises:
            VectorStoreError: 削除に失敗した場合
        """
        pass

    @abstractmethod
    def list_documents(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """ストア内のドキュメント一覧を取得

        Args:
            limit: 返すドキュメント数の上限

        Returns:
            ドキュメント情報の辞書のリスト

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """コレクション内のすべてのドキュメントを削除

        Raises:
            VectorStoreError: クリアに失敗した場合
        """
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """コレクション内のドキュメントチャンク数を取得

        Returns:
            チャンク数

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """ベクトルストアのクライアント接続を閉じる

        リソース解放が必要な場合に使用します。
        """
        pass

    def __enter__(self):
        """コンテキストマネージャーのエントリ"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.close()
        return False
```

**完了条件:**
- [ ] ファイル作成
- [ ] すべての抽象メソッドが定義されている
- [ ] 日本語docstringが記載されている
- [ ] VectorStoreErrorクラスが定義されている

---

#### Task 1.2: 既存VectorStoreのリファクタリング

**ファイル:** `src/rag/vector_store/chroma_store.py`

**目的:** 既存のChromaDB実装を新しいアーキテクチャに適合させる

**作業内容:**

1. **ファイル移動**
   ```bash
   # 既存ファイルを新しいディレクトリに移動
   mkdir -p src/rag/vector_store
   mv src/rag/vector_store.py src/rag/vector_store/chroma_store.py
   ```

2. **クラス名変更**
   ```python
   # Before
   class VectorStore:

   # After
   class ChromaVectorStore(BaseVectorStore):
   ```

3. **インポート修正**
   ```python
   from .base import BaseVectorStore, VectorStoreError
   from ...models.document import Chunk, SearchResult
   from ...utils.config import Config
   ```

4. **メソッドシグネチャ統一**
   - `search()` メソッドに `**kwargs` を追加
   - `delete()` メソッドに `**kwargs` を追加

**完了条件:**
- [ ] ファイルが正しい場所に配置されている
- [ ] クラス名が`ChromaVectorStore`に変更されている
- [ ] `BaseVectorStore`を継承している
- [ ] 既存の全機能が動作する
- [ ] 既存のテストが通る

---

#### Task 1.3: ファクトリークラスの作成

**ファイル:** `src/rag/vector_store/factory.py`

**目的:** 設定に基づいて適切なベクトルストア実装を生成

**実装内容:**

```python
"""ベクトルストアファクトリー

設定に基づいて適切なベクトルストア実装を生成します。
"""

import logging
from typing import Optional

from .base import BaseVectorStore, VectorStoreError
from ...utils.config import Config

logger = logging.getLogger(__name__)


def create_vector_store(
    config: Config,
    collection_name: str = "documents"
) -> BaseVectorStore:
    """設定に基づいてベクトルストアを生成

    環境変数 VECTOR_DB_TYPE に基づいて適切なベクトルストア実装を返します。

    Args:
        config: アプリケーション設定
        collection_name: コレクション名

    Returns:
        ベクトルストアインスタンス

    Raises:
        VectorStoreError: サポートされていないDB種別が指定された場合

    Examples:
        >>> config = Config()
        >>> vector_store = create_vector_store(config)
        >>> vector_store.initialize()
    """
    vector_db_type = config.vector_db_type.lower()

    logger.info(f"ベクトルストアを作成中: {vector_db_type}")

    if vector_db_type == "chroma":
        from .chroma_store import ChromaVectorStore
        return ChromaVectorStore(config, collection_name)

    elif vector_db_type == "qdrant":
        from .qdrant_store import QdrantVectorStore
        return QdrantVectorStore(config, collection_name)

    elif vector_db_type == "milvus":
        from .milvus_store import MilvusVectorStore
        return MilvusVectorStore(config, collection_name)

    elif vector_db_type == "weaviate":
        from .weaviate_store import WeaviateVectorStore
        return WeaviateVectorStore(config, collection_name)

    else:
        supported_types = ["chroma", "qdrant", "milvus", "weaviate"]
        raise VectorStoreError(
            f"サポートされていないベクトルDB種別: {vector_db_type}\n"
            f"サポート対象: {', '.join(supported_types)}"
        )


def get_supported_db_types() -> list[str]:
    """サポートされているベクトルDB種別のリストを取得

    Returns:
        サポート対象のDB種別リスト
    """
    return ["chroma", "qdrant", "milvus", "weaviate"]


def is_db_available(db_type: str) -> bool:
    """指定されたベクトルDBが利用可能か確認

    必要なクライアントライブラリがインストールされているか確認します。

    Args:
        db_type: DB種別 ("chroma", "qdrant", "milvus", "weaviate")

    Returns:
        利用可能な場合True
    """
    try:
        if db_type == "chroma":
            import chromadb
            return True
        elif db_type == "qdrant":
            import qdrant_client
            return True
        elif db_type == "milvus":
            import pymilvus
            return True
        elif db_type == "weaviate":
            import weaviate
            return True
        else:
            return False
    except ImportError:
        return False
```

**完了条件:**
- [ ] ファイル作成
- [ ] すべてのDB種別に対応したファクトリー関数が実装されている
- [ ] ユーティリティ関数が実装されている
- [ ] エラーハンドリングが適切

---

#### Task 1.4: パッケージ初期化ファイル

**ファイル:** `src/rag/vector_store/__init__.py`

**目的:** パッケージの公開APIを定義

**実装内容:**

```python
"""ベクトルストアパッケージ

複数のベクトルデータベース実装を提供します。
"""

from .base import BaseVectorStore, VectorStoreError
from .factory import create_vector_store, get_supported_db_types, is_db_available

# 公開API
__all__ = [
    # 基底クラスとエラー
    "BaseVectorStore",
    "VectorStoreError",

    # ファクトリー関数
    "create_vector_store",
    "get_supported_db_types",
    "is_db_available",
]

# バージョン情報
__version__ = "2.0.0"
```

**完了条件:**
- [ ] ファイル作成
- [ ] 必要なシンボルがエクスポートされている

---

### Phase 2: 各ベクトルDB実装

#### Task 2.1: Qdrant実装

**ファイル:** `src/rag/vector_store/qdrant_store.py`

**依存関係:**
```bash
uv add qdrant-client
```

**実装内容:**

```python
"""Qdrantベクトルストア実装

Qdrantベクトルデータベースの管理・操作を担当します。
"""

import logging
import uuid
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from .base import BaseVectorStore, VectorStoreError
from ...models.document import Chunk, SearchResult
from ...utils.config import Config

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Qdrantベクトルストアの管理クラス

    QdrantClientを使用してデータの永続化と検索を行います。

    Attributes:
        config: アプリケーション設定
        collection_name: コレクション名
        client: Qdrantクライアント
        vector_size: ベクトルの次元数（初期化時に設定）
    """

    def __init__(self, config: Config, collection_name: str = "documents"):
        """初期化

        Args:
            config: アプリケーション設定
            collection_name: コレクション名
        """
        super().__init__(config, collection_name)
        self.client: Optional[QdrantClient] = None
        self.vector_size: Optional[int] = None

    def initialize(self) -> None:
        """Qdrantクライアントとコレクションの初期化

        Raises:
            VectorStoreError: 初期化に失敗した場合
        """
        try:
            logger.info(
                f"Qdrantに接続中: {self.config.qdrant_host}:{self.config.qdrant_port}"
            )

            # Qdrantクライアントの作成
            self.client = QdrantClient(
                host=self.config.qdrant_host,
                port=self.config.qdrant_port,
                api_key=self.config.qdrant_api_key if self.config.qdrant_api_key else None,
                timeout=30.0,
            )

            # 接続確認
            collections = self.client.get_collections()
            logger.info(f"Qdrantに接続しました (既存コレクション数: {len(collections.collections)})")

            # コレクションが存在しない場合は後で作成
            # (ベクトル次元数は最初のadd_documents時に決定)

        except Exception as e:
            error_msg = f"Qdrantの初期化に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def _ensure_collection(self, vector_size: int) -> None:
        """コレクションが存在しない場合は作成

        Args:
            vector_size: ベクトルの次元数
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        try:
            # コレクションの存在確認
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(
                    f"コレクション '{self.collection_name}' を作成中 "
                    f"(ベクトル次元: {vector_size})..."
                )

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"コレクション '{self.collection_name}' を作成しました")

            self.vector_size = vector_size

        except Exception as e:
            error_msg = f"コレクションの作成に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def add_documents(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]]
    ) -> None:
        """ドキュメントチャンクをベクトルストアに追加

        Args:
            chunks: 追加するChunkオブジェクトのリスト
            embeddings: 各チャンクの埋め込みベクトルのリスト

        Raises:
            VectorStoreError: 追加に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"チャンク数({len(chunks)})と埋め込み数({len(embeddings)})が一致しません"
            )

        if not chunks:
            logger.warning("追加するチャンクがありません")
            return

        try:
            # コレクションの作成（初回のみ）
            vector_size = len(embeddings[0])
            self._ensure_collection(vector_size)

            # Qdrant用のポイントを作成
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                point = PointStruct(
                    id=chunk.chunk_id,
                    vector=embedding,
                    payload={
                        "content": chunk.content,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                        **chunk.metadata,
                    }
                )
                points.append(point)

            logger.info(f"{len(points)}個のポイントを追加中...")

            # バッチでQdrantに追加
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            # ドキュメント数を取得
            count = self.get_document_count()
            logger.info(
                f"{len(points)}個のチャンクを正常に追加しました "
                f"(総ドキュメント数: {count})"
            )

        except Exception as e:
            error_msg = f"ドキュメントの追加に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        **kwargs
    ) -> list[SearchResult]:
        """埋め込みベクトルを使用して類似ドキュメントを検索

        Args:
            query_embedding: クエリの埋め込みベクトル
            n_results: 返す結果の最大数
            **kwargs: 追加のフィルタ条件
                - document_id: ドキュメントIDでフィルタ
                - その他のメタデータフィルタ

        Returns:
            SearchResultオブジェクトのリスト（類似度の高い順）

        Raises:
            VectorStoreError: 検索に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        try:
            logger.debug(f"類似検索を実行中（結果数: {n_results}）...")

            # フィルタの構築
            query_filter = None
            if "document_id" in kwargs:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=kwargs["document_id"])
                        )
                    ]
                )

            # Qdrantで検索
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=n_results,
                query_filter=query_filter,
            )

            # 結果が空の場合
            if not search_results:
                logger.info("検索結果が見つかりませんでした")
                return []

            # SearchResultオブジェクトに変換
            results = []
            for rank, hit in enumerate(search_results, start=1):
                payload = hit.payload

                # Chunkオブジェクトを再構築
                chunk = Chunk(
                    content=payload["content"],
                    chunk_id=str(hit.id),
                    document_id=payload["document_id"],
                    chunk_index=payload["chunk_index"],
                    start_char=payload["start_char"],
                    end_char=payload["end_char"],
                    metadata=payload
                )

                search_result = SearchResult(
                    chunk=chunk,
                    score=hit.score,
                    document_name=payload.get("document_name", "Unknown"),
                    document_source=payload.get("source", "Unknown"),
                    rank=rank,
                    metadata=payload
                )
                results.append(search_result)

            logger.info(f"{len(results)}件の検索結果を取得しました")
            return results

        except Exception as e:
            error_msg = f"検索に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def delete(
        self,
        document_id: Optional[str] = None,
        chunk_ids: Optional[list[str]] = None,
        **kwargs
    ) -> int:
        """ドキュメントまたはチャンクを削除

        Args:
            document_id: 削除するドキュメントID
            chunk_ids: 削除する特定のチャンクIDのリスト
            **kwargs: 追加の削除条件

        Returns:
            削除されたチャンク数

        Raises:
            VectorStoreError: 削除に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        try:
            initial_count = self.get_document_count()

            # document_idによる削除
            if document_id:
                logger.info(f"ドキュメント '{document_id}' を削除中...")

                delete_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )

                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=delete_filter
                )

            # chunk_idsによる削除
            elif chunk_ids:
                logger.info(f"{len(chunk_ids)}個のチャンクを削除中...")
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=chunk_ids
                )

            else:
                raise VectorStoreError(
                    "削除条件が指定されていません（document_idまたはchunk_idsが必要）"
                )

            final_count = self.get_document_count()
            deleted_count = initial_count - final_count

            logger.info(
                f"{deleted_count}個のチャンクを削除しました "
                f"(残りドキュメント数: {final_count})"
            )

            return deleted_count

        except Exception as e:
            error_msg = f"削除に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def list_documents(self, limit: Optional[int] = None) -> list[dict[str, Any]]:
        """ストア内のドキュメント一覧を取得

        Args:
            limit: 返すドキュメント数の上限

        Returns:
            ドキュメント情報の辞書のリスト

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        try:
            count = self.get_document_count()

            if count == 0:
                logger.info("ストアにドキュメントがありません")
                return []

            # スクロールAPIですべてのポイントを取得
            scroll_limit = limit if limit else count
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=scroll_limit,
                with_payload=True,
            )

            # ドキュメントIDごとにグループ化
            documents_map: dict[str, dict[str, Any]] = {}

            for point in points:
                payload = point.payload
                doc_id = payload.get("document_id", "unknown")

                if doc_id not in documents_map:
                    documents_map[doc_id] = {
                        "document_id": doc_id,
                        "document_name": payload.get("document_name", "Unknown"),
                        "source": payload.get("source", "Unknown"),
                        "doc_type": payload.get("doc_type", "Unknown"),
                        "chunk_count": 0,
                        "total_size": 0
                    }

                documents_map[doc_id]["chunk_count"] += 1
                documents_map[doc_id]["total_size"] += payload.get("size", 0)

            documents = list(documents_map.values())
            logger.info(f"{len(documents)}個のドキュメントを取得しました")

            return documents

        except Exception as e:
            error_msg = f"ドキュメント一覧の取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def clear(self) -> None:
        """コレクション内のすべてのドキュメントを削除

        Raises:
            VectorStoreError: クリアに失敗した場合
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        try:
            count = self.get_document_count()

            if count == 0:
                logger.info("コレクションは既に空です")
                return

            logger.warning(
                f"コレクション '{self.collection_name}' の全データを削除中..."
            )

            # コレクションを削除
            self.client.delete_collection(self.collection_name)

            logger.info(f"{count}個のドキュメントを削除しました")

        except Exception as e:
            error_msg = f"コレクションのクリアに失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def get_document_count(self) -> int:
        """コレクション内のドキュメントチャンク数を取得

        Returns:
            チャンク数

        Raises:
            VectorStoreError: 取得に失敗した場合
        """
        if not self.client:
            raise VectorStoreError("クライアントが初期化されていません")

        try:
            # コレクションが存在しない場合は0を返す
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                return 0

            collection_info = self.client.get_collection(self.collection_name)
            return collection_info.points_count

        except Exception as e:
            error_msg = f"ドキュメント数の取得に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise VectorStoreError(error_msg) from e

    def close(self) -> None:
        """Qdrantクライアントを閉じる"""
        logger.info("Qdrantクライアントをクローズしています...")
        if self.client:
            self.client.close()
        self.client = None
```

**Docker Compose設定:**

**ファイル:** `docker/qdrant/docker-compose.yml`

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: rag-qdrant
    ports:
      - "6333:6333"  # HTTP API
      - "6334:6334"  # gRPC API
    volumes:
      - ./data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5
```

**完了条件:**
- [ ] `qdrant_store.py` 実装完了
- [ ] Docker Compose設定ファイル作成
- [ ] 依存関係追加 (`qdrant-client`)
- [ ] 基本的なCRUD操作が動作する
- [ ] エラーハンドリングが適切

---

#### Task 2.2: Milvus実装

**ファイル:** `src/rag/vector_store/milvus_store.py`

**依存関係:**
```bash
uv add pymilvus
```

**実装の要点:**
- コレクションスキーマの定義
- インデックス作成 (IVF_FLAT または HNSW)
- 検索パラメータの設定
- メタデータフィルタリング

**Docker Compose設定:**

**ファイル:** `docker/milvus/docker-compose.yml`

```yaml
version: '3.8'

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    container_name: rag-milvus-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - ./data/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    healthcheck:
      test: ["CMD", "etcdctl", "endpoint", "health"]
      interval: 30s
      timeout: 20s
      retries: 3

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    container_name: rag-milvus-minio
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9001:9001"
      - "9000:9000"
    volumes:
      - ./data/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 20s
      retries: 3

  milvus:
    image: milvusdb/milvus:v2.3.3
    container_name: rag-milvus
    depends_on:
      - etcd
      - minio
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - ./data/milvus:/var/lib/milvus
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    command: ["milvus", "run", "standalone"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9091/healthz"]
      interval: 30s
      start_period: 90s
      timeout: 20s
      retries: 3
```

**完了条件:**
- [ ] `milvus_store.py` 実装完了
- [ ] Docker Compose設定ファイル作成（etcd + MinIO + Milvus）
- [ ] 依存関係追加 (`pymilvus`)
- [ ] インデックス作成が正常に動作
- [ ] 検索が高速に実行される

---

#### Task 2.3: Weaviate実装 (オプション)

**ファイル:** `src/rag/vector_store/weaviate_store.py`

**依存関係:**
```bash
uv add weaviate-client
```

**Docker Compose設定:**

**ファイル:** `docker/weaviate/docker-compose.yml`

```yaml
version: '3.8'

services:
  weaviate:
    image: semitechnologies/weaviate:1.23.1
    container_name: rag-weaviate
    ports:
      - "8080:8080"
    volumes:
      - ./data:/var/lib/weaviate
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      CLUSTER_HOSTNAME: 'node1'
    restart: unless-stopped
```

**完了条件:**
- [ ] `weaviate_store.py` 実装完了
- [ ] Docker Compose設定ファイル作成
- [ ] スキーマ定義が適切
- [ ] GraphQL API経由での操作が可能

---

### Phase 3: 設定管理の拡張

#### Task 3.1: Config クラスの拡張

**ファイル:** `src/utils/config.py`

**追加する設定項目:**

```python
class Config:
    """アプリケーション設定クラス"""

    # ... 既存のデフォルト値 ...

    # ベクトルDB選択
    DEFAULT_VECTOR_DB_TYPE = "chroma"

    # Qdrant設定
    DEFAULT_QDRANT_HOST = "localhost"
    DEFAULT_QDRANT_PORT = 6333
    DEFAULT_QDRANT_GRPC_PORT = 6334
    DEFAULT_QDRANT_API_KEY = None

    # Milvus設定
    DEFAULT_MILVUS_HOST = "localhost"
    DEFAULT_MILVUS_PORT = 19530
    DEFAULT_MILVUS_USER = None
    DEFAULT_MILVUS_PASSWORD = None

    # Weaviate設定
    DEFAULT_WEAVIATE_URL = "http://localhost:8080"
    DEFAULT_WEAVIATE_API_KEY = None

    def _load_and_validate(self):
        """環境変数から設定値を読み込み、バリデーションを実行"""

        # ... 既存の設定読み込み ...

        # ベクトルDB種別
        self.vector_db_type = os.getenv(
            "VECTOR_DB_TYPE",
            self.DEFAULT_VECTOR_DB_TYPE
        ).lower()

        # Qdrant設定
        self.qdrant_host = os.getenv("QDRANT_HOST", self.DEFAULT_QDRANT_HOST)
        self.qdrant_port = int(os.getenv("QDRANT_PORT", self.DEFAULT_QDRANT_PORT))
        self.qdrant_grpc_port = int(os.getenv(
            "QDRANT_GRPC_PORT",
            self.DEFAULT_QDRANT_GRPC_PORT
        ))
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")

        # Milvus設定
        self.milvus_host = os.getenv("MILVUS_HOST", self.DEFAULT_MILVUS_HOST)
        self.milvus_port = int(os.getenv("MILVUS_PORT", self.DEFAULT_MILVUS_PORT))
        self.milvus_user = os.getenv("MILVUS_USER")
        self.milvus_password = os.getenv("MILVUS_PASSWORD")

        # Weaviate設定
        self.weaviate_url = os.getenv("WEAVIATE_URL", self.DEFAULT_WEAVIATE_URL)
        self.weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

        # バリデーション実行
        self._validate()

    def _validate(self):
        """設定値のバリデーション"""

        # ... 既存のバリデーション ...

        # ベクトルDB種別のバリデーション
        valid_db_types = ["chroma", "qdrant", "milvus", "weaviate"]
        if self.vector_db_type not in valid_db_types:
            raise ConfigError(
                f"VECTOR_DB_TYPE must be one of {valid_db_types}, "
                f"got: {self.vector_db_type}"
            )

    def to_dict(self) -> dict:
        """設定値を辞書形式で取得"""
        base_dict = {
            # ... 既存の設定 ...

            # ベクトルDB設定
            "vector_db_type": self.vector_db_type,

            # Qdrant設定
            "qdrant_host": self.qdrant_host,
            "qdrant_port": self.qdrant_port,
            "qdrant_grpc_port": self.qdrant_grpc_port,
            "qdrant_api_key": "***" if self.qdrant_api_key else None,

            # Milvus設定
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
            "milvus_user": self.milvus_user,
            "milvus_password": "***" if self.milvus_password else None,

            # Weaviate設定
            "weaviate_url": self.weaviate_url,
            "weaviate_api_key": "***" if self.weaviate_api_key else None,
        }
        return base_dict
```

**完了条件:**
- [ ] 新しい設定項目が追加されている
- [ ] バリデーションロジックが実装されている
- [ ] `to_dict()` が更新されている
- [ ] 機密情報（API Key等）がマスクされている

---

#### Task 3.2: .env.sample の更新

**ファイル:** `.env.sample`

**追加内容:**

```bash
# Vector Database Configuration
# ==============================
# Select which vector database to use
# Options: chroma | qdrant | milvus | weaviate
VECTOR_DB_TYPE=chroma

# ChromaDB Configuration (default)
# =================================
# Directory path for ChromaDB persistent storage
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Qdrant Configuration
# ====================
# Qdrant server connection settings
# Note: Start Qdrant with: docker compose --profile qdrant up -d
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
# Optional: API key for Qdrant Cloud
QDRANT_API_KEY=

# Milvus Configuration
# ====================
# Milvus server connection settings
# Note: Start Milvus with: docker compose --profile milvus up -d
MILVUS_HOST=localhost
MILVUS_PORT=19530
# Optional: Authentication credentials
MILVUS_USER=
MILVUS_PASSWORD=

# Weaviate Configuration
# ======================
# Weaviate server connection settings
# Note: Start Weaviate with: docker compose --profile weaviate up -d
WEAVIATE_URL=http://localhost:8080
# Optional: API key for Weaviate Cloud
WEAVIATE_API_KEY=
```

**完了条件:**
- [ ] すべてのベクトルDB設定が記載されている
- [ ] コメントで使用方法が説明されている
- [ ] Docker起動コマンドが記載されている

---

### Phase 4: 既存コードの修正

#### Task 4.1: RAG Engineの修正

**ファイル:** `src/rag/engine.py`

**変更内容:**

```python
# Before
from .vector_store import VectorStore

class RAGEngine:
    def __init__(
        self,
        config: Config,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        # ...

# After
from .vector_store import create_vector_store, BaseVectorStore

class RAGEngine:
    def __init__(
        self,
        config: Config,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[BaseVectorStore] = None,
    ):
        self.config = config

        # ベクトルストアの初期化（渡されない場合はファクトリーで生成）
        if vector_store is not None:
            self.vector_store = vector_store
        else:
            self.vector_store = create_vector_store(config)

        # ... 残りは既存のまま ...
```

**完了条件:**
- [ ] インポート文が更新されている
- [ ] ファクトリー関数を使用している
- [ ] 型ヒントが`BaseVectorStore`に変更されている
- [ ] 既存の機能が正常に動作する

---

#### Task 4.2: CLIコマンドの修正

**影響ファイル:**
- `src/commands/document.py`
- `src/commands/query.py`
- `src/commands/config.py`

**変更内容:**

```python
# 各コマンドファイルで

# Before
from ..rag.vector_store import VectorStore

# After
from ..rag.vector_store import create_vector_store, BaseVectorStore

# 使用箇所の変更例
def add_documents(file_path: str):
    config = get_config()

    # Before
    # vector_store = VectorStore(config)

    # After
    vector_store = create_vector_store(config)

    # ... 残りは同じ ...
```

**config.pyへの追加機能:**

```python
@click.command()
def list_db_types():
    """サポートされているベクトルDB種別を表示"""
    from ..rag.vector_store import get_supported_db_types, is_db_available

    console = Console()

    table = Table(title="サポートされているベクトルDB")
    table.add_column("DB種別", style="cyan")
    table.add_column("利用可能", style="green")
    table.add_column("説明")

    db_descriptions = {
        "chroma": "軽量・組み込み型（開発向け）",
        "qdrant": "高性能・本番環境向け",
        "milvus": "大規模・エンタープライズ向け",
        "weaviate": "セマンティック検索特化",
    }

    for db_type in get_supported_db_types():
        available = "✓" if is_db_available(db_type) else "✗"
        description = db_descriptions.get(db_type, "")
        table.add_row(db_type, available, description)

    console.print(table)
```

**完了条件:**
- [ ] すべてのコマンドファイルが更新されている
- [ ] `list-db-types` コマンドが追加されている
- [ ] CLIヘルプが更新されている
- [ ] 既存のコマンドが正常に動作する

---

### Phase 5: Docker環境整備

#### Task 5.1: 統合 Docker Compose ファイル

**ファイル:** `docker-compose.yml` (プロジェクトルート)

```yaml
version: '3.8'

# すべてのベクトルDBを統合管理
# プロファイル機能で個別起動が可能

services:
  # ==================
  # Qdrant
  # ==================
  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: rag-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./docker/qdrant/data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__LOG_LEVEL=INFO
    restart: unless-stopped
    profiles: ["qdrant", "all"]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - rag-network

  # ==================
  # Milvus + 依存サービス
  # ==================
  milvus-etcd:
    image: quay.io/coreos/etcd:v3.5.5
    container_name: rag-milvus-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
    volumes:
      - ./docker/milvus/data/etcd:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    profiles: ["milvus", "all"]
    networks:
      - rag-network

  milvus-minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    container_name: rag-milvus-minio
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - ./docker/milvus/data/minio:/minio_data
    command: minio server /minio_data --console-address ":9001"
    profiles: ["milvus", "all"]
    networks:
      - rag-network

  milvus:
    image: milvusdb/milvus:v2.3.3
    container_name: rag-milvus
    depends_on:
      - milvus-etcd
      - milvus-minio
    ports:
      - "19530:19530"
      - "9091:9091"
    volumes:
      - ./docker/milvus/data/milvus:/var/lib/milvus
    environment:
      ETCD_ENDPOINTS: milvus-etcd:2379
      MINIO_ADDRESS: milvus-minio:9000
    command: ["milvus", "run", "standalone"]
    profiles: ["milvus", "all"]
    networks:
      - rag-network

  # ==================
  # Weaviate
  # ==================
  weaviate:
    image: semitechnologies/weaviate:1.23.1
    container_name: rag-weaviate
    ports:
      - "8080:8080"
    volumes:
      - ./docker/weaviate/data:/var/lib/weaviate
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      CLUSTER_HOSTNAME: 'node1'
    restart: unless-stopped
    profiles: ["weaviate", "all"]
    networks:
      - rag-network

networks:
  rag-network:
    driver: bridge

volumes:
  qdrant-data:
  milvus-data:
  weaviate-data:
```

**使用方法:**

```bash
# Qdrantのみ起動
docker compose --profile qdrant up -d

# Milvusのみ起動
docker compose --profile milvus up -d

# Weaviateのみ起動
docker compose --profile weaviate up -d

# すべて起動
docker compose --profile all up -d

# 停止
docker compose --profile qdrant down
docker compose --profile all down

# ログ確認
docker compose logs qdrant -f
```

**完了条件:**
- [ ] ファイル作成
- [ ] すべてのDBサービスが定義されている
- [ ] プロファイル機能が正しく設定されている
- [ ] ヘルスチェックが設定されている
- [ ] 各サービスが正常に起動する

---

#### Task 5.2: Docker管理スクリプト

**ファイル:** `scripts/docker_manager.sh`

```bash
#!/bin/bash
# ベクトルDBのDocker管理スクリプト

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# カラー出力
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ヘルプメッセージ
show_help() {
    cat << EOF
ベクトルDB Docker管理スクリプト

使用方法:
    ./scripts/docker_manager.sh <command> [db_type]

コマンド:
    start <db_type>     指定されたベクトルDBを起動
    stop <db_type>      指定されたベクトルDBを停止
    restart <db_type>   指定されたベクトルDBを再起動
    status              すべてのサービスの状態を表示
    logs <db_type>      指定されたDBのログを表示
    clean <db_type>     指定されたDBのデータをクリア（警告: データ削除）
    clean-all           すべてのDBデータをクリア（警告: データ削除）

DB種別:
    chroma      ChromaDB（Dockerなし）
    qdrant      Qdrant
    milvus      Milvus
    weaviate    Weaviate
    all         すべてのDB

例:
    ./scripts/docker_manager.sh start qdrant
    ./scripts/docker_manager.sh stop all
    ./scripts/docker_manager.sh logs milvus
EOF
}

# DBの起動
start_db() {
    local db_type=$1

    if [ "$db_type" = "chroma" ]; then
        echo -e "${YELLOW}ChromaDBは組み込み型のため、Dockerは不要です${NC}"
        return 0
    fi

    echo -e "${GREEN}$db_type を起動中...${NC}"
    docker compose --profile "$db_type" up -d

    echo -e "${GREEN}$db_type が起動しました${NC}"
}

# DBの停止
stop_db() {
    local db_type=$1

    if [ "$db_type" = "chroma" ]; then
        echo -e "${YELLOW}ChromaDBは組み込み型のため、Dockerは不要です${NC}"
        return 0
    fi

    echo -e "${YELLOW}$db_type を停止中...${NC}"
    docker compose --profile "$db_type" down

    echo -e "${GREEN}$db_type が停止しました${NC}"
}

# DBの再起動
restart_db() {
    local db_type=$1

    stop_db "$db_type"
    sleep 2
    start_db "$db_type"
}

# ステータス表示
show_status() {
    echo -e "${GREEN}=== Docker サービス状態 ===${NC}"
    docker compose ps
}

# ログ表示
show_logs() {
    local db_type=$1

    if [ "$db_type" = "chroma" ]; then
        echo -e "${YELLOW}ChromaDBにはDockerログがありません${NC}"
        return 0
    fi

    echo -e "${GREEN}$db_type のログを表示中...${NC}"

    case $db_type in
        qdrant)
            docker compose logs -f qdrant
            ;;
        milvus)
            docker compose logs -f milvus milvus-etcd milvus-minio
            ;;
        weaviate)
            docker compose logs -f weaviate
            ;;
        all)
            docker compose logs -f
            ;;
        *)
            echo -e "${RED}不明なDB種別: $db_type${NC}"
            return 1
            ;;
    esac
}

# データクリーンアップ
clean_data() {
    local db_type=$1

    read -p "警告: $db_type のすべてのデータを削除します。続行しますか？ (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "キャンセルしました"
        return 0
    fi

    case $db_type in
        chroma)
            echo -e "${YELLOW}ChromaDBデータを削除中...${NC}"
            rm -rf ./chroma_db
            echo -e "${GREEN}削除完了${NC}"
            ;;
        qdrant)
            echo -e "${YELLOW}Qdrantデータを削除中...${NC}"
            docker compose --profile qdrant down -v
            rm -rf ./docker/qdrant/data
            echo -e "${GREEN}削除完了${NC}"
            ;;
        milvus)
            echo -e "${YELLOW}Milvusデータを削除中...${NC}"
            docker compose --profile milvus down -v
            rm -rf ./docker/milvus/data
            echo -e "${GREEN}削除完了${NC}"
            ;;
        weaviate)
            echo -e "${YELLOW}Weaviateデータを削除中...${NC}"
            docker compose --profile weaviate down -v
            rm -rf ./docker/weaviate/data
            echo -e "${GREEN}削除完了${NC}"
            ;;
        all)
            clean_data "chroma"
            clean_data "qdrant"
            clean_data "milvus"
            clean_data "weaviate"
            ;;
        *)
            echo -e "${RED}不明なDB種別: $db_type${NC}"
            return 1
            ;;
    esac
}

# メイン処理
main() {
    if [ $# -lt 1 ]; then
        show_help
        exit 1
    fi

    local command=$1
    local db_type=${2:-}

    case $command in
        start)
            if [ -z "$db_type" ]; then
                echo -e "${RED}DB種別を指定してください${NC}"
                show_help
                exit 1
            fi
            start_db "$db_type"
            ;;
        stop)
            if [ -z "$db_type" ]; then
                echo -e "${RED}DB種別を指定してください${NC}"
                show_help
                exit 1
            fi
            stop_db "$db_type"
            ;;
        restart)
            if [ -z "$db_type" ]; then
                echo -e "${RED}DB種別を指定してください${NC}"
                show_help
                exit 1
            fi
            restart_db "$db_type"
            ;;
        status)
            show_status
            ;;
        logs)
            if [ -z "$db_type" ]; then
                echo -e "${RED}DB種別を指定してください${NC}"
                show_help
                exit 1
            fi
            show_logs "$db_type"
            ;;
        clean)
            if [ -z "$db_type" ]; then
                echo -e "${RED}DB種別を指定してください${NC}"
                show_help
                exit 1
            fi
            clean_data "$db_type"
            ;;
        clean-all)
            clean_data "all"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo -e "${RED}不明なコマンド: $command${NC}"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
```

**使用方法:**

```bash
# 実行権限を付与
chmod +x scripts/docker_manager.sh

# Qdrant起動
./scripts/docker_manager.sh start qdrant

# ステータス確認
./scripts/docker_manager.sh status

# ログ表示
./scripts/docker_manager.sh logs qdrant

# データクリーンアップ
./scripts/docker_manager.sh clean qdrant
```

**完了条件:**
- [ ] スクリプトファイル作成
- [ ] 実行権限が付与されている
- [ ] すべてのコマンドが正常に動作する
- [ ] エラーハンドリングが適切

---

### Phase 6: テスト実装

#### Task 6.1: 統合テストの拡張

**ファイル:** `tests/integration/test_vector_stores.py`

```python
"""ベクトルストア統合テスト

すべてのベクトルDB実装に対して共通のテストを実行します。
"""

import pytest
from src.rag.vector_store import create_vector_store, get_supported_db_types
from src.models.document import Chunk, SearchResult
from src.utils.config import Config


@pytest.fixture(scope="module")
def sample_chunks():
    """テスト用のサンプルチャンク"""
    return [
        Chunk(
            content="これはテストドキュメントの最初のチャンクです。",
            chunk_id="chunk-001",
            document_id="doc-001",
            chunk_index=0,
            start_char=0,
            end_char=50,
            metadata={
                "document_name": "test.txt",
                "source": "/tmp/test.txt",
                "doc_type": "text",
                "size": 50,
            }
        ),
        Chunk(
            content="これはテストドキュメントの2番目のチャンクです。",
            chunk_id="chunk-002",
            document_id="doc-001",
            chunk_index=1,
            start_char=50,
            end_char=100,
            metadata={
                "document_name": "test.txt",
                "source": "/tmp/test.txt",
                "doc_type": "text",
                "size": 50,
            }
        ),
    ]


@pytest.fixture(scope="module")
def sample_embeddings():
    """テスト用のサンプル埋め込みベクトル"""
    import random
    random.seed(42)

    # 384次元のランダムベクトル（nomic-embed-textと同じ次元）
    return [
        [random.random() for _ in range(384)],
        [random.random() for _ in range(384)],
    ]


@pytest.mark.parametrize("db_type", ["chroma", "qdrant", "milvus"])
def test_vector_store_initialization(db_type):
    """ベクトルストアの初期化テスト"""
    config = Config()
    config.vector_db_type = db_type

    vector_store = create_vector_store(config)

    try:
        vector_store.initialize()
        assert vector_store is not None
    except Exception as e:
        pytest.skip(f"{db_type} が利用できません: {str(e)}")
    finally:
        vector_store.close()


@pytest.mark.parametrize("db_type", ["chroma", "qdrant", "milvus"])
def test_add_and_search(db_type, sample_chunks, sample_embeddings):
    """ドキュメント追加と検索のテスト"""
    config = Config()
    config.vector_db_type = db_type

    vector_store = create_vector_store(config, collection_name=f"test_{db_type}")

    try:
        # 初期化
        vector_store.initialize()

        # ドキュメント追加
        vector_store.add_documents(sample_chunks, sample_embeddings)

        # ドキュメント数確認
        count = vector_store.get_document_count()
        assert count == 2

        # 検索実行
        results = vector_store.search(
            query_embedding=sample_embeddings[0],
            n_results=2
        )

        # 結果検証
        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0

    except Exception as e:
        pytest.skip(f"{db_type} が利用できません: {str(e)}")
    finally:
        # クリーンアップ
        try:
            vector_store.clear()
        except:
            pass
        vector_store.close()


@pytest.mark.parametrize("db_type", ["chroma", "qdrant", "milvus"])
def test_delete_operations(db_type, sample_chunks, sample_embeddings):
    """削除操作のテスト"""
    config = Config()
    config.vector_db_type = db_type

    vector_store = create_vector_store(config, collection_name=f"test_delete_{db_type}")

    try:
        # 初期化とデータ追加
        vector_store.initialize()
        vector_store.add_documents(sample_chunks, sample_embeddings)

        initial_count = vector_store.get_document_count()
        assert initial_count == 2

        # 1つ削除
        deleted_count = vector_store.delete(chunk_ids=["chunk-001"])
        assert deleted_count == 1

        # 残り確認
        remaining_count = vector_store.get_document_count()
        assert remaining_count == 1

    except Exception as e:
        pytest.skip(f"{db_type} が利用できません: {str(e)}")
    finally:
        try:
            vector_store.clear()
        except:
            pass
        vector_store.close()


@pytest.mark.parametrize("db_type", ["chroma", "qdrant", "milvus"])
def test_list_documents(db_type, sample_chunks, sample_embeddings):
    """ドキュメント一覧取得のテスト"""
    config = Config()
    config.vector_db_type = db_type

    vector_store = create_vector_store(config, collection_name=f"test_list_{db_type}")

    try:
        # 初期化とデータ追加
        vector_store.initialize()
        vector_store.add_documents(sample_chunks, sample_embeddings)

        # ドキュメント一覧取得
        documents = vector_store.list_documents()

        assert len(documents) == 1  # 1つのドキュメントに2つのチャンク
        assert documents[0]["document_id"] == "doc-001"
        assert documents[0]["chunk_count"] == 2

    except Exception as e:
        pytest.skip(f"{db_type} が利用できません: {str(e)}")
    finally:
        try:
            vector_store.clear()
        except:
            pass
        vector_store.close()
```

**完了条件:**
- [ ] テストファイル作成
- [ ] すべてのDB種別でテストが実行される
- [ ] テストが成功する（利用可能なDBのみ）
- [ ] エラーハンドリングが適切

---

#### Task 6.2: Dockerサービステストフィクスチャ

**ファイル:** `tests/conftest.py` (既存ファイルに追加)

```python
import pytest
import subprocess
import time


@pytest.fixture(scope="session")
def docker_services():
    """Dockerサービスの起動・停止管理

    テスト実行時に必要なDockerサービスを自動で起動・停止します。
    """
    # 起動が必要なサービスのリスト
    services_to_start = []

    # 環境変数からテスト対象のDBを判定
    import os
    test_db_types = os.getenv("TEST_VECTOR_DBS", "chroma").split(",")

    for db_type in test_db_types:
        if db_type in ["qdrant", "milvus", "weaviate"]:
            services_to_start.append(db_type)

    # Dockerサービスの起動
    for service in services_to_start:
        try:
            subprocess.run(
                ["docker", "compose", "--profile", service, "up", "-d"],
                check=True,
                capture_output=True
            )
            print(f"Started {service} service")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start {service}: {e}")

    # サービスの起動待機
    if services_to_start:
        print("Waiting for services to be ready...")
        time.sleep(10)

    yield

    # テスト終了後のクリーンアップ（オプション）
    # 環境変数で制御
    if os.getenv("CLEANUP_DOCKER", "false").lower() == "true":
        for service in services_to_start:
            try:
                subprocess.run(
                    ["docker", "compose", "--profile", service, "down"],
                    check=True,
                    capture_output=True
                )
                print(f"Stopped {service} service")
            except subprocess.CalledProcessError as e:
                print(f"Failed to stop {service}: {e}")


@pytest.fixture
def vector_store_factory(docker_services):
    """ベクトルストアファクトリーフィクスチャ

    テストで簡単にベクトルストアを作成できるようにします。
    """
    from src.rag.vector_store import create_vector_store
    from src.utils.config import Config

    created_stores = []

    def factory(db_type: str, collection_name: str = "test"):
        config = Config()
        config.vector_db_type = db_type
        store = create_vector_store(config, collection_name)
        created_stores.append(store)
        return store

    yield factory

    # クリーンアップ
    for store in created_stores:
        try:
            store.clear()
            store.close()
        except:
            pass
```

**使用方法:**

```bash
# ChromaDBのみテスト
uv run pytest tests/integration/test_vector_stores.py

# QdrantとMilvusをテスト（Dockerサービス自動起動）
TEST_VECTOR_DBS=chroma,qdrant,milvus uv run pytest tests/integration/

# テスト後にDockerサービスを停止
CLEANUP_DOCKER=true TEST_VECTOR_DBS=qdrant uv run pytest tests/integration/
```

**完了条件:**
- [ ] フィクスチャが実装されている
- [ ] Dockerサービスが自動起動・停止する
- [ ] 環境変数で制御可能

---

### Phase 7: ドキュメント整備

#### Task 7.1: READMEの更新

**ファイル:** `README.md`

**追加セクション:**

````markdown
## ベクトルデータベースの選択

このアプリケーションは複数のベクトルデータベースに対応しています:

| ベクトルDB | 特徴 | 推奨用途 | セットアップ難易度 |
|-----------|------|---------|-----------------|
| **ChromaDB** | 軽量・組み込み型 | 開発・プロトタイプ | ⭐ 簡単 |
| **Qdrant** | 高性能・本番向け | 本番環境（中規模） | ⭐⭐ 普通 |
| **Milvus** | 超大規模対応 | 本番環境（大規模） | ⭐⭐⭐ やや難 |
| **Weaviate** | セマンティック検索 | 特殊用途 | ⭐⭐ 普通 |

### ベクトルDBのセットアップ

#### ChromaDB（デフォルト）

別途セットアップ不要。そのまま使用できます。

```bash
# .env ファイル
VECTOR_DB_TYPE=chroma
```

#### Qdrant

```bash
# 1. Dockerサービス起動
docker compose --profile qdrant up -d

# 2. .env ファイル設定
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# 3. Python依存関係インストール
uv sync --extra qdrant
```

#### Milvus

```bash
# 1. Dockerサービス起動（etcd + MinIO + Milvus）
docker compose --profile milvus up -d

# 2. .env ファイル設定
VECTOR_DB_TYPE=milvus
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 3. Python依存関係インストール
uv sync --extra milvus
```

### Docker管理スクリプト

便利なスクリプトでDockerサービスを管理できます:

```bash
# Qdrant起動
./scripts/docker_manager.sh start qdrant

# ステータス確認
./scripts/docker_manager.sh status

# ログ表示
./scripts/docker_manager.sh logs qdrant

# 停止
./scripts/docker_manager.sh stop qdrant

# データクリーンアップ
./scripts/docker_manager.sh clean qdrant
```

### ベクトルDBの切り替え

`.env` ファイルで `VECTOR_DB_TYPE` を変更するだけで切り替え可能:

```bash
# ChromaDB → Qdrant へ切り替え

# 1. データをエクスポート（オプション）
uv run rag-cli export --output backup.json

# 2. .env を編集
VECTOR_DB_TYPE=qdrant

# 3. Qdrant起動
docker compose --profile qdrant up -d

# 4. データをインポート（オプション）
uv run rag-cli import --input backup.json
```
````

**完了条件:**
- [ ] ベクトルDB選択セクションが追加されている
- [ ] セットアップ手順が明確
- [ ] 切り替え方法が記載されている

---

#### Task 7.2: CLAUDE.mdの更新

**ファイル:** `CLAUDE.md`

**更新箇所:**

```markdown
## Tech Stack

- **Python 3.13+** with uv package manager
- **Ollama** - Local LLM execution
- **LangChain** - LLM application framework
- **Vector Databases** (複数対応):
  - **ChromaDB** - 軽量・組み込み型（デフォルト）
  - **Qdrant** - 高性能・本番環境向け
  - **Milvus** - 大規模・エンタープライズ向け
  - **Weaviate** - セマンティック検索特化
- **Click** - CLI framework
- **Rich** - Terminal UI formatting

## Architecture

### ベクトルストアアーキテクチャ

```
src/rag/vector_store/
├── base.py              # BaseVectorStore抽象基底クラス
├── chroma_store.py     # ChromaDB実装
├── qdrant_store.py     # Qdrant実装
├── milvus_store.py     # Milvus実装
├── weaviate_store.py   # Weaviate実装
└── factory.py          # create_vector_store ファクトリー
```

**設計パターン:**
- **ストラテジーパターン**: 実行時にベクトルDB実装を切り替え
- **ファクトリーパターン**: 設定に基づいて適切な実装を生成
- **依存性逆転の原則**: 抽象に依存、具象に依存しない

### ベクトルDBの選択基準

- **ChromaDB**: 開発・検証、小規模データ（～100万ベクトル）
- **Qdrant**: 本番環境、中規模データ（～数千万ベクトル）
- **Milvus**: 大規模本番、エンタープライズ（数億～数十億ベクトル）
- **Weaviate**: GraphQL API、複雑なスキーマ管理が必要な場合

## Development Guidelines

### ベクトルストア実装の追加

新しいベクトルDBを追加する場合:

1. `src/rag/vector_store/your_db_store.py` を作成
2. `BaseVectorStore` を継承
3. すべての抽象メソッドを実装
4. `factory.py` にファクトリーロジックを追加
5. `Config` クラスに設定項目を追加
6. Docker Compose設定を追加（必要に応じて）
7. 統合テストを追加

### テスト実行

```bash
# 特定のベクトルDBのみテスト
TEST_VECTOR_DBS=chroma uv run pytest

# 複数のベクトルDBをテスト
TEST_VECTOR_DBS=chroma,qdrant,milvus uv run pytest

# テスト後にDockerサービスを自動停止
CLEANUP_DOCKER=true TEST_VECTOR_DBS=qdrant uv run pytest
```
```

**完了条件:**
- [ ] アーキテクチャセクションが更新されている
- [ ] 開発ガイドラインが追加されている
- [ ] テスト実行方法が記載されている

---

#### Task 7.3: マイグレーションガイドの作成

**ファイル:** `docs/vector-db-migration-guide.md`

**内容:** （既に本ドキュメントの「マイグレーションガイド」セクションに記載）

**完了条件:**
- [ ] ファイル作成
- [ ] ユーザー向けの移行手順が明確
- [ ] トラブルシューティングが含まれている

---

## Docker環境構築

### ディレクトリ構造

```
docker/
├── qdrant/
│   ├── docker-compose.yml
│   └── data/              # 永続化データ（.gitignore）
├── milvus/
│   ├── docker-compose.yml
│   └── data/              # 永続化データ（.gitignore）
│       ├── etcd/
│       ├── minio/
│       └── milvus/
└── weaviate/
    ├── docker-compose.yml
    └── data/              # 永続化データ（.gitignore）
```

### ポート割り当て

| サービス | ポート | 用途 |
|---------|--------|------|
| Qdrant HTTP | 6333 | REST API |
| Qdrant gRPC | 6334 | gRPC API |
| Milvus | 19530 | gRPC API |
| Milvus Metrics | 9091 | Prometheus metrics |
| MinIO (Milvus) | 9000 | S3互換ストレージ |
| MinIO Console | 9001 | Web UI |
| Weaviate | 8080 | REST + GraphQL API |

### .gitignoreへの追加

```gitignore
# Vector DB data directories
docker/qdrant/data/
docker/milvus/data/
docker/weaviate/data/
chroma_db/
```

---

## 設定管理

### 環境変数一覧

#### ベクトルDB選択

```bash
# ベクトルDB種別の選択
VECTOR_DB_TYPE=chroma  # chroma | qdrant | milvus | weaviate
```

#### ChromaDB設定

```bash
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

#### Qdrant設定

```bash
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_API_KEY=  # Qdrant Cloud使用時のみ
```

#### Milvus設定

```bash
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=  # 認証有効時のみ
MILVUS_PASSWORD=  # 認証有効時のみ
```

#### Weaviate設定

```bash
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=  # Weaviate Cloud使用時のみ
```

### 設定例

**開発環境（ChromaDB）:**
```bash
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

**ステージング環境（Qdrant）:**
```bash
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=staging-qdrant.example.com
QDRANT_PORT=6333
QDRANT_API_KEY=your-api-key
```

**本番環境（Milvus）:**
```bash
VECTOR_DB_TYPE=milvus
MILVUS_HOST=prod-milvus.example.com
MILVUS_PORT=19530
MILVUS_USER=admin
MILVUS_PASSWORD=secure-password
```

---

## マイグレーションガイド

### 既存ChromaDBユーザーの移行

#### ステップ1: データのバックアップ

```bash
# 現在のChromaDBデータをバックアップ
cp -r ./chroma_db ./chroma_db.backup

# （将来実装）データエクスポート機能
uv run rag-cli export --output ./backup.json
```

#### ステップ2: 新しいベクトルDBの準備

```bash
# Qdrantに移行する場合
docker compose --profile qdrant up -d

# Milvusに移行する場合
docker compose --profile milvus up -d
```

#### ステップ3: 設定変更

`.env` ファイルを編集:

```bash
# Before
VECTOR_DB_TYPE=chroma

# After
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

#### ステップ4: データ再インデックス

```bash
# 方法1: 元のドキュメントから再登録
uv run rag-cli add ./docs

# 方法2: （将来実装）バックアップからインポート
uv run rag-cli import --input ./backup.json
```

#### ステップ5: 動作確認

```bash
# ドキュメント数確認
uv run rag-cli status

# 検索テスト
uv run rag-cli query "テストクエリ"
```

#### ステップ6: 旧データの削除（オプション）

```bash
# ChromaDBデータの削除
rm -rf ./chroma_db
```

### 複数環境の並行運用

異なる環境で異なるベクトルDBを使用する場合:

```bash
# 開発環境用 .env.development
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db

# 本番環境用 .env.production
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=prod-qdrant.example.com
QDRANT_PORT=6333
QDRANT_API_KEY=prod-api-key
```

起動時に環境を指定:

```bash
# 開発環境
cp .env.development .env
uv run rag-cli query "テスト"

# 本番環境
cp .env.production .env
uv run rag-cli query "本番クエリ"
```

---

## テスト戦略

### テストレベル

1. **ユニットテスト**: 各ベクトルストア実装の個別機能
2. **統合テスト**: ベクトルストアとRAG Engineの連携
3. **E2Eテスト**: CLIコマンドからベクトルDBまでの全体フロー

### テストマトリックス

| テストケース | ChromaDB | Qdrant | Milvus | Weaviate |
|-------------|----------|--------|--------|----------|
| 初期化 | ✓ | ✓ | ✓ | ✓ |
| ドキュメント追加 | ✓ | ✓ | ✓ | ✓ |
| 類似検索 | ✓ | ✓ | ✓ | ✓ |
| メタデータフィルタ | ✓ | ✓ | ✓ | ✓ |
| 削除 | ✓ | ✓ | ✓ | ✓ |
| 大量データ | - | ✓ | ✓ | - |
| パフォーマンス | - | ✓ | ✓ | - |

### CI/CDでのテスト

`.github/workflows/test.yml` (例):

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      qdrant:
        image: qdrant/qdrant:v1.7.4
        ports:
          - 6333:6333

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install uv
        run: pip install uv

      - name: Install dependencies
        run: uv sync --extra all_vectordbs

      - name: Run tests
        env:
          TEST_VECTOR_DBS: chroma,qdrant
        run: uv run pytest tests/ -v
```

---

## 実装スケジュール

### フェーズ別スケジュール

| Phase | タスク | 担当 | 期間 | 依存関係 | 完了条件 |
|-------|--------|------|------|---------|---------|
| **Phase 1** | リファクタリング | Dev | 2-3日 | - | 既存テスト通過 |
| 1.1 | 抽象基底クラス作成 | Dev | 0.5日 | - | ファイル作成・レビュー |
| 1.2 | ChromaStore移行 | Dev | 1日 | 1.1 | テスト通過 |
| 1.3 | ファクトリー作成 | Dev | 0.5日 | 1.2 | 動作確認 |
| 1.4 | パッケージ初期化 | Dev | 0.5日 | 1.3 | インポート確認 |
| **Phase 2** | DB実装 | Dev | 6-9日 | Phase 1 | 各DB動作確認 |
| 2.1 | Qdrant実装 | Dev | 2-3日 | Phase 1 | CRUD動作 |
| 2.2 | Milvus実装 | Dev | 2-3日 | Phase 1 | CRUD動作 |
| 2.3 | Weaviate実装 | Dev | 2-3日 | Phase 1 | CRUD動作 (オプション) |
| **Phase 3** | 設定管理 | Dev | 1日 | Phase 1 | バリデーション動作 |
| 3.1 | Config拡張 | Dev | 0.5日 | Phase 1 | テスト通過 |
| 3.2 | .env.sample更新 | Dev | 0.5日 | 3.1 | ドキュメント完成 |
| **Phase 4** | 既存コード修正 | Dev | 1-2日 | Phase 2 | 全機能動作 |
| 4.1 | RAG Engine修正 | Dev | 0.5日 | Phase 2 | テスト通過 |
| 4.2 | CLI修正 | Dev | 1日 | 4.1 | コマンド動作 |
| **Phase 5** | Docker環境 | DevOps | 1-2日 | - | サービス起動 |
| 5.1 | Docker Compose | DevOps | 0.5日 | - | 全サービス起動 |
| 5.2 | 管理スクリプト | DevOps | 0.5日 | 5.1 | スクリプト動作 |
| **Phase 6** | テスト | QA | 2-3日 | Phase 4 | カバレッジ>80% |
| 6.1 | 統合テスト | QA | 1.5日 | Phase 4 | テスト通過 |
| 6.2 | Dockerフィクスチャ | QA | 0.5日 | Phase 5 | 自動化動作 |
| **Phase 7** | ドキュメント | Tech Writer | 1-2日 | Phase 6 | レビュー完了 |
| 7.1 | README更新 | Writer | 0.5日 | Phase 6 | レビュー承認 |
| 7.2 | CLAUDE.md更新 | Writer | 0.5日 | Phase 6 | レビュー承認 |
| 7.3 | マイグレーションガイド | Writer | 0.5日 | Phase 6 | ユーザー検証 |

### 推奨実装順序

```
Phase 1 (必須)
  ↓
Phase 3 (設定)
  ↓
Phase 2.1 (Qdrant - 優先度高)
  ↓
Phase 4 (統合)
  ↓
Phase 5 (Docker)
  ↓
Phase 6 (テスト)
  ↓
Phase 2.2 (Milvus - 優先度中)
  ↓
Phase 7 (ドキュメント)
  ↓
Phase 2.3 (Weaviate - オプション)
```

### マイルストーン

- **M1: 基盤完成** (Phase 1完了)
  - 抽象基底クラスとファクトリーパターンの実装
  - 既存ChromaDB実装のリファクタリング完了

- **M2: Qdrant対応** (Phase 2.1完了)
  - Qdrant実装とDocker環境整備
  - 最初の追加ベクトルDB稼働

- **M3: 統合完了** (Phase 4完了)
  - 全コンポーネントの統合
  - エンドツーエンドでの動作確認

- **M4: テスト完了** (Phase 6完了)
  - 全ベクトルDBでの統合テスト通過
  - CI/CD環境構築

- **M5: リリース準備** (Phase 7完了)
  - ドキュメント完成
  - ユーザー向けマイグレーションガイド提供

---

## リスクと対策

### 技術的リスク

#### リスク1: 各ベクトルDBのAPI差異

**影響度:** 高
**発生確率:** 高

**内容:**
- 各ベクトルDBでメタデータフィルタリングの記法が異なる
- 検索結果のスコア計算方法が異なる
- サポートする距離関数が異なる

**対策:**
- 抽象基底クラスで共通インターフェースを厳密に定義
- DB固有の機能は`**kwargs`で拡張可能にする
- 統一的なスコア正規化処理を実装
- 詳細な統合テストで互換性を検証

#### リスク2: Docker環境の複雑化

**影響度:** 中
**発生確率:** 中

**内容:**
- 複数のDockerサービス管理が煩雑
- ポート競合の可能性
- ディスク容量の圧迫

**対策:**
- Docker Composeのprofile機能で個別起動
- ポート番号を明確にドキュメント化
- データボリュームのクリーンアップスクリプト提供
- 最小構成（Qdrantのみ）での動作を優先

#### リスク3: パフォーマンスの劣化

**影響度:** 中
**発生確率:** 低

**内容:**
- 抽象化レイヤーによるオーバーヘッド
- 各DBの性能特性を活かせない可能性

**対策:**
- ベンチマークテストの実施
- DB固有の最適化オプションを`**kwargs`で提供
- パフォーマンスクリティカルな部分の最適化

#### リスク4: テストの複雑化

**影響度:** 中
**発生確率:** 高

**内容:**
- 各ベクトルDBごとにテスト環境が必要
- CI/CD環境でのDocker管理が複雑

**対策:**
- parametrizeでテストコードを共通化
- Dockerサービスの自動起動フィクスチャ
- 環境変数でテスト対象DBを制御
- ChromaDBのみでの最小テストを可能に

---

### プロジェクトリスク

#### リスク5: スコープクリープ

**影響度:** 高
**発生確率:** 中

**内容:**
- 各DBの高度な機能対応で実装範囲が拡大
- 追加のベクトルDBサポート要求

**対策:**
- Phase 1-4を最優先（ChromaDB + Qdrant）
- Milvus, Weaviateはオプション扱い
- MVP（Minimum Viable Product）の明確化

#### リスク6: 後方互換性の維持

**影響度:** 高
**発生確率:** 低

**内容:**
- 既存のChromaDBユーザーへの影響
- 設定ファイルの互換性

**対策:**
- デフォルトは`VECTOR_DB_TYPE=chroma`で維持
- 既存の`.env`ファイルで動作保証
- マイグレーションガイドの提供
- バージョン2.0として明示

---

## 成功基準

### 機能要件

- [ ] 4種類のベクトルDB（ChromaDB, Qdrant, Milvus, Weaviate）に対応
- [ ] `.env`ファイルでDBを切り替え可能
- [ ] すべてのDB実装で共通のCRUD操作が動作
- [ ] 既存のChromaDBユーザーが影響を受けない

### 非機能要件

- [ ] 統合テストカバレッジ > 80%
- [ ] すべてのベクトルDBで検索性能が劣化しない（±10%以内）
- [ ] Docker Composeで各DBが正常に起動
- [ ] ドキュメントが完備され、ユーザーが自力でセットアップ可能

### ドキュメント要件

- [ ] README.mdにベクトルDB選択ガイドが記載
- [ ] CLAUDE.mdに新アーキテクチャが記載
- [ ] マイグレーションガイドが提供されている
- [ ] Docker環境のセットアップ手順が明確

---

## 付録

### A. 依存関係インストールコマンド

```bash
# すべてのベクトルDB対応
uv sync --extra all_vectordbs

# 特定のベクトルDBのみ
uv sync --extra qdrant
uv sync --extra milvus
uv sync --extra weaviate

# 開発用依存関係も含める
uv sync --extra all_vectordbs --group dev
```

### B. トラブルシューティング

#### Qdrant接続エラー

**症状:**
```
VectorStoreError: Qdrantの初期化に失敗しました: Connection refused
```

**確認事項:**
```bash
# Dockerサービスが起動しているか確認
docker ps | grep qdrant

# ポートが開いているか確認
curl http://localhost:6333/health

# ログ確認
docker compose logs qdrant
```

**解決策:**
```bash
# サービス再起動
docker compose --profile qdrant down
docker compose --profile qdrant up -d

# ヘルスチェック待機
sleep 10
```

#### Milvus起動失敗

**症状:**
```
Milvusサービスが起動しない
```

**確認事項:**
```bash
# 依存サービス（etcd, MinIO）の状態確認
docker compose ps

# ディスク容量確認
df -h

# ログ確認
docker compose logs milvus
```

**解決策:**
```bash
# データクリーンアップ
./scripts/docker_manager.sh clean milvus

# サービス再起動
docker compose --profile milvus up -d
```

#### ChromaDBからの移行時にデータが見つからない

**症状:**
```
移行後に検索結果が空
```

**確認事項:**
```bash
# ドキュメント数確認
uv run rag-cli status

# .env設定確認
cat .env | grep VECTOR_DB_TYPE
```

**解決策:**
```bash
# データ再登録
uv run rag-cli add ./docs

# または元のドキュメントから再インデックス
uv run rag-cli clear
uv run rag-cli add ./path/to/original/docs
```

### C. 参考リンク

- **ChromaDB**: https://www.trychroma.com/
- **Qdrant**: https://qdrant.tech/
- **Milvus**: https://milvus.io/
- **Weaviate**: https://weaviate.io/
- **Docker Compose**: https://docs.docker.com/compose/

---

## まとめ

この実装計画に従うことで、RAGアプリケーションを複数のベクトルデータベースに対応させ、開発環境から本番環境まで柔軟に対応できるようになります。

**重要なポイント:**
1. ストラテジーパターンで拡張性を確保
2. Docker Composeで簡単にインフラ構築
3. 既存ユーザーへの影響を最小限に
4. 段階的な実装で確実に進行

実装を開始する際は、Phase 1から順に進め、各Phaseの完了条件を満たしてから次に進むことを推奨します。

---

**ドキュメント履歴:**
- v1.0 (2025-10-17): 初版作成
