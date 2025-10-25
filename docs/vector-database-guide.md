# ベクトルデータベース選択・設定ガイド

このドキュメントでは、RAGシステムで使用可能なベクトルデータベースの選び方と設定方法について説明します。

## 目次

- [ベクトルDBの比較と選択基準](#ベクトルdbの比較と選択基準)
- [サポートされているベクトルDB](#サポートされているベクトルdb)
- [各ベクトルDBのセットアップ方法](#各ベクトルdbのセットアップ方法)
  - [ChromaDB（デフォルト）](#chromadbデフォルト)
  - [Qdrant](#qdrant)
  - [Milvus](#milvus)
  - [Weaviate](#weaviate)
- [切り替え方法](#切り替え方法)
- [トラブルシューティング](#トラブルシューティング)

---

## ベクトルDBの比較と選択基準

### 各ベクトルDBの特徴比較

| 特徴 | ChromaDB | Qdrant | Milvus | Weaviate |
|------|----------|--------|--------|----------|
| **セットアップ難易度** | 簡単 | 普通 | やや難しい | 普通 |
| **ローカル実行** | ✅ 埋め込み | ✅ Docker | ✅ Docker | ✅ Docker |
| **クラウド対応** | ❌ | ✅ | ✅ | ✅ |
| **スケーラビリティ** | 小～中規模 | 中～大規模 | 大規模 | 中～大規模 |
| **メモリ使用量** | 軽量 | 中程度 | 大きい | 中程度 |
| **検索速度** | 普通 | 高速 | 非常に高速 | 高速 |
| **マルチモーダル対応** | ✅ | ✅ | ✅ | ✅ |
| **フィルタリング機能** | 基本 | 高度 | 非常に高度 | 高度 |
| **ライセンス** | Apache 2.0 | Apache 2.0 | Apache 2.0 | BSD-3 |

### 選択基準とユースケース

#### ChromaDB - 推奨対象
- ✅ **初めて使う方**
- ✅ **ローカル開発・プロトタイピング**
- ✅ **小～中規模データ（～100万ドキュメント）**
- ✅ **セットアップの手間を最小限にしたい**
- ✅ **サーバー不要で動作させたい**

**選ぶべき理由:**
- 別サーバー不要（Pythonプロセスに埋め込み）
- 依存関係が最小限
- 設定ファイルの変更だけで使用可能
- 開発・テストに最適

**注意点:**
- 大規模データには不向き
- 複数サーバーでの分散処理は非サポート
- クラウドデプロイには別途考慮が必要

---

#### Qdrant - 推奨対象
- ✅ **本番環境での運用**
- ✅ **中～大規模データ（100万～1000万ドキュメント）**
- ✅ **高速な検索パフォーマンスが必要**
- ✅ **Dockerでの運用が可能**
- ✅ **クラウド対応が必要な場合がある**

**選ぶべき理由:**
- 高速な検索性能
- REST APIとgRPCの両方をサポート
- メモリ効率が良い
- Qdrant Cloudで簡単にスケールアウト可能
- 高度なフィルタリング機能

**注意点:**
- Dockerまたは別サーバーが必要
- 初期セットアップに多少の手間がかかる

---

#### Milvus - 推奨対象
- ✅ **大規模データ（1000万ドキュメント以上）**
- ✅ **エンタープライズ用途**
- ✅ **最高のパフォーマンスが必要**
- ✅ **複雑な検索クエリとフィルタリング**
- ✅ **分散システムでの運用**

**選ぶべき理由:**
- 大規模データに最適化
- GPU対応で超高速検索
- 水平スケーリング対応
- 高度なインデックス戦略
- エンタープライズサポートあり

**注意点:**
- セットアップが最も複雑
- リソース消費が大きい（メモリ・CPU）
- 小規模データではオーバースペック

---

#### Weaviate - 推奨対象
- ✅ **セマンティック検索に特化したい**
- ✅ **グラフベースの検索も必要**
- ✅ **中～大規模データ**
- ✅ **RESTful APIでの統合**
- ✅ **多様なモジュールとの統合**

**選ぶべき理由:**
- セマンティック検索に強い
- グラフベースのデータ構造
- 多数の統合モジュール（OpenAI、Cohere等）
- GraphQLサポート
- 拡張性が高い

**注意点:**
- 学習曲線がやや急
- 設定項目が多い
- 他のDBに比べてコミュニティが小さい

---

## サポートされているベクトルDB

本システムでは以下のベクトルデータベースをサポートしています：

1. **ChromaDB** (デフォルト) - 埋め込み型、セットアップ不要
2. **Qdrant** - 高性能、クラウド対応
3. **Milvus** - 大規模向け、エンタープライズグレード
4. **Weaviate** - セマンティック検索特化

---

## 各ベクトルDBのセットアップ方法

### ChromaDB（デフォルト）

#### 特徴
- **埋め込み型**: 別サーバー不要、Pythonプロセス内で動作
- **セットアップ不要**: `uv sync`で自動インストール
- **開発向き**: ローカル開発・プロトタイピングに最適

#### 1. 依存関係のインストール

```bash
# ChromaDBは標準でインストール済み
uv sync
```

#### 2. 環境変数の設定

`.env`ファイルを作成または編集：

```bash
# ベクトルDB種別（省略可、デフォルト: chroma）
VECTOR_DB_TYPE=chroma

# ChromaDBデータ保存先ディレクトリ
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

#### 3. 動作確認

```bash
# 設定確認
uv run rag config

# ドキュメント追加テスト
uv run rag add ./docs/sample.txt

# 検索テスト
uv run rag query "テストクエリ"
```

#### データの永続化

ChromaDBのデータは`CHROMA_PERSIST_DIRECTORY`で指定したディレクトリに保存されます：

```bash
# データディレクトリの確認
ls -la ./chroma_db/

# バックアップ
tar -czf chroma_backup_$(date +%Y%m%d).tar.gz ./chroma_db/

# リストア
tar -xzf chroma_backup_20250101.tar.gz
```

---

### Qdrant

#### 特徴
- **高性能**: 高速なベクトル検索
- **柔軟な運用**: ローカル（Docker）とクラウド両対応
- **REST/gRPC**: 複数のAPIプロトコル対応

#### 1. Qdrantサーバーのセットアップ

**方法A: Dockerで起動（推奨）**

```bash
# Qdrantコンテナを起動
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# 起動確認
curl http://localhost:6333/
```

**方法B: Docker Composeで起動**

`docker-compose.yml`を作成：

```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "6333:6333"  # REST API
      - "6334:6334"  # gRPC
    volumes:
      - ./qdrant_storage:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
```

起動：

```bash
docker-compose up -d
```

**方法C: Qdrant Cloud（クラウド）**

1. [Qdrant Cloud](https://cloud.qdrant.io/)でアカウント作成
2. クラスターを作成
3. API KeyとエンドポイントURLを取得

#### 2. Pythonクライアントのインストール

```bash
# Qdrantクライアントをインストール
uv add qdrant-client
```

#### 3. 環境変数の設定

`.env`ファイルに追加：

```bash
# ベクトルDB種別
VECTOR_DB_TYPE=qdrant

# Qdrant接続設定（ローカルDocker）
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_GRPC_PORT=6334

# Qdrant Cloud使用時（オプション）
# QDRANT_HOST=xxx-xxx-xxx.aws.cloud.qdrant.io
# QDRANT_PORT=6333
# QDRANT_API_KEY=your-api-key-here
```

#### 4. 動作確認

```bash
# 設定確認
uv run rag config

# ドキュメント追加
uv run rag add ./docs/sample.txt

# 検索
uv run rag query "検索クエリ"
```

#### Qdrantの管理

```bash
# Web UIにアクセス（ブラウザで開く）
open http://localhost:6333/dashboard

# コレクション一覧の確認
curl http://localhost:6333/collections

# データのバックアップ
docker exec qdrant qdrant-backup create

# コンテナの停止・削除
docker stop qdrant
docker rm qdrant
```

---

### Milvus

#### 特徴
- **大規模対応**: 数億ベクトルに対応
- **高性能**: GPU対応で超高速検索
- **分散システム**: 水平スケーリング対応

#### 1. Milvusサーバーのセットアップ

**方法A: Docker Composeで起動（推奨）**

公式のDocker Compose設定をダウンロード：

```bash
# Milvusのセットアップディレクトリを作成
mkdir milvus && cd milvus

# Docker Compose設定をダウンロード
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 起動
docker-compose up -d

# 起動確認（ヘルスチェック）
docker-compose ps
```

**方法B: Milvus Lite（開発用）**

軽量版のMilvus（単一プロセス）：

```bash
# Milvus Liteをインストール
uv add milvus

# Python内で起動（コード例は後述）
```

#### 2. Pythonクライアントのインストール

```bash
# PyMilvusをインストール
uv add pymilvus
```

#### 3. 環境変数の設定

`.env`ファイルに追加：

```bash
# ベクトルDB種別
VECTOR_DB_TYPE=milvus

# Milvus接続設定
MILVUS_HOST=localhost
MILVUS_PORT=19530

# 認証が必要な場合（オプション）
# MILVUS_USER=your-username
# MILVUS_PASSWORD=your-password
```

#### 4. 動作確認

```bash
# 設定確認
uv run rag config

# ドキュメント追加
uv run rag add ./docs/sample.txt

# 検索
uv run rag query "検索クエリ"
```

#### Milvusの管理

```bash
# Attu（Web UI）を起動
docker run -d \
  --name milvus-attu \
  -p 8000:3000 \
  -e MILVUS_URL=localhost:19530 \
  zilliz/attu:latest

# Web UIにアクセス
open http://localhost:8000

# コンテナの停止
docker-compose down

# データを保持して停止（volumeは削除しない）
docker-compose stop

# データも含めて完全削除
docker-compose down -v
```

---

### Weaviate

#### 特徴
- **セマンティック検索**: 意味ベースの高度な検索
- **GraphQL対応**: 柔軟なクエリ言語
- **モジュール統合**: OpenAI、Cohere等と統合

#### 1. Weaviateサーバーのセットアップ

**Docker Composeで起動**

`docker-compose-weaviate.yml`を作成：

```yaml
version: '3.8'

services:
  weaviate:
    image: semitechnologies/weaviate:latest
    container_name: weaviate
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - ./weaviate_data:/var/lib/weaviate
    restart: unless-stopped
```

起動：

```bash
docker-compose -f docker-compose-weaviate.yml up -d

# 起動確認
curl http://localhost:8080/v1/meta
```

#### 2. Pythonクライアントのインストール

```bash
# Weaviateクライアントをインストール
uv add weaviate-client
```

#### 3. 環境変数の設定

`.env`ファイルに追加：

```bash
# ベクトルDB種別
VECTOR_DB_TYPE=weaviate

# Weaviate接続設定
WEAVIATE_URL=http://localhost:8080

# 認証が必要な場合（オプション）
# WEAVIATE_API_KEY=your-api-key-here
```

#### 4. 動作確認

```bash
# 設定確認
uv run rag config

# ドキュメント追加
uv run rag add ./docs/sample.txt

# 検索
uv run rag query "検索クエリ"
```

#### Weaviateの管理

```bash
# Weaviate Console（Web UI）
open http://localhost:8080/v1/console

# スキーマの確認
curl http://localhost:8080/v1/schema

# コンテナの停止
docker-compose -f docker-compose-weaviate.yml down
```

---

## 切り替え方法

### ベクトルDBの切り替え手順

#### 1. 環境変数の変更

`.env`ファイルの`VECTOR_DB_TYPE`を変更：

```bash
# 現在の設定を確認
grep VECTOR_DB_TYPE .env

# ChromaDBからQdrantに切り替える例
# 変更前
VECTOR_DB_TYPE=chroma

# 変更後
VECTOR_DB_TYPE=qdrant
```

#### 2. 必要な依存関係のインストール

```bash
# Qdrantに切り替える場合
uv add qdrant-client

# Milvusに切り替える場合
uv add pymilvus

# Weaviateに切り替える場合
uv add weaviate-client
```

#### 3. 新しいDBサーバーの起動（必要な場合）

```bash
# ChromaDB以外は別サーバーが必要

# Qdrant
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest

# Milvus
cd milvus && docker-compose up -d

# Weaviate
docker-compose -f docker-compose-weaviate.yml up -d
```

#### 4. データの再インデックス

**重要**: ベクトルDBを切り替えた場合、既存のデータは新しいDBに自動的には移行されません。
ドキュメントを再度追加する必要があります。

```bash
# 既存データの確認
uv run rag list

# 新しいDBにデータを再追加
uv run rag add ./docs/
uv run rag add ./images/

# 確認
uv run rag status
```

#### 5. 動作確認

```bash
# 設定の確認
uv run rag config

# 検索テスト
uv run rag query "テストクエリ"

# チャットテスト
uv run rag chat
```

### データ移行の例

ChromaDBからQdrantへの移行スクリプト例：

```python
#!/usr/bin/env python3
"""
ベクトルDB移行スクリプト
ChromaDB → Qdrant の例
"""

import os
from pathlib import Path
from src.rag.vector_store.factory import create_vector_store
from src.utils.config import Config

def migrate_data():
    """データ移行を実行"""

    # 元のDB（ChromaDB）に接続
    os.environ['VECTOR_DB_TYPE'] = 'chroma'
    config_old = Config()
    old_store = create_vector_store(config_old)
    old_store.initialize()

    # ドキュメント一覧を取得
    documents = old_store.list_documents()
    print(f"移行対象: {len(documents)}件のドキュメント")

    # 新しいDB（Qdrant）に接続
    os.environ['VECTOR_DB_TYPE'] = 'qdrant'
    config_new = Config()
    new_store = create_vector_store(config_new)
    new_store.initialize()

    # データをコピー（実装は各DBのAPIに依存）
    for doc in documents:
        # 元のDBからチャンクと埋め込みを取得
        chunks = old_store.get_document_chunks(doc['id'])
        embeddings = old_store.get_document_embeddings(doc['id'])

        # 新しいDBに追加
        new_store.add_documents(chunks, embeddings)
        print(f"移行完了: {doc['name']}")

    print("データ移行が完了しました")

    old_store.close()
    new_store.close()

if __name__ == '__main__':
    migrate_data()
```

---

## トラブルシューティング

### よくある問題と解決方法

#### 1. "サポートされていないベクトルDB種別" エラー

**エラーメッセージ:**
```
VectorStoreError: サポートされていないベクトルDB種別: xxx
```

**原因:**
- `.env`の`VECTOR_DB_TYPE`の値が間違っている
- サポート対象: `chroma`, `qdrant`, `milvus`, `weaviate`

**解決方法:**
```bash
# .envを確認
cat .env | grep VECTOR_DB_TYPE

# 正しい値に修正
VECTOR_DB_TYPE=chroma  # または qdrant, milvus, weaviate
```

---

#### 2. "クライアントライブラリがインストールされていない" エラー

**エラーメッセージ:**
```
ModuleNotFoundError: No module named 'qdrant_client'
```

**原因:**
選択したベクトルDBのクライアントライブラリが未インストール

**解決方法:**
```bash
# Qdrantの場合
uv add qdrant-client

# Milvusの場合
uv add pymilvus

# Weaviateの場合
uv add weaviate-client

# 再度実行
uv run rag config
```

---

#### 3. "接続できません" エラー

**エラーメッセージ:**
```
ConnectionError: Could not connect to Qdrant at localhost:6333
```

**原因:**
- ベクトルDBサーバーが起動していない
- ポート番号が間違っている
- ファイアウォールでブロックされている

**解決方法:**

```bash
# サーバーが起動しているか確認
docker ps | grep qdrant  # Qdrantの場合

# 起動していない場合は起動
docker start qdrant

# ポート確認
netstat -an | grep 6333

# 接続テスト
curl http://localhost:6333/  # Qdrant
curl http://localhost:8080/v1/meta  # Weaviate
```

---

#### 4. パフォーマンスが遅い

**症状:**
- 検索に時間がかかる
- ドキュメント追加が遅い

**原因と解決方法:**

**ChromaDBの場合:**
```bash
# データ量が多すぎる（100万ドキュメント以上）
# → Qdrantへの移行を検討

# ディスクI/Oが遅い
# → SSDを使用、CHROMA_PERSIST_DIRECTORYを高速なドライブに変更
```

**Qdrant/Milvusの場合:**
```bash
# メモリ不足
# → Dockerのメモリ制限を増やす
docker update --memory="4g" qdrant

# インデックス設定の最適化
# → 各DBのドキュメントを参照してインデックスタイプを調整
```

---

#### 5. データが消えた・見つからない

**原因:**
- ベクトルDBを切り替えた
- データディレクトリを削除した
- Dockerボリュームを削除した

**解決方法:**

```bash
# ChromaDBの場合
ls -la ./chroma_db/  # データディレクトリの確認

# Dockerボリュームの確認
docker volume ls
docker volume inspect qdrant_storage

# バックアップから復元
tar -xzf chroma_backup_20250101.tar.gz

# データを再追加
uv run rag add ./docs/
```

---

#### 6. メモリ不足エラー

**エラーメッセージ:**
```
MemoryError: Unable to allocate array
```

**原因:**
- 大量のドキュメントを一度に処理
- 埋め込みベクトルのサイズが大きい

**解決方法:**

```bash
# バッチサイズを小さくしてドキュメントを追加
# 一度に100ファイルずつ処理
find ./docs -name "*.txt" | head -100 | xargs uv run rag add

# Dockerのメモリ制限を増やす（Milvus/Qdrant）
docker update --memory="8g" --memory-swap="16g" milvus
```

---

## 推奨構成例

### 開発環境

```bash
VECTOR_DB_TYPE=chroma
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

**理由:** セットアップ不要で即座に開発開始可能

---

### ステージング環境

```bash
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

**理由:** 本番に近い環境でテスト可能、Docker Composeで簡単に構築

---

### 本番環境（中規模）

```bash
VECTOR_DB_TYPE=qdrant
QDRANT_HOST=qdrant.example.com
QDRANT_PORT=6333
QDRANT_API_KEY=***************
```

**理由:** クラウド対応、高速、運用が容易

---

### 本番環境（大規模）

```bash
VECTOR_DB_TYPE=milvus
MILVUS_HOST=milvus-cluster.example.com
MILVUS_PORT=19530
MILVUS_USER=admin
MILVUS_PASSWORD=***************
```

**理由:** 数億ベクトルに対応、水平スケーリング可能

---

## まとめ

### 選択のフローチャート

```
開始
 ↓
[初めて使う / 開発環境?] → Yes → ChromaDB
 ↓ No
[データ量は100万件未満?] → Yes → ChromaDB
 ↓ No
[1000万件以上?] → Yes → Milvus
 ↓ No
[クラウド利用?] → Yes → Qdrant Cloud または Milvus Cloud
 ↓ No
[セマンティック検索特化?] → Yes → Weaviate
 ↓ No
 → Qdrant（推奨）
```

### クイックスタート

**最も簡単に始める:**
```bash
# 何も設定せずに開始（ChromaDB使用）
uv sync
uv run rag add ./docs/
uv run rag query "質問"
```

**本番環境に近い環境で始める:**
```bash
# Qdrantをセットアップ
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

# .envを設定
echo "VECTOR_DB_TYPE=qdrant" >> .env

# 依存関係追加
uv add qdrant-client

# 実行
uv run rag add ./docs/
uv run rag query "質問"
```

---

## 参考リンク

- [ChromaDB 公式ドキュメント](https://docs.trychroma.com/)
- [Qdrant 公式ドキュメント](https://qdrant.tech/documentation/)
- [Milvus 公式ドキュメント](https://milvus.io/docs)
- [Weaviate 公式ドキュメント](https://weaviate.io/developers/weaviate)
- [本システムのベクトルDB移行計画](./vector-db-migration-plan.md)
