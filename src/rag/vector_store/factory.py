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
