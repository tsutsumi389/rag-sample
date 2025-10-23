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
