"""ファイル関連ユーティリティ

ファイルタイプの判定など、ファイル操作に関する共通ユーティリティを提供します。
"""

from pathlib import Path

# 画像ファイル拡張子の定義
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


def is_image_file(file_path: str | Path) -> bool:
    """ファイルが画像かどうかを拡張子から判定

    Args:
        file_path: チェックするファイルのパス

    Returns:
        画像ファイルの場合True
    """
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS
