"""ファイルユーティリティのテスト"""

import pytest
from pathlib import Path

from src.services.file_utils import is_image_file, IMAGE_EXTENSIONS


class TestIsImageFile:
    """is_image_file関数のテストクラス"""

    def test_is_image_file_with_jpg(self):
        """JPG拡張子のファイルを画像として認識"""
        assert is_image_file("test.jpg") is True
        assert is_image_file("test.JPG") is True
        assert is_image_file("test.jpeg") is True
        assert is_image_file("test.JPEG") is True

    def test_is_image_file_with_png(self):
        """PNG拡張子のファイルを画像として認識"""
        assert is_image_file("test.png") is True
        assert is_image_file("test.PNG") is True

    def test_is_image_file_with_other_formats(self):
        """その他の画像形式を認識"""
        assert is_image_file("test.gif") is True
        assert is_image_file("test.bmp") is True
        assert is_image_file("test.webp") is True
        assert is_image_file("test.tiff") is True
        assert is_image_file("test.tif") is True

    def test_is_image_file_with_path_object(self):
        """Pathオブジェクトでも動作"""
        assert is_image_file(Path("test.jpg")) is True
        assert is_image_file(Path("test.png")) is True

    def test_is_not_image_file(self):
        """画像でないファイルは False を返す"""
        assert is_image_file("test.txt") is False
        assert is_image_file("test.pdf") is False
        assert is_image_file("test.docx") is False
        assert is_image_file("test.py") is False
        assert is_image_file("test.md") is False

    def test_is_image_file_with_full_path(self):
        """フルパスでも拡張子を正しく判定"""
        assert is_image_file("/path/to/file.jpg") is True
        assert is_image_file("/path/to/file.txt") is False
        assert is_image_file("./relative/path/image.png") is True

    def test_is_image_file_case_insensitive(self):
        """大文字小文字を区別しない"""
        assert is_image_file("test.JpG") is True
        assert is_image_file("test.PnG") is True
        assert is_image_file("test.GIF") is True


class TestImageExtensions:
    """IMAGE_EXTENSIONS定数のテスト"""

    def test_image_extensions_contains_common_formats(self):
        """一般的な画像形式がすべて含まれている"""
        expected = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        assert IMAGE_EXTENSIONS == expected

    def test_image_extensions_is_set(self):
        """IMAGE_EXTENSIONSがsetであること（高速検索のため）"""
        assert isinstance(IMAGE_EXTENSIONS, set)
