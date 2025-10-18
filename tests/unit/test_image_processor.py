"""ImageProcessorクラスのユニットテスト

画像ファイルの読み込み、バリデーション、処理機能をテストします。
"""

import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from src.rag.image_processor import (
    ImageProcessor,
    ImageProcessorError,
    UnsupportedImageTypeError,
    create_image_processor
)
from src.models.document import ImageDocument


class TestImageProcessor:
    """ImageProcessorクラスのテスト"""

    def test_init(self, mock_vision_embeddings, multimodal_config):
        """初期化のテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)

        assert processor.vision_embeddings == mock_vision_embeddings
        assert processor.config == multimodal_config
        assert ImageProcessor.SUPPORTED_EXTENSIONS == {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    def test_is_supported_file(self, mock_vision_embeddings, multimodal_config):
        """サポートされているファイル形式のチェックテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)

        # サポートされている形式
        assert processor.is_supported_file(Path("image.jpg")) is True
        assert processor.is_supported_file(Path("image.jpeg")) is True
        assert processor.is_supported_file(Path("image.png")) is True
        assert processor.is_supported_file(Path("image.gif")) is True
        assert processor.is_supported_file(Path("image.bmp")) is True
        assert processor.is_supported_file(Path("image.webp")) is True

        # 大文字でもOK
        assert processor.is_supported_file(Path("image.JPG")) is True
        assert processor.is_supported_file(Path("image.PNG")) is True

        # サポートされていない形式
        assert processor.is_supported_file(Path("document.txt")) is False
        assert processor.is_supported_file(Path("document.pdf")) is False
        assert processor.is_supported_file(Path("image.svg")) is False

    def test_validate_image_success(self, mock_vision_embeddings, multimodal_config, sample_image_files):
        """画像バリデーション成功のテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)

        # サンプル画像は小さいのでバリデーションを通過
        assert processor.validate_image(sample_image_files["sample1"]) is True
        assert processor.validate_image(sample_image_files["sample2"]) is True

    def test_validate_image_file_not_found(self, mock_vision_embeddings, multimodal_config):
        """存在しないファイルのバリデーションテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)

        with pytest.raises(FileNotFoundError):
            processor.validate_image("nonexistent.jpg")

    def test_validate_image_unsupported_format(self, mock_vision_embeddings, multimodal_config, tmp_path):
        """サポートされていない形式のバリデーションテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)

        # テキストファイルを作成
        text_file = tmp_path / "test.txt"
        text_file.write_text("not an image")

        with pytest.raises(UnsupportedImageTypeError) as exc_info:
            processor.validate_image(text_file)

        assert "Unsupported image format" in str(exc_info.value)

    def test_validate_image_directory(self, mock_vision_embeddings, multimodal_config, tmp_path):
        """ディレクトリのバリデーションテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)

        # ディレクトリを作成
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        with pytest.raises(ImageProcessorError) as exc_info:
            processor.validate_image(test_dir)

        assert "directory" in str(exc_info.value).lower()

    def test_encode_image_base64(self, mock_vision_embeddings, multimodal_config, sample_image_files):
        """Base64エンコードのテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)

        encoded = processor.encode_image_base64(sample_image_files["sample1"])

        assert isinstance(encoded, str)
        assert len(encoded) > 0
        # Base64文字列は英数字と+/=のみ
        assert all(c.isalnum() or c in '+/=' for c in encoded)

    def test_load_image_success(self, mock_vision_embeddings, multimodal_config, sample_image_files):
        """画像読み込み成功のテスト"""
        # キャプション生成をモック
        mock_vision_embeddings.generate_caption.return_value = "テスト画像"

        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)
        image_doc = processor.load_image(sample_image_files["sample1"])

        assert isinstance(image_doc, ImageDocument)
        assert image_doc.file_path == sample_image_files["sample1"]
        assert image_doc.file_name == "sample1.jpg"
        assert image_doc.image_type == "jpg"
        assert image_doc.caption == "テスト画像"
        assert isinstance(image_doc.created_at, datetime)
        assert image_doc.image_data is None  # デフォルトではBase64なし

    def test_load_image_with_manual_caption(self, mock_vision_embeddings, multimodal_config, sample_image_files):
        """手動キャプション指定のテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)

        manual_caption = "手動で指定したキャプション"
        image_doc = processor.load_image(
            sample_image_files["sample1"],
            caption=manual_caption
        )

        assert image_doc.caption == manual_caption
        # 手動キャプションの場合、generate_captionは呼ばれない
        mock_vision_embeddings.generate_caption.assert_not_called()

    def test_load_image_with_base64(self, mock_vision_embeddings, multimodal_config, sample_image_files):
        """Base64データ含むテスト"""
        mock_vision_embeddings.generate_caption.return_value = "テスト画像"

        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)
        image_doc = processor.load_image(
            sample_image_files["sample1"],
            include_base64=True
        )

        assert image_doc.image_data is not None
        assert isinstance(image_doc.image_data, str)
        assert len(image_doc.image_data) > 0

    def test_load_image_with_tags(self, mock_vision_embeddings, multimodal_config, sample_image_files):
        """タグ付きの画像読み込みテスト"""
        mock_vision_embeddings.generate_caption.return_value = "テスト画像"

        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)
        image_doc = processor.load_image(
            sample_image_files["sample1"],
            tags=["test", "blue", "sample"]
        )

        assert "tags" in image_doc.metadata
        assert image_doc.metadata["tags"] == ["test", "blue", "sample"]

    def test_load_image_no_auto_caption(self, mock_vision_embeddings, multimodal_config, sample_image_files):
        """自動キャプション無効のテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)

        image_doc = processor.load_image(
            sample_image_files["sample1"],
            auto_caption=False
        )

        assert image_doc.caption.startswith("Image:")
        mock_vision_embeddings.generate_caption.assert_not_called()

    def test_load_images_from_directory(self, mock_vision_embeddings, multimodal_config):
        """ディレクトリから画像一括読み込みテスト"""
        mock_vision_embeddings.generate_caption.return_value = "テスト画像"

        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)
        images = processor.load_images_from_directory("tests/fixtures/images")

        assert len(images) >= 3  # sample1, sample2, sample3
        assert all(isinstance(img, ImageDocument) for img in images)
        assert all(img.image_type in ["jpg", "png"] for img in images)

    def test_load_images_from_directory_not_found(self, mock_vision_embeddings, multimodal_config):
        """存在しないディレクトリのテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)

        with pytest.raises(FileNotFoundError):
            processor.load_images_from_directory("nonexistent_dir")

    def test_load_images_from_directory_is_file(self, mock_vision_embeddings, multimodal_config, sample_image_files):
        """ファイルパスを指定した場合のテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)

        with pytest.raises(ImageProcessorError) as exc_info:
            processor.load_images_from_directory(sample_image_files["sample1"])

        assert "not a directory" in str(exc_info.value).lower()

    def test_repr(self, mock_vision_embeddings, multimodal_config):
        """文字列表現のテスト"""
        processor = ImageProcessor(mock_vision_embeddings, multimodal_config)
        repr_str = repr(processor)

        assert "ImageProcessor" in repr_str
        assert "llava" in repr_str
        assert "10.0MB" in repr_str


class TestCreateImageProcessor:
    """create_image_processor便利関数のテスト"""

    def test_create_default(self, mock_vision_embeddings):
        """デフォルト設定での作成テスト"""
        processor = create_image_processor(mock_vision_embeddings)

        assert isinstance(processor, ImageProcessor)
        assert processor.vision_embeddings == mock_vision_embeddings

    def test_create_with_config(self, mock_vision_embeddings, multimodal_config):
        """カスタム設定での作成テスト"""
        processor = create_image_processor(mock_vision_embeddings, multimodal_config)

        assert processor.config == multimodal_config
