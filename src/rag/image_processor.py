"""画像処理モジュール

このモジュールは画像ファイルの読み込み、バリデーション、Base64エンコードを行います。
複数の画像形式（JPEG, PNG, GIF, BMP, WebP）に対応し、
ImageDocumentオブジェクトの生成とディレクトリ内の一括読み込みをサポートします。
"""

import base64
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.document import ImageDocument
from src.rag.vision_embeddings import VisionEmbeddings
from src.utils.config import Config, get_config

# ロガー設定
logger = logging.getLogger(__name__)


class ImageProcessorError(Exception):
    """画像処理エラー"""
    pass


class UnsupportedImageTypeError(ImageProcessorError):
    """サポートされていない画像タイプエラー"""
    pass


class ImageProcessor:
    """画像ファイルの処理クラス

    画像ファイルの読み込み、バリデーション、Base64エンコード、
    ImageDocumentオブジェクトの生成を行います。

    Attributes:
        config: アプリケーション設定
        vision_embeddings: ビジョン埋め込み生成器
        supported_extensions: サポートされている画像拡張子
    """

    # サポートされている画像拡張子
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    def __init__(
        self,
        vision_embeddings: VisionEmbeddings,
        config: Optional[Config] = None
    ):
        """画像プロセッサーの初期化

        Args:
            vision_embeddings: ビジョン埋め込み生成器
            config: アプリケーション設定（省略時はデフォルト設定を使用）
        """
        self.config = config or get_config()
        self.vision_embeddings = vision_embeddings
        logger.info("ImageProcessor initialized")

    def is_supported_file(self, file_path: Path) -> bool:
        """ファイルがサポートされている画像形式かチェック

        Args:
            file_path: チェックするファイルパス

        Returns:
            bool: サポートされている場合True
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def validate_image(self, file_path: str | Path) -> bool:
        """画像ファイルのバリデーション

        ファイルの存在、フォーマット、サイズを確認します。

        Args:
            file_path: バリデーションする画像ファイルのパス

        Returns:
            bool: バリデーションに成功した場合True

        Raises:
            ImageProcessorError: バリデーションに失敗した場合
            FileNotFoundError: ファイルが存在しない場合
        """
        path = Path(file_path)

        # ファイルの存在確認
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # ファイルがディレクトリでないか確認
        if path.is_dir():
            raise ImageProcessorError(f"Path is a directory, not a file: {file_path}")

        # サポートされているフォーマットか確認
        if not self.is_supported_file(path):
            raise UnsupportedImageTypeError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        # ファイルサイズの確認
        file_size_mb = path.stat().st_size / (1024 * 1024)
        max_size_mb = self.config.max_image_size_mb
        if file_size_mb > max_size_mb:
            raise ImageProcessorError(
                f"Image file too large: {file_size_mb:.2f}MB "
                f"(max: {max_size_mb}MB). File: {file_path}"
            )

        logger.debug(f"Image validation passed: {path.name} ({file_size_mb:.2f}MB)")
        return True

    def encode_image_base64(self, file_path: str | Path) -> str:
        """画像ファイルをBase64エンコード

        Args:
            file_path: エンコードする画像ファイルのパス

        Returns:
            str: Base64エンコードされた画像データ

        Raises:
            ImageProcessorError: エンコードに失敗した場合
        """
        try:
            path = Path(file_path)
            with open(path, 'rb') as f:
                image_data = f.read()
            encoded = base64.b64encode(image_data).decode('utf-8')
            logger.debug(f"Successfully encoded image to Base64: {path.name}")
            return encoded
        except Exception as e:
            raise ImageProcessorError(
                f"Failed to encode image '{file_path}' to Base64: {e}"
            ) from e

    def _generate_image_id(self, file_path: Path) -> str:
        """画像の一意IDを生成

        ファイルパスとタイムスタンプからハッシュを生成します。

        Args:
            file_path: 画像ファイルのパス

        Returns:
            str: 一意のID
        """
        content = f"{file_path.absolute()}{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def load_image(
        self,
        file_path: str | Path,
        auto_caption: Optional[bool] = None,
        caption: Optional[str] = None,
        include_base64: bool = False,
        tags: Optional[list[str]] = None
    ) -> ImageDocument:
        """画像ファイルを読み込んでImageDocumentオブジェクトを作成

        Args:
            file_path: 読み込む画像ファイルのパス
            auto_caption: キャプションを自動生成するか（Noneの場合は設定値を使用）
            caption: 手動で指定するキャプション（auto_captionより優先）
            include_base64: Base64エンコードした画像データを含めるか
            tags: 画像に付与するタグのリスト

        Returns:
            ImageDocument: 読み込んだ画像ドキュメント

        Raises:
            ImageProcessorError: 画像の読み込みまたは処理に失敗した場合
            FileNotFoundError: ファイルが存在しない場合
            UnsupportedImageTypeError: サポートされていない画像形式の場合

        Example:
            >>> processor = ImageProcessor(vision_embeddings)
            >>> image_doc = processor.load_image("path/to/image.jpg")
            >>> print(image_doc.caption)
        """
        # Pathオブジェクトに変換
        path = Path(file_path)

        # バリデーション
        self.validate_image(path)

        logger.info(f"Loading image: {path.name}")

        try:
            # 画像メタデータの取得
            image_id = self._generate_image_id(path)
            image_type = path.suffix.lstrip('.').lower()
            file_name = path.name
            file_size_mb = path.stat().st_size / (1024 * 1024)

            # キャプションの生成
            if caption:
                # 手動指定のキャプションを使用
                final_caption = caption
                logger.debug(f"Using manual caption for {file_name}")
            else:
                # auto_captionの設定
                should_auto_caption = (
                    auto_caption if auto_caption is not None
                    else self.config.image_caption_auto_generate
                )

                if should_auto_caption:
                    logger.debug(f"Generating caption for {file_name}")
                    final_caption = self.vision_embeddings.generate_caption(path)
                else:
                    final_caption = f"Image: {file_name}"
                    logger.debug(f"Skipping caption generation for {file_name}")

            # Base64エンコード（オプション）
            image_data = None
            if include_base64:
                logger.debug(f"Encoding image to Base64: {file_name}")
                image_data = self.encode_image_base64(path)

            # メタデータの構築
            metadata = {
                'file_size_mb': file_size_mb,
                'absolute_path': str(path.absolute()),
                'tags': tags or [],
            }

            # ImageDocumentの作成
            image_doc = ImageDocument(
                id=image_id,
                file_path=path,
                file_name=file_name,
                image_type=image_type,
                caption=final_caption,
                metadata=metadata,
                created_at=datetime.now(),
                image_data=image_data
            )

            logger.info(
                f"Successfully loaded image: {file_name} "
                f"(ID: {image_id}, Size: {file_size_mb:.2f}MB)"
            )
            return image_doc

        except Exception as e:
            raise ImageProcessorError(
                f"Failed to load image '{file_path}': {e}"
            ) from e

    def load_images_from_directory(
        self,
        dir_path: str | Path,
        recursive: bool = False,
        auto_caption: Optional[bool] = None,
        include_base64: bool = False,
        tags: Optional[list[str]] = None
    ) -> list[ImageDocument]:
        """ディレクトリ内の画像ファイルを一括読み込み

        Args:
            dir_path: 読み込むディレクトリのパス
            recursive: サブディレクトリも再帰的に探索するか
            auto_caption: キャプションを自動生成するか（Noneの場合は設定値を使用）
            include_base64: Base64エンコードした画像データを含めるか
            tags: すべての画像に付与するタグのリスト

        Returns:
            list[ImageDocument]: 読み込んだ画像ドキュメントのリスト

        Raises:
            ImageProcessorError: ディレクトリの読み込みに失敗した場合
            FileNotFoundError: ディレクトリが存在しない場合

        Example:
            >>> processor = ImageProcessor(vision_embeddings)
            >>> images = processor.load_images_from_directory("./images")
            >>> print(f"Loaded {len(images)} images")
        """
        path = Path(dir_path)

        # ディレクトリの存在確認
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        if not path.is_dir():
            raise ImageProcessorError(f"Path is not a directory: {dir_path}")

        logger.info(
            f"Loading images from directory: {path} "
            f"(recursive: {recursive})"
        )

        # 画像ファイルの探索
        image_files = []
        if recursive:
            for ext in self.SUPPORTED_EXTENSIONS:
                image_files.extend(path.rglob(f"*{ext}"))
                # 大文字のパターンも追加
                image_files.extend(path.rglob(f"*{ext.upper()}"))
        else:
            for ext in self.SUPPORTED_EXTENSIONS:
                image_files.extend(path.glob(f"*{ext}"))
                # 大文字のパターンも追加
                image_files.extend(path.glob(f"*{ext.upper()}"))

        # 重複を削除
        image_files = list(set(image_files))
        logger.info(f"Found {len(image_files)} image files")

        # 各画像を読み込み
        image_documents = []
        failed_files = []

        for i, image_file in enumerate(image_files, 1):
            try:
                logger.debug(f"Processing image {i}/{len(image_files)}: {image_file.name}")
                image_doc = self.load_image(
                    image_file,
                    auto_caption=auto_caption,
                    include_base64=include_base64,
                    tags=tags
                )
                image_documents.append(image_doc)
            except Exception as e:
                logger.error(f"Failed to load image '{image_file}': {e}")
                failed_files.append(str(image_file))

        # 結果のサマリー
        logger.info(
            f"Successfully loaded {len(image_documents)} images "
            f"from {path} (failed: {len(failed_files)})"
        )

        if failed_files:
            logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")

        return image_documents

    def __repr__(self) -> str:
        """文字列表現"""
        return (
            f"ImageProcessor("
            f"vision_model='{self.vision_embeddings.model_name}', "
            f"max_size={self.config.max_image_size_mb}MB"
            f")"
        )


# 便利関数: デフォルト設定で画像プロセッサーを作成
def create_image_processor(
    vision_embeddings: VisionEmbeddings,
    config: Optional[Config] = None
) -> ImageProcessor:
    """画像プロセッサーを作成

    Args:
        vision_embeddings: ビジョン埋め込み生成器
        config: アプリケーション設定（省略時はデフォルト設定を使用）

    Returns:
        ImageProcessor: 画像プロセッサーインスタンス

    Example:
        >>> from src.rag.vision_embeddings import create_vision_embeddings
        >>> vision_emb = create_vision_embeddings()
        >>> processor = create_image_processor(vision_emb)
        >>> images = processor.load_images_from_directory("./images")
    """
    return ImageProcessor(vision_embeddings, config)
