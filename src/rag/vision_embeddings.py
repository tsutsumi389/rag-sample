"""ビジョン埋め込み生成モジュール

このモジュールはOllamaのビジョンモデル（llava/bakllava）を使用して
画像のベクトル埋め込みを生成し、画像のキャプション（説明文）を生成します。
マルチモーダルRAGシステムにおける画像検索の基盤となります。
"""

import base64
import logging
from pathlib import Path
from typing import Optional

import ollama

from src.utils.config import Config, get_config

# ロガー設定
logger = logging.getLogger(__name__)


class VisionEmbeddingError(Exception):
    """ビジョン埋め込み生成エラー"""
    pass


class VisionEmbeddings:
    """画像埋め込み生成クラス

    Ollamaのビジョンモデル（llava/bakllava）を使用して画像をベクトル化します。
    画像埋め込みの生成と画像キャプションの生成をサポートします。

    Attributes:
        config: アプリケーション設定
        model_name: 使用するビジョンモデル名
        base_url: Ollama APIのベースURL
        client: Ollama Clientインスタンス
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """ビジョン埋め込み生成器の初期化

        Args:
            config: 設定インスタンス（省略時はデフォルト設定を使用）
            model_name: ビジョンモデル名（省略時は設定ファイルの値を使用）
            base_url: Ollama APIのベースURL（省略時は設定ファイルの値を使用）

        Raises:
            VisionEmbeddingError: Ollamaとの接続に失敗した場合
        """
        self.config = config or get_config()
        self.model_name = model_name or getattr(
            self.config, 'ollama_vision_model', 'llava'
        )
        self.base_url = base_url or self.config.ollama_base_url

        # Ollama Clientの初期化
        try:
            self.client = ollama.Client(host=self.base_url)
            # モデルの存在確認
            self._verify_model()
            logger.info(
                f"VisionEmbeddings initialized with model '{self.model_name}' "
                f"at {self.base_url}"
            )
        except Exception as e:
            raise VisionEmbeddingError(
                f"Failed to initialize Ollama client: {e}\n"
                f"Make sure Ollama is running at {self.base_url} and "
                f"model '{self.model_name}' is available (run: ollama pull {self.model_name})."
            ) from e

    def _verify_model(self) -> None:
        """モデルの存在を確認する

        Raises:
            VisionEmbeddingError: モデルが利用できない場合
        """
        try:
            models = self.client.list()
            # ollamaライブラリの型は辞書のようにアクセスできないため、属性アクセスを使用
            # modelオブジェクトの`model`属性にモデル名が格納されている（例: "llava:latest"）
            model_list = models.get('models', []) if isinstance(models, dict) else models.models
            model_names = []
            for model in model_list:
                if isinstance(model, dict):
                    # Try 'name' first, fallback to 'model'
                    name = model.get('name') or model.get('model')
                    if name:
                        model_names.append(name.split(':')[0])
                else:
                    # Assume object with .model attribute
                    if hasattr(model, 'model'):
                        model_names.append(model.model.split(':')[0])
            if self.model_name not in model_names:
                raise VisionEmbeddingError(
                    f"Vision model '{self.model_name}' is not available. "
                    f"Please run: ollama pull {self.model_name}"
                )
        except ollama.ResponseError as e:
            raise VisionEmbeddingError(
                f"Failed to verify model '{self.model_name}': {e}"
            ) from e

    def _encode_image_base64(self, image_path: str | Path) -> str:
        """画像ファイルをBase64エンコードする

        Args:
            image_path: 画像ファイルのパス

        Returns:
            str: Base64エンコードされた画像データ

        Raises:
            VisionEmbeddingError: 画像の読み込みまたはエンコードに失敗した場合
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise VisionEmbeddingError(
                f"Failed to encode image '{image_path}': {e}"
            ) from e

    def embed_image(self, image_path: str | Path) -> list[float]:
        """画像をベクトル化

        画像ファイルを読み込み、ベクトル表現を生成します。
        画像の詳細なキャプションを生成し、そのテキストをベクトル化することで
        画像の意味的な埋め込みを作成します。

        Args:
            image_path: ベクトル化する画像ファイルのパス

        Returns:
            list[float]: 画像のベクトル表現

        Raises:
            VisionEmbeddingError: ベクトル化に失敗した場合
            ValueError: image_pathが無効な場合

        Example:
            >>> generator = VisionEmbeddings()
            >>> embedding = generator.embed_image("path/to/image.jpg")
            >>> isinstance(embedding, list)
            True
        """
        if not image_path:
            raise ValueError("image_path cannot be empty")

        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"Image file not found: {image_path}")

        try:
            logger.debug(f"Generating embedding for image: {path.name}")

            # ステップ1: 画像の詳細な説明を生成
            # より包括的なプロンプトで画像の内容を詳しく抽出
            caption_prompt = (
                "この画像について、以下の観点から詳しく説明してください:\n"
                "1. 何が写っているか（オブジェクト、人物、場所など）\n"
                "2. 色、形、テクスチャなどの視覚的特徴\n"
                "3. 画像の雰囲気や文脈\n"
                "4. テキストが含まれている場合はその内容\n"
                "簡潔かつ具体的に記述してください。"
            )

            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': caption_prompt,
                        'images': [str(path)]
                    }
                ],
                options={
                    'num_predict': 500,  # より詳細な説明を得るために長めに設定
                }
            )

            caption = response.get('message', {}).get('content', '').strip()
            if not caption:
                raise VisionEmbeddingError(
                    f"Empty caption returned for image: {image_path}"
                )

            logger.debug(f"Generated caption for {path.name}: {caption[:100]}...")

            # ステップ2: キャプションのテキストをベクトル化
            # テキスト埋め込みモデルを使用（nomic-embed-textなど）
            embedding_model = getattr(
                self.config, 'ollama_embedding_model', 'nomic-embed-text'
            )

            embed_response = self.client.embeddings(
                model=embedding_model,
                prompt=caption
            )

            embedding = embed_response.get('embedding', [])
            if not embedding:
                raise VisionEmbeddingError(
                    f"Empty embedding returned for caption of image: {image_path}"
                )

            logger.debug(
                f"Successfully generated embedding (dim: {len(embedding)}) "
                f"for {path.name}"
            )
            return embedding

        except ollama.ResponseError as e:
            raise VisionEmbeddingError(
                f"Failed to embed image '{image_path}': {e}"
            ) from e
        except Exception as e:
            raise VisionEmbeddingError(
                f"Unexpected error while embedding image '{image_path}': {e}"
            ) from e

    def embed_images(self, image_paths: list[str | Path]) -> list[list[float]]:
        """複数の画像をバッチでベクトル化

        画像パスのリストを受け取り、それぞれのベクトル表現を生成します。

        Args:
            image_paths: ベクトル化する画像ファイルパスのリスト

        Returns:
            list[list[float]]: 各画像のベクトル表現のリスト

        Raises:
            VisionEmbeddingError: ベクトル化に失敗した場合
            ValueError: image_pathsが空または無効な場合

        Example:
            >>> generator = VisionEmbeddings()
            >>> paths = ["image1.jpg", "image2.png"]
            >>> embeddings = generator.embed_images(paths)
            >>> len(embeddings) == len(paths)
            True
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        logger.info(f"Generating embeddings for {len(image_paths)} images")
        embeddings = []
        failed_images = []

        for i, image_path in enumerate(image_paths, 1):
            try:
                embedding = self.embed_image(image_path)
                embeddings.append(embedding)
                logger.debug(f"Progress: {i}/{len(image_paths)} images processed")
            except Exception as e:
                logger.error(f"Failed to embed image '{image_path}': {e}")
                failed_images.append(str(image_path))
                # エラーが発生した画像は空のベクトルで代替（または例外を投げる）
                # ここでは厳密にエラーとして扱う
                raise VisionEmbeddingError(
                    f"Failed to embed image '{image_path}' "
                    f"(index {i}/{len(image_paths)}): {e}"
                ) from e

        logger.info(
            f"Successfully generated embeddings for {len(embeddings)} images"
        )
        return embeddings

    def generate_caption(
        self,
        image_path: str | Path,
        prompt: str = "この画像について簡潔に説明してください。",
        max_tokens: int = 200
    ) -> str:
        """画像のキャプション（説明文）を生成

        ビジョンモデルを使用して画像の内容を説明するテキストを生成します。

        Args:
            image_path: キャプションを生成する画像ファイルのパス
            prompt: キャプション生成用のプロンプト
            max_tokens: 生成する最大トークン数

        Returns:
            str: 生成されたキャプション

        Raises:
            VisionEmbeddingError: キャプション生成に失敗した場合
            ValueError: image_pathが無効な場合

        Example:
            >>> generator = VisionEmbeddings()
            >>> caption = generator.generate_caption("path/to/image.jpg")
            >>> isinstance(caption, str)
            True
        """
        if not image_path:
            raise ValueError("image_path cannot be empty")

        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"Image file not found: {image_path}")

        try:
            logger.debug(f"Generating caption for image: {path.name}")
            # Ollamaのchat APIを使用
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [str(path)]
                    }
                ],
                options={
                    'num_predict': max_tokens,
                }
            )
            caption = response.get('message', {}).get('content', '').strip()
            if not caption:
                raise VisionEmbeddingError(
                    f"Empty caption returned for image: {image_path}"
                )
            logger.debug(f"Successfully generated caption for {path.name}")
            return caption
        except ollama.ResponseError as e:
            raise VisionEmbeddingError(
                f"Failed to generate caption for image '{image_path}': {e}"
            ) from e
        except Exception as e:
            raise VisionEmbeddingError(
                f"Unexpected error while generating caption for image '{image_path}': {e}"
            ) from e

    def get_embedding_dimension(self) -> int:
        """埋め込みベクトルの次元数を取得

        ビジョンモデルが生成するベクトルの次元数を返します。

        Returns:
            int: ベクトルの次元数

        Raises:
            VisionEmbeddingError: 次元数の取得に失敗した場合

        Note:
            この操作にはサンプル画像が必要です。
            実際のプロジェクトではダミー画像を使用するか、
            事前に次元数を設定ファイルで定義することを推奨します。
        """
        # Note: 実際には画像を使わずに次元数を取得する方法がない場合、
        # 設定ファイルに固定値として定義することを推奨
        try:
            # ダミー画像の埋め込みを生成して次元数を取得
            # （実運用では設定ファイルで定義する方が効率的）
            logger.warning(
                "get_embedding_dimension() requires a sample image. "
                "Consider defining dimension in config file."
            )
            raise NotImplementedError(
                "Dimension detection requires a sample image. "
                "Please use a test image or define dimension in config."
            )
        except Exception as e:
            raise VisionEmbeddingError(
                f"Failed to get embedding dimension: {e}"
            ) from e

    def __repr__(self) -> str:
        """文字列表現"""
        return (
            f"VisionEmbeddings("
            f"model='{self.model_name}', "
            f"base_url='{self.base_url}'"
            f")"
        )


# 便利関数: デフォルト設定でビジョン埋め込み生成器を作成
def create_vision_embeddings(
    model_name: Optional[str] = None,
    base_url: Optional[str] = None
) -> VisionEmbeddings:
    """ビジョン埋め込み生成器を作成

    デフォルト設定を使用してビジョン埋め込み生成器を作成します。

    Args:
        model_name: ビジョンモデル名（省略時は設定ファイルの値を使用）
        base_url: Ollama APIのベースURL（省略時は設定ファイルの値を使用）

    Returns:
        VisionEmbeddings: ビジョン埋め込み生成器インスタンス

    Example:
        >>> generator = create_vision_embeddings()
        >>> embedding = generator.embed_image("path/to/image.jpg")
    """
    return VisionEmbeddings(model_name=model_name, base_url=base_url)
