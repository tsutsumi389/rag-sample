"""設定管理モジュール

このモジュールはアプリケーション全体で使用する設定を一元管理します。
環境変数の読み込み、デフォルト設定の定義、設定値のバリデーションを行います。
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


class ConfigError(Exception):
    """設定エラー"""
    pass


class Config:
    """アプリケーション設定クラス

    環境変数から設定値を読み込み、デフォルト値を提供します。
    設定値のバリデーションも行います。
    """

    # デフォルト値
    DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
    DEFAULT_OLLAMA_LLM_MODEL = "gpt-oss"
    DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
    DEFAULT_OLLAMA_MULTIMODAL_LLM_MODEL = "gemma3"
    DEFAULT_OLLAMA_VISION_MODEL = "llava"
    DEFAULT_CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_LOG_LEVEL = "INFO"
    DEFAULT_IMAGE_CAPTION_AUTO_GENERATE = True
    DEFAULT_MAX_IMAGE_SIZE_MB = 10
    DEFAULT_IMAGE_RESIZE_ENABLED = False
    DEFAULT_IMAGE_RESIZE_MAX_WIDTH = 1024
    DEFAULT_IMAGE_RESIZE_MAX_HEIGHT = 1024
    DEFAULT_MULTIMODAL_SEARCH_TEXT_WEIGHT = 0.5
    DEFAULT_MULTIMODAL_SEARCH_IMAGE_WEIGHT = 0.5

    # バリデーション用の定数
    VALID_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    MIN_CHUNK_SIZE = 100
    MAX_CHUNK_SIZE = 10000
    MIN_CHUNK_OVERLAP = 0

    def __init__(self, env_file: Optional[str] = None):
        """設定の初期化

        Args:
            env_file: .envファイルのパス（省略時は.envを探索）
        """
        # .envファイルの読み込み
        if env_file:
            load_dotenv(env_file)
        else:
            # プロジェクトルートの.envを探索
            load_dotenv()

        # 設定値の読み込みとバリデーション
        self._load_and_validate()

    def _load_and_validate(self):
        """環境変数から設定値を読み込み、バリデーションを実行"""

        # Ollama設定
        self.ollama_base_url = os.getenv(
            "OLLAMA_BASE_URL",
            self.DEFAULT_OLLAMA_BASE_URL
        )
        self.ollama_llm_model = os.getenv(
            "OLLAMA_LLM_MODEL",
            self.DEFAULT_OLLAMA_LLM_MODEL
        )
        self.ollama_embedding_model = os.getenv(
            "OLLAMA_EMBEDDING_MODEL",
            self.DEFAULT_OLLAMA_EMBEDDING_MODEL
        )
        self.ollama_multimodal_llm_model = os.getenv(
            "OLLAMA_MULTIMODAL_LLM_MODEL",
            self.DEFAULT_OLLAMA_MULTIMODAL_LLM_MODEL
        )
        self.ollama_vision_model = os.getenv(
            "OLLAMA_VISION_MODEL",
            self.DEFAULT_OLLAMA_VISION_MODEL
        )

        # ChromaDB設定
        self.chroma_persist_directory = os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            self.DEFAULT_CHROMA_PERSIST_DIRECTORY
        )

        # チャンク設定
        try:
            self.chunk_size = int(os.getenv("CHUNK_SIZE", self.DEFAULT_CHUNK_SIZE))
        except ValueError:
            raise ConfigError(
                f"CHUNK_SIZE must be an integer, got: {os.getenv('CHUNK_SIZE')}"
            )

        try:
            self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", self.DEFAULT_CHUNK_OVERLAP))
        except ValueError:
            raise ConfigError(
                f"CHUNK_OVERLAP must be an integer, got: {os.getenv('CHUNK_OVERLAP')}"
            )

        # ログ設定
        self.log_level = os.getenv("LOG_LEVEL", self.DEFAULT_LOG_LEVEL).upper()

        # 画像処理設定
        self.image_caption_auto_generate = os.getenv(
            "IMAGE_CAPTION_AUTO_GENERATE",
            str(self.DEFAULT_IMAGE_CAPTION_AUTO_GENERATE)
        ).lower() == "true"

        try:
            self.max_image_size_mb = float(os.getenv(
                "MAX_IMAGE_SIZE_MB",
                str(self.DEFAULT_MAX_IMAGE_SIZE_MB)
            ))
        except ValueError:
            raise ConfigError(
                f"MAX_IMAGE_SIZE_MB must be a number, got: {os.getenv('MAX_IMAGE_SIZE_MB')}"
            )

        self.image_resize_enabled = os.getenv(
            "IMAGE_RESIZE_ENABLED",
            str(self.DEFAULT_IMAGE_RESIZE_ENABLED)
        ).lower() == "true"

        try:
            self.image_resize_max_width = int(os.getenv(
                "IMAGE_RESIZE_MAX_WIDTH",
                str(self.DEFAULT_IMAGE_RESIZE_MAX_WIDTH)
            ))
        except ValueError:
            raise ConfigError(
                f"IMAGE_RESIZE_MAX_WIDTH must be an integer, got: {os.getenv('IMAGE_RESIZE_MAX_WIDTH')}"
            )

        try:
            self.image_resize_max_height = int(os.getenv(
                "IMAGE_RESIZE_MAX_HEIGHT",
                str(self.DEFAULT_IMAGE_RESIZE_MAX_HEIGHT)
            ))
        except ValueError:
            raise ConfigError(
                f"IMAGE_RESIZE_MAX_HEIGHT must be an integer, got: {os.getenv('IMAGE_RESIZE_MAX_HEIGHT')}"
            )

        # マルチモーダル検索設定
        try:
            self.multimodal_search_text_weight = float(os.getenv(
                "MULTIMODAL_SEARCH_TEXT_WEIGHT",
                str(self.DEFAULT_MULTIMODAL_SEARCH_TEXT_WEIGHT)
            ))
        except ValueError:
            raise ConfigError(
                f"MULTIMODAL_SEARCH_TEXT_WEIGHT must be a number, got: {os.getenv('MULTIMODAL_SEARCH_TEXT_WEIGHT')}"
            )

        try:
            self.multimodal_search_image_weight = float(os.getenv(
                "MULTIMODAL_SEARCH_IMAGE_WEIGHT",
                str(self.DEFAULT_MULTIMODAL_SEARCH_IMAGE_WEIGHT)
            ))
        except ValueError:
            raise ConfigError(
                f"MULTIMODAL_SEARCH_IMAGE_WEIGHT must be a number, got: {os.getenv('MULTIMODAL_SEARCH_IMAGE_WEIGHT')}"
            )

        # バリデーション実行
        self._validate()

    def _validate(self):
        """設定値のバリデーション

        Raises:
            ConfigError: 設定値が不正な場合
        """
        # URLバリデーション
        if not self.ollama_base_url.startswith(("http://", "https://")):
            raise ConfigError(
                f"OLLAMA_BASE_URL must start with http:// or https://, "
                f"got: {self.ollama_base_url}"
            )

        # モデル名バリデーション（空文字チェック）
        if not self.ollama_llm_model.strip():
            raise ConfigError("OLLAMA_LLM_MODEL cannot be empty")

        if not self.ollama_embedding_model.strip():
            raise ConfigError("OLLAMA_EMBEDDING_MODEL cannot be empty")

        # チャンクサイズバリデーション
        if self.chunk_size < self.MIN_CHUNK_SIZE or self.chunk_size > self.MAX_CHUNK_SIZE:
            raise ConfigError(
                f"CHUNK_SIZE must be between {self.MIN_CHUNK_SIZE} and {self.MAX_CHUNK_SIZE}, "
                f"got: {self.chunk_size}"
            )

        # チャンクオーバーラップバリデーション
        if self.chunk_overlap < self.MIN_CHUNK_OVERLAP:
            raise ConfigError(
                f"CHUNK_OVERLAP must be >= {self.MIN_CHUNK_OVERLAP}, "
                f"got: {self.chunk_overlap}"
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be less than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )

        # ログレベルバリデーション
        if self.log_level not in self.VALID_LOG_LEVELS:
            raise ConfigError(
                f"LOG_LEVEL must be one of {self.VALID_LOG_LEVELS}, "
                f"got: {self.log_level}"
            )

        # 画像サイズバリデーション
        if self.max_image_size_mb <= 0:
            raise ConfigError(
                f"MAX_IMAGE_SIZE_MB must be greater than 0, "
                f"got: {self.max_image_size_mb}"
            )

        # 画像リサイズ設定バリデーション
        if self.image_resize_max_width <= 0:
            raise ConfigError(
                f"IMAGE_RESIZE_MAX_WIDTH must be greater than 0, "
                f"got: {self.image_resize_max_width}"
            )

        if self.image_resize_max_height <= 0:
            raise ConfigError(
                f"IMAGE_RESIZE_MAX_HEIGHT must be greater than 0, "
                f"got: {self.image_resize_max_height}"
            )

        # マルチモーダル検索の重みバリデーション
        if not (0.0 <= self.multimodal_search_text_weight <= 1.0):
            raise ConfigError(
                f"MULTIMODAL_SEARCH_TEXT_WEIGHT must be between 0.0 and 1.0, "
                f"got: {self.multimodal_search_text_weight}"
            )

        if not (0.0 <= self.multimodal_search_image_weight <= 1.0):
            raise ConfigError(
                f"MULTIMODAL_SEARCH_IMAGE_WEIGHT must be between 0.0 and 1.0, "
                f"got: {self.multimodal_search_image_weight}"
            )

    def get_chroma_path(self) -> Path:
        """ChromaDBの永続化ディレクトリパスを取得

        Returns:
            Path: ChromaDBディレクトリのPathオブジェクト
        """
        return Path(self.chroma_persist_directory).resolve()

    def ensure_chroma_directory(self):
        """ChromaDBディレクトリが存在しない場合は作成"""
        chroma_path = self.get_chroma_path()
        chroma_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """設定値を辞書形式で取得

        Returns:
            dict: 全設定値
        """
        return {
            "ollama_base_url": self.ollama_base_url,
            "ollama_llm_model": self.ollama_llm_model,
            "ollama_embedding_model": self.ollama_embedding_model,
            "ollama_multimodal_llm_model": self.ollama_multimodal_llm_model,
            "ollama_vision_model": self.ollama_vision_model,
            "chroma_persist_directory": self.chroma_persist_directory,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "log_level": self.log_level,
            "image_caption_auto_generate": self.image_caption_auto_generate,
            "max_image_size_mb": self.max_image_size_mb,
            "image_resize_enabled": self.image_resize_enabled,
            "image_resize_max_width": self.image_resize_max_width,
            "image_resize_max_height": self.image_resize_max_height,
            "multimodal_search_text_weight": self.multimodal_search_text_weight,
            "multimodal_search_image_weight": self.multimodal_search_image_weight,
        }

    def __repr__(self) -> str:
        """設定の文字列表現"""
        config_dict = self.to_dict()
        config_str = "\n".join(f"  {k}: {v}" for k, v in config_dict.items())
        return f"Config(\n{config_str}\n)"


# グローバル設定インスタンス（シングルトンパターン）
_config_instance: Optional[Config] = None


def get_config(env_file: Optional[str] = None, reload: bool = False) -> Config:
    """設定インスタンスを取得

    Args:
        env_file: .envファイルのパス（省略時は.envを探索）
        reload: Trueの場合、設定を再読み込み

    Returns:
        Config: 設定インスタンス
    """
    global _config_instance

    if _config_instance is None or reload:
        _config_instance = Config(env_file)

    return _config_instance
