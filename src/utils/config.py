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
    DEFAULT_CHROMA_PERSIST_DIRECTORY = "./chroma_db"
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    DEFAULT_LOG_LEVEL = "INFO"

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
            "chroma_persist_directory": self.chroma_persist_directory,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "log_level": self.log_level,
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
