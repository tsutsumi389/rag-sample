"""設定管理のユニットテスト

Config クラスの機能を検証します。
"""

import os
import tempfile
from pathlib import Path
import pytest

from src.utils.config import Config, ConfigError, get_config


class TestConfigNormalCases:
    """Config クラスの正常系テスト"""

    def test_default_config_creation(self, monkeypatch, tmp_path):
        """デフォルト値でのConfig作成"""
        # 環境変数をクリア
        for key in [
            "OLLAMA_BASE_URL",
            "OLLAMA_LLM_MODEL",
            "OLLAMA_EMBEDDING_MODEL",
            "CHROMA_PERSIST_DIRECTORY",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "LOG_LEVEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        # 空の.envファイルを作成して、プロジェクトの.envが読み込まれないようにする
        empty_env_file = tmp_path / "empty.env"
        empty_env_file.write_text("")

        config = Config(env_file=str(empty_env_file))

        # デフォルト値の確認
        assert config.ollama_base_url == Config.DEFAULT_OLLAMA_BASE_URL
        assert config.ollama_llm_model == Config.DEFAULT_OLLAMA_LLM_MODEL
        assert config.ollama_embedding_model == Config.DEFAULT_OLLAMA_EMBEDDING_MODEL
        assert config.chroma_persist_directory == Config.DEFAULT_CHROMA_PERSIST_DIRECTORY
        assert config.chunk_size == Config.DEFAULT_CHUNK_SIZE
        assert config.chunk_overlap == Config.DEFAULT_CHUNK_OVERLAP
        assert config.log_level == Config.DEFAULT_LOG_LEVEL

    def test_config_from_environment_variables(self, monkeypatch):
        """環境変数からの設定読み込み"""
        # テスト用の環境変数を設定
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://test-server:8080")
        monkeypatch.setenv("OLLAMA_LLM_MODEL", "test-llm-model")
        monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "test-embedding-model")
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "./test_chroma")
        monkeypatch.setenv("CHUNK_SIZE", "500")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        config = Config(env_file=None)

        # 環境変数から読み込まれた値を確認
        assert config.ollama_base_url == "http://test-server:8080"
        assert config.ollama_llm_model == "test-llm-model"
        assert config.ollama_embedding_model == "test-embedding-model"
        assert config.chroma_persist_directory == "./test_chroma"
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100
        assert config.log_level == "DEBUG"

    def test_config_from_custom_env_file(self, tmp_path, monkeypatch):
        """カスタム.envファイルからの読み込み"""
        # 環境変数をクリア
        for key in [
            "OLLAMA_BASE_URL",
            "OLLAMA_LLM_MODEL",
            "OLLAMA_EMBEDDING_MODEL",
            "CHROMA_PERSIST_DIRECTORY",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "LOG_LEVEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        # テスト用の.envファイルを作成
        env_file = tmp_path / "test.env"
        env_file.write_text(
            "OLLAMA_BASE_URL=http://custom-server:9090\n"
            "OLLAMA_LLM_MODEL=custom-llm\n"
            "OLLAMA_EMBEDDING_MODEL=custom-embedding\n"
            "CHROMA_PERSIST_DIRECTORY=./custom_chroma\n"
            "CHUNK_SIZE=800\n"
            "CHUNK_OVERLAP=150\n"
            "LOG_LEVEL=WARNING\n"
        )

        config = Config(env_file=str(env_file))

        # .envファイルから読み込まれた値を確認
        assert config.ollama_base_url == "http://custom-server:9090"
        assert config.ollama_llm_model == "custom-llm"
        assert config.ollama_embedding_model == "custom-embedding"
        assert config.chroma_persist_directory == "./custom_chroma"
        assert config.chunk_size == 800
        assert config.chunk_overlap == 150
        assert config.log_level == "WARNING"

    def test_to_dict_method(self, monkeypatch, tmp_path):
        """to_dict()メソッドが全設定を返すことを確認"""
        # 環境変数をクリア
        for key in [
            "OLLAMA_BASE_URL",
            "OLLAMA_LLM_MODEL",
            "OLLAMA_EMBEDDING_MODEL",
            "CHROMA_PERSIST_DIRECTORY",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "LOG_LEVEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        # 空の.envファイルを作成して、プロジェクトの.envが読み込まれないようにする
        empty_env_file = tmp_path / "empty.env"
        empty_env_file.write_text("")

        config = Config(env_file=str(empty_env_file))
        config_dict = config.to_dict()

        # 辞書に全ての設定項目が含まれることを確認
        assert "ollama_base_url" in config_dict
        assert "ollama_llm_model" in config_dict
        assert "ollama_embedding_model" in config_dict
        assert "chroma_persist_directory" in config_dict
        assert "chunk_size" in config_dict
        assert "chunk_overlap" in config_dict
        assert "log_level" in config_dict

        # 値が正しいことを確認
        assert config_dict["ollama_base_url"] == Config.DEFAULT_OLLAMA_BASE_URL
        assert config_dict["ollama_llm_model"] == Config.DEFAULT_OLLAMA_LLM_MODEL
        assert config_dict["ollama_embedding_model"] == Config.DEFAULT_OLLAMA_EMBEDDING_MODEL
        assert config_dict["chroma_persist_directory"] == Config.DEFAULT_CHROMA_PERSIST_DIRECTORY
        assert config_dict["chunk_size"] == Config.DEFAULT_CHUNK_SIZE
        assert config_dict["chunk_overlap"] == Config.DEFAULT_CHUNK_OVERLAP
        assert config_dict["log_level"] == Config.DEFAULT_LOG_LEVEL

    def test_get_chroma_path_returns_path_object(self, monkeypatch):
        """get_chroma_path()が正しいPathオブジェクトを返すことを確認"""
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", "./test_chroma_db")

        config = Config(env_file=None)
        chroma_path = config.get_chroma_path()

        # Pathオブジェクトが返されることを確認
        assert isinstance(chroma_path, Path)
        # 絶対パスに変換されることを確認
        assert chroma_path.is_absolute()
        # パスの末尾が正しいことを確認
        assert chroma_path.name == "test_chroma_db"

    def test_ensure_chroma_directory_creates_directory(self, tmp_path, monkeypatch):
        """ensure_chroma_directory()でディレクトリが作成されることを確認"""
        # テスト用の一時ディレクトリパスを設定
        test_chroma_dir = tmp_path / "new_chroma_db"
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(test_chroma_dir))

        config = Config(env_file=None)

        # ディレクトリが存在しないことを確認
        assert not test_chroma_dir.exists()

        # ディレクトリを作成
        config.ensure_chroma_directory()

        # ディレクトリが作成されたことを確認
        assert test_chroma_dir.exists()
        assert test_chroma_dir.is_dir()

    def test_ensure_chroma_directory_with_existing_directory(self, tmp_path, monkeypatch):
        """ensure_chroma_directory()で既存ディレクトリが問題なく処理されることを確認"""
        # 既存のディレクトリを作成
        test_chroma_dir = tmp_path / "existing_chroma_db"
        test_chroma_dir.mkdir()
        monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(test_chroma_dir))

        config = Config(env_file=None)

        # 既存ディレクトリに対してエラーが発生しないことを確認
        config.ensure_chroma_directory()

        # ディレクトリが存在することを確認
        assert test_chroma_dir.exists()
        assert test_chroma_dir.is_dir()


class TestGetConfigFunction:
    """get_config 関数のテスト"""

    def test_singleton_pattern(self, monkeypatch):
        """シングルトンパターンが機能することを確認"""
        # 環境変数をクリア
        for key in [
            "OLLAMA_BASE_URL",
            "OLLAMA_LLM_MODEL",
            "OLLAMA_EMBEDDING_MODEL",
            "CHROMA_PERSIST_DIRECTORY",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "LOG_LEVEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        # グローバルインスタンスをリセット
        import src.utils.config as config_module
        config_module._config_instance = None

        # 2回取得して同じインスタンスが返されることを確認
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_reload_flag_reloads_config(self, monkeypatch):
        """reload=Trueで設定が再読み込みされることを確認"""
        # 環境変数をクリア
        for key in [
            "OLLAMA_BASE_URL",
            "OLLAMA_LLM_MODEL",
            "OLLAMA_EMBEDDING_MODEL",
            "CHROMA_PERSIST_DIRECTORY",
            "CHUNK_SIZE",
            "CHUNK_OVERLAP",
            "LOG_LEVEL",
        ]:
            monkeypatch.delenv(key, raising=False)

        # グローバルインスタンスをリセット
        import src.utils.config as config_module
        config_module._config_instance = None

        # 最初の設定を取得
        config1 = get_config()
        original_url = config1.ollama_base_url

        # 環境変数を変更
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://changed-server:7777")

        # reload=Falseでは変更されない
        config2 = get_config(reload=False)
        assert config2.ollama_base_url == original_url

        # reload=Trueで変更が反映される
        config3 = get_config(reload=True)
        assert config3.ollama_base_url == "http://changed-server:7777"
