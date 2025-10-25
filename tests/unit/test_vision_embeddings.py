"""VisionEmbeddingsクラスのユニットテスト

ビジョン埋め込み生成とキャプション生成機能をテストします。
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.rag.vision_embeddings import (
    VisionEmbeddings,
    VisionEmbeddingError,
    create_vision_embeddings
)


class TestVisionEmbeddings:
    """VisionEmbeddingsクラスのテスト"""

    def test_init_with_default_config(self, multimodal_config):
        """デフォルト設定での初期化テスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'llava:latest'}]
            }
            mock_client_class.return_value = mock_client

            embeddings = VisionEmbeddings(config=multimodal_config)

            # モデル名は設定から取得される（.envファイルまたはデフォルト値）
            # ルートディレクトリの.envがある場合はそちらが優先される
            assert embeddings.model_name in ["llava", "llava:latest"]
            assert embeddings.base_url == "http://localhost:11434"
            assert embeddings.config == multimodal_config
            mock_client_class.assert_called_once()

    def test_init_with_custom_model(self, multimodal_config):
        """カスタムモデル名での初期化テスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'bakllava:latest'}]
            }
            mock_client_class.return_value = mock_client

            embeddings = VisionEmbeddings(
                config=multimodal_config,
                model_name="bakllava"
            )

            assert embeddings.model_name == "bakllava"

    def test_verify_model_success(self, multimodal_config):
        """モデル検証成功のテスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [
                    {'name': 'llava:latest'},
                    {'name': 'gpt-oss:latest'}
                ]
            }
            mock_client_class.return_value = mock_client

            # 例外が発生しないことを確認
            embeddings = VisionEmbeddings(config=multimodal_config)
            assert embeddings.model_name in ["llava", "llava:latest"]

    def test_verify_model_not_found(self, multimodal_config):
        """モデルが見つからない場合のテスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'gpt-oss:latest'}]
            }
            mock_client_class.return_value = mock_client

            with pytest.raises(VisionEmbeddingError) as exc_info:
                VisionEmbeddings(config=multimodal_config)

            assert "not available" in str(exc_info.value)
            assert "llava" in str(exc_info.value)

    def test_embed_image_success(self, multimodal_config, sample_image_files):
        """画像埋め込み生成成功のテスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'llava:latest'}]
            }
            # chat APIのモック（キャプション生成）
            mock_client.chat.return_value = {
                'message': {'content': 'A test image with blue background and white circle'}
            }
            # embeddings APIのモック
            mock_client.embeddings.return_value = {
                'embedding': [0.1, 0.2, 0.3] * 100
            }
            mock_client_class.return_value = mock_client

            embeddings = VisionEmbeddings(config=multimodal_config)
            result = embeddings.embed_image(sample_image_files["sample1"])

            assert isinstance(result, list)
            assert len(result) == 300
            assert all(isinstance(x, float) for x in result)
            mock_client.chat.assert_called_once()
            mock_client.embeddings.assert_called_once()

    def test_embed_image_file_not_found(self, multimodal_config):
        """存在しない画像ファイルのテスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'llava:latest'}]
            }
            mock_client_class.return_value = mock_client

            embeddings = VisionEmbeddings(config=multimodal_config)

            with pytest.raises(ValueError) as exc_info:
                embeddings.embed_image("nonexistent.jpg")

            assert "not found" in str(exc_info.value).lower()

    def test_embed_image_empty_path(self, multimodal_config):
        """空のパスでのテスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'llava:latest'}]
            }
            mock_client_class.return_value = mock_client

            embeddings = VisionEmbeddings(config=multimodal_config)

            with pytest.raises(ValueError) as exc_info:
                embeddings.embed_image("")

            assert "cannot be empty" in str(exc_info.value)

    def test_embed_images_batch(self, multimodal_config, sample_image_files):
        """複数画像のバッチ埋め込みテスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'llava:latest'}]
            }
            # chat APIのモック（各画像のキャプション生成）
            mock_client.chat.return_value = {
                'message': {'content': 'Test image description'}
            }
            # 各画像に対して異なる埋め込みを返す
            mock_client.embeddings.side_effect = [
                {'embedding': [0.1] * 300},
                {'embedding': [0.2] * 300},
                {'embedding': [0.3] * 300},
            ]
            mock_client_class.return_value = mock_client

            embeddings = VisionEmbeddings(config=multimodal_config)
            image_paths = list(sample_image_files.values())
            results = embeddings.embed_images(image_paths)

            assert len(results) == 3
            assert all(len(emb) == 300 for emb in results)
            assert mock_client.embeddings.call_count == 3

    def test_embed_images_empty_list(self, multimodal_config):
        """空のリストでのテスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'llava:latest'}]
            }
            mock_client_class.return_value = mock_client

            embeddings = VisionEmbeddings(config=multimodal_config)

            with pytest.raises(ValueError) as exc_info:
                embeddings.embed_images([])

            assert "cannot be empty" in str(exc_info.value)

    def test_generate_caption_success(self, multimodal_config, sample_image_files):
        """キャプション生成成功のテスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'llava:latest'}]
            }
            # chat APIのモック
            mock_client.chat.return_value = {
                'message': {
                    'content': 'これは青い背景に白い円が描かれた画像です。'
                }
            }
            mock_client_class.return_value = mock_client

            embeddings = VisionEmbeddings(config=multimodal_config)
            caption = embeddings.generate_caption(sample_image_files["sample1"])

            assert isinstance(caption, str)
            assert len(caption) > 0
            assert "画像" in caption
            mock_client.chat.assert_called_once()

    def test_generate_caption_custom_prompt(self, multimodal_config, sample_image_files):
        """カスタムプロンプトでのキャプション生成テスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'llava:latest'}]
            }
            mock_client.chat.return_value = {
                'message': {
                    'content': '青い四角形'
                }
            }
            mock_client_class.return_value = mock_client

            embeddings = VisionEmbeddings(config=multimodal_config)
            caption = embeddings.generate_caption(
                sample_image_files["sample1"],
                prompt="この画像の主な形状を一言で説明してください。"
            )

            assert isinstance(caption, str)
            mock_client.chat.assert_called_once()

    def test_generate_caption_file_not_found(self, multimodal_config):
        """存在しない画像のキャプション生成テスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'llava:latest'}]
            }
            mock_client_class.return_value = mock_client

            embeddings = VisionEmbeddings(config=multimodal_config)

            with pytest.raises(ValueError) as exc_info:
                embeddings.generate_caption("nonexistent.jpg")

            assert "not found" in str(exc_info.value).lower()

    def test_repr(self, multimodal_config):
        """文字列表現のテスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'llava:latest'}]
            }
            mock_client_class.return_value = mock_client

            embeddings = VisionEmbeddings(config=multimodal_config)
            repr_str = repr(embeddings)

            assert "VisionEmbeddings" in repr_str
            assert "llava" in repr_str
            assert "localhost" in repr_str


class TestCreateVisionEmbeddings:
    """create_vision_embeddings便利関数のテスト"""

    def test_create_default(self):
        """デフォルト設定での作成テスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'llava:latest'}]
            }
            mock_client_class.return_value = mock_client

            embeddings = create_vision_embeddings()

            assert isinstance(embeddings, VisionEmbeddings)
            assert embeddings.model_name in ["llava", "llava:latest"]

    def test_create_custom_model(self):
        """カスタムモデルでの作成テスト"""
        with patch('src.rag.vision_embeddings.ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client.list.return_value = {
                'models': [{'name': 'bakllava:latest'}]
            }
            mock_client_class.return_value = mock_client

            embeddings = create_vision_embeddings(model_name="bakllava")

            assert embeddings.model_name == "bakllava"
