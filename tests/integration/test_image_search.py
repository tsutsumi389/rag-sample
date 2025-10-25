"""画像検索機能の統合テスト

画像の追加、検索、取得のエンドツーエンドフローをテストします。
実際のOllamaのビジョンモデルとChromaDBを使用します。

注意: これらのテストは実際のOllamaサービスとビジョンモデル（llava）が必要です。
Ollamaが起動していない場合、またはビジョンモデルがインストールされていない場合、
テストはスキップされます。
"""

import pytest
from pathlib import Path
import tempfile
import shutil
from PIL import Image

from src.rag.vector_store import BaseVectorStore, create_vector_store
from src.rag.vision_embeddings import VisionEmbeddings
from src.rag.image_processor import ImageProcessor
from src.rag.embeddings import EmbeddingGenerator
from src.utils.config import Config


# Ollamaとビジョンモデルの起動チェック用のfixture
@pytest.fixture(scope="module")
def check_ollama_vision():
    """Ollamaサービスとビジョンモデルが利用可能かチェックする

    Yields:
        bool: Ollamaとビジョンモデルが利用可能な場合True
    """
    import requests

    try:
        # Ollamaサービスのチェック
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            pytest.skip("Ollama service is not responding correctly")

        # ビジョンモデルがインストールされているかチェック
        models = response.json().get("models", [])
        vision_model_installed = any(
            "llava" in model.get("name", "").lower() for model in models
        )

        if not vision_model_installed:
            pytest.skip("Vision model (llava) is not installed. Run: ollama pull llava")

        yield True

    except Exception as e:
        pytest.skip(f"Ollama service is not available: {e}")


@pytest.fixture
def sample_images(tmp_path):
    """テスト用のサンプル画像を作成する

    Args:
        tmp_path: pytest提供の一時ディレクトリ

    Returns:
        dict[str, Path]: 画像名とパスのマッピング
    """
    images_dir = tmp_path / "test_images"
    images_dir.mkdir()

    # 異なる色のシンプルな画像を作成
    images = {}

    # 赤い画像
    red_img = Image.new('RGB', (100, 100), color='red')
    red_path = images_dir / "red_image.jpg"
    red_img.save(red_path)
    images["red"] = red_path

    # 青い画像
    blue_img = Image.new('RGB', (100, 100), color='blue')
    blue_path = images_dir / "blue_image.png"
    blue_img.save(blue_path)
    images["blue"] = blue_path

    # 緑の画像
    green_img = Image.new('RGB', (100, 100), color='green')
    green_path = images_dir / "green_image.jpg"
    green_img.save(green_path)
    images["green"] = green_path

    return images


@pytest.mark.integration
class TestImageSearch:
    """画像検索のエンドツーエンドテスト（実際のChromaDB使用）"""

    def test_add_and_list_images(
        self,
        integration_config: Config,
        sample_images: dict[str, Path],
        check_ollama_vision
    ):
        """画像の追加と一覧表示のテスト

        Args:
            integration_config: 統合テスト用の設定
            sample_images: サンプル画像ファイルの辞書
            check_ollama_vision: Ollamaビジョンモデルチェック
        """
        # コンポーネントの初期化
        vector_store = create_vector_store(integration_config)
        vision_embeddings = VisionEmbeddings(integration_config)
        image_processor = ImageProcessor(vision_embeddings, integration_config)

        try:
            # 1. ベクトルストアの初期化
            vector_store.initialize()

            # 2. 画像の読み込み
            red_image_path = str(sample_images["red"])
            image_doc = image_processor.load_image(red_image_path)

            assert image_doc is not None
            assert image_doc.file_name == "red_image.jpg"
            assert image_doc.image_type == "jpg"
            assert len(image_doc.caption) > 0
            print(f"Loaded image: {image_doc.file_name}")
            print(f"Caption: {image_doc.caption}")

            # 3. 埋め込みの生成
            embedding = vision_embeddings.embed_image(red_image_path)

            assert len(embedding) > 0
            print(f"Generated embedding with dimension: {len(embedding)}")

            # 4. ベクトルストアに追加
            image_ids = vector_store.add_images([image_doc], [embedding])

            assert len(image_ids) == 1
            print(f"Added image with ID: {image_ids[0]}")

            # 5. 画像一覧の取得
            images = vector_store.list_images()

            assert len(images) >= 1
            assert any(img.file_name == "red_image.jpg" for img in images)
            print(f"Total images in store: {len(images)}")

        finally:
            # クリーンアップ
            vector_store.clear()

    def test_search_images_by_text(
        self,
        integration_config: Config,
        sample_images: dict[str, Path],
        check_ollama_vision
    ):
        """テキストクエリによる画像検索のテスト

        Args:
            integration_config: 統合テスト用の設定
            sample_images: サンプル画像ファイルの辞書
            check_ollama_vision: Ollamaビジョンモデルチェック
        """
        # コンポーネントの初期化
        vector_store = create_vector_store(integration_config)
        vision_embeddings = VisionEmbeddings(integration_config)
        image_processor = ImageProcessor(vision_embeddings, integration_config)
        embedding_generator = EmbeddingGenerator(integration_config)

        try:
            # 1. ベクトルストアの初期化
            vector_store.initialize()

            # 2. 複数の画像を追加
            images = []
            embeddings = []

            for color, image_path in sample_images.items():
                image_doc = image_processor.load_image(str(image_path))
                images.append(image_doc)

                embedding = vision_embeddings.embed_image(str(image_path))
                embeddings.append(embedding)

                print(f"Loaded {color} image: {image_doc.caption[:50]}...")

            # ベクトルストアに追加
            image_ids = vector_store.add_images(images, embeddings)
            assert len(image_ids) == len(sample_images)
            print(f"Added {len(image_ids)} images to vector store")

            # 3. テキストクエリで検索
            # キャプションベースの検索（テキスト埋め込みを使用）
            query = "red color"
            query_embedding = embedding_generator.embed_query(query)

            search_results = vector_store.search_images(
                query_embedding=query_embedding,
                top_k=3
            )

            assert len(search_results) > 0
            print(f"Search for '{query}' returned {len(search_results)} results")

            # 検索結果の確認
            for i, result in enumerate(search_results, 1):
                print(f"\nResult {i}:")
                print(f"  File: {result.chunk.metadata.get('file_name', 'N/A')}")
                print(f"  Score: {result.score:.4f}")
                caption = result.chunk.metadata.get('caption', 'N/A')
                print(f"  Caption: {caption[:50] if caption != 'N/A' else 'N/A'}...")

            # スコアが0-1の範囲であることを確認
            assert all(
                0 <= result.score <= 1 for result in search_results
            )

        finally:
            # クリーンアップ
            vector_store.clear()

    def test_remove_image(
        self,
        integration_config: Config,
        sample_images: dict[str, Path],
        check_ollama_vision
    ):
        """画像の削除テスト

        Args:
            integration_config: 統合テスト用の設定
            sample_images: サンプル画像ファイルの辞書
            check_ollama_vision: Ollamaビジョンモデルチェック
        """
        # コンポーネントの初期化
        vector_store = create_vector_store(integration_config)
        vision_embeddings = VisionEmbeddings(integration_config)
        image_processor = ImageProcessor(vision_embeddings, integration_config)

        try:
            # 1. ベクトルストアの初期化
            vector_store.initialize()

            # 2. 画像を追加
            image_path = str(sample_images["red"])
            image_doc = image_processor.load_image(image_path)
            embedding = vision_embeddings.embed_image(image_path)

            image_ids = vector_store.add_images([image_doc], [embedding])
            image_id = image_ids[0]

            # 追加確認
            images = vector_store.list_images()
            assert len(images) >= 1
            print(f"Added image with ID: {image_id}")

            # 3. 画像を取得
            retrieved_image = vector_store.get_image_by_id(image_id)
            assert retrieved_image is not None
            assert retrieved_image.id == image_id
            print(f"Retrieved image: {retrieved_image.file_name}")

            # 4. 画像を削除
            success = vector_store.remove_image(image_id)
            assert success is True
            print(f"Removed image: {image_id}")

            # 削除確認
            removed_image = vector_store.get_image_by_id(image_id)
            assert removed_image is None

            images_after = vector_store.list_images()
            assert not any(img.id == image_id for img in images_after)

        finally:
            # クリーンアップ
            vector_store.clear()

    def test_load_images_from_directory(
        self,
        integration_config: Config,
        sample_images: dict[str, Path],
        check_ollama_vision
    ):
        """ディレクトリから複数画像を読み込むテスト

        Args:
            integration_config: 統合テスト用の設定
            sample_images: サンプル画像ファイルの辞書
            check_ollama_vision: Ollamaビジョンモデルチェック
        """
        # コンポーネントの初期化
        vector_store = create_vector_store(integration_config)
        vision_embeddings = VisionEmbeddings(integration_config)
        image_processor = ImageProcessor(vision_embeddings, integration_config)

        try:
            # 1. ベクトルストアの初期化
            vector_store.initialize()

            # 2. ディレクトリから画像を一括読み込み
            images_dir = sample_images["red"].parent
            loaded_images = image_processor.load_images_from_directory(
                str(images_dir),
                tags=["test", "sample"]
            )

            assert len(loaded_images) == len(sample_images)
            print(f"Loaded {len(loaded_images)} images from directory")

            # タグが正しく設定されているか確認
            for img in loaded_images:
                assert "test" in img.metadata.get("tags", [])
                assert "sample" in img.metadata.get("tags", [])

            # 3. 埋め込みを生成してベクトルストアに追加
            image_paths = [img.file_path for img in loaded_images]
            embeddings = vision_embeddings.embed_images(image_paths)

            image_ids = vector_store.add_images(loaded_images, embeddings)
            assert len(image_ids) == len(loaded_images)
            print(f"Added {len(image_ids)} images with batch processing")

            # 4. 一覧表示で確認
            images = vector_store.list_images()
            assert len(images) >= len(sample_images)

            # タグでフィルタリングできることを確認（オプション）
            test_tagged_images = [
                img for img in images
                if "test" in img.metadata.get("tags", [])
            ]
            assert len(test_tagged_images) >= len(sample_images)

        finally:
            # クリーンアップ
            vector_store.clear()
