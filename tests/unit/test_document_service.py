"""ドキュメントサービスのテスト"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from src.services.document_service import DocumentService, DocumentServiceError
from src.rag.vector_store import VectorStoreError
from src.rag.document_processor import DocumentProcessorError, UnsupportedFileTypeError
from src.rag.image_processor import ImageProcessorError
from src.rag.vision_embeddings import VisionEmbeddingError
from src.models.document import Document, Chunk, ImageDocument


@pytest.fixture
def mock_config():
    """モック設定"""
    config = Mock()
    config.chroma_persist_directory = "./test_chroma_db"
    config.chunk_size = 1000
    config.chunk_overlap = 200
    config.ollama_base_url = "http://localhost:11434"
    config.ollama_embedding_model = "nomic-embed-text"
    config.ollama_llm_model = "llama3.2-vision"
    return config


@pytest.fixture
def document_service(mock_config):
    """DocumentServiceのフィクスチャ"""
    with patch('src.services.document_service.VectorStore'), \
         patch('src.services.document_service.DocumentProcessor'), \
         patch('src.services.document_service.EmbeddingGenerator'), \
         patch('src.services.document_service.VisionEmbeddings'), \
         patch('src.services.document_service.ImageProcessor'):
        service = DocumentService(mock_config)
        return service


class TestDocumentServiceInit:
    """DocumentServiceの初期化テスト"""

    def test_init_creates_components(self, mock_config):
        """初期化時に必要なコンポーネントが作成される"""
        with patch('src.services.document_service.VectorStore') as mock_vs, \
             patch('src.services.document_service.DocumentProcessor') as mock_dp, \
             patch('src.services.document_service.EmbeddingGenerator') as mock_eg, \
             patch('src.services.document_service.VisionEmbeddings') as mock_ve, \
             patch('src.services.document_service.ImageProcessor') as mock_ip:

            service = DocumentService(mock_config)

            # VectorStoreが2回作成される（documents, images）
            assert mock_vs.call_count == 2
            # その他のコンポーネントが作成される
            assert mock_dp.called
            assert mock_eg.called
            assert mock_ve.called
            assert mock_ip.called

            # VectorStoreが初期化される
            assert service.doc_vector_store.initialize.called
            assert service.img_vector_store.initialize.called


class TestAddFile:
    """add_fileメソッドのテスト"""

    def test_add_file_nonexistent(self, document_service):
        """存在しないファイルの追加はエラー"""
        result = document_service.add_file("/nonexistent/file.txt")

        assert result["success"] is False
        assert "見つかりません" in result["message"]
        assert result["error"] == "FileNotFoundError"

    def test_add_file_directory(self, document_service, tmp_path):
        """ディレクトリの追加は未サポート"""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()

        result = document_service.add_file(str(test_dir))

        assert result["success"] is False
        assert "ディレクトリ" in result["message"]
        assert result["error"] == "DirectoryNotSupported"

    def test_add_file_image(self, document_service, tmp_path):
        """画像ファイルはadd_image_fileにルーティング"""
        image_file = tmp_path / "test.jpg"
        image_file.write_bytes(b"fake image data")

        # add_image_fileをモック
        document_service.add_image_file = Mock(return_value={"success": True})

        result = document_service.add_file(str(image_file))

        # add_image_fileが呼ばれたことを確認
        document_service.add_image_file.assert_called_once()

    def test_add_file_document(self, document_service, tmp_path):
        """テキストファイルはadd_document_fileにルーティング"""
        doc_file = tmp_path / "test.txt"
        doc_file.write_text("test content")

        # add_document_fileをモック
        document_service.add_document_file = Mock(return_value={"success": True})

        result = document_service.add_file(str(doc_file))

        # add_document_fileが呼ばれたことを確認
        document_service.add_document_file.assert_called_once()


class TestListDocuments:
    """list_documentsメソッドのテスト"""

    def test_list_documents_empty(self, document_service):
        """ドキュメントがない場合"""
        document_service.doc_vector_store.list_documents = Mock(return_value=[])
        document_service.img_vector_store.list_images = Mock(return_value=[])

        result = document_service.list_documents()

        assert result["success"] is True
        assert result["total_count"] == 0
        assert len(result["documents"]) == 0
        assert len(result["images"]) == 0
        assert "登録されているドキュメントはありません" in result["message"]

    def test_list_documents_with_data(self, document_service):
        """ドキュメントと画像がある場合"""
        # モックデータ
        mock_docs = [
            {"document_id": "doc1", "document_name": "test1.txt"},
            {"document_id": "doc2", "document_name": "test2.txt"}
        ]
        mock_img = ImageDocument(
            id="img1",
            file_path="/path/to/image.jpg",
            file_name="image.jpg",
            image_type="jpg",
            caption="test image",
            metadata={}
        )

        document_service.doc_vector_store.list_documents = Mock(return_value=mock_docs)
        document_service.img_vector_store.list_images = Mock(return_value=[mock_img])

        result = document_service.list_documents()

        assert result["success"] is True
        assert result["total_count"] == 3
        assert len(result["documents"]) == 2
        assert len(result["images"]) == 1

    def test_list_documents_exclude_images(self, document_service):
        """画像を除外する場合"""
        mock_docs = [{"document_id": "doc1"}]
        document_service.doc_vector_store.list_documents = Mock(return_value=mock_docs)

        result = document_service.list_documents(include_images=False)

        assert result["success"] is True
        assert len(result["documents"]) == 1
        assert len(result["images"]) == 0
        # list_imagesが呼ばれていないことを確認
        document_service.img_vector_store.list_images.assert_not_called()

    def test_list_documents_with_limit(self, document_service):
        """limit指定がある場合"""
        document_service.doc_vector_store.list_documents = Mock(return_value=[])
        document_service.img_vector_store.list_images = Mock(return_value=[])

        result = document_service.list_documents(limit=10)

        # limitが渡されることを確認
        document_service.doc_vector_store.list_documents.assert_called_once_with(limit=10)
        document_service.img_vector_store.list_images.assert_called_once_with(limit=10)


class TestRemoveDocument:
    """remove_documentメソッドのテスト"""

    def test_remove_document_success(self, document_service):
        """ドキュメント削除成功"""
        mock_docs = [
            {"document_id": "doc1", "document_name": "test.txt"}
        ]
        document_service.doc_vector_store.list_documents = Mock(return_value=mock_docs)
        document_service.doc_vector_store.delete = Mock(return_value=5)

        result = document_service.remove_document("doc1", item_type="document")

        assert result["success"] is True
        assert result["item_type"] == "document"
        assert result["deleted_chunks"] == 5
        document_service.doc_vector_store.delete.assert_called_once_with(document_id="doc1")

    def test_remove_image_success(self, document_service):
        """画像削除成功"""
        mock_img = ImageDocument(
            id="img1",
            file_path="/path/to/image.jpg",
            file_name="image.jpg",
            image_type="jpg",
            caption="test",
            metadata={}
        )
        document_service.img_vector_store.get_image_by_id = Mock(return_value=mock_img)
        document_service.img_vector_store.remove_image = Mock(return_value=True)

        result = document_service.remove_document("img1", item_type="image")

        assert result["success"] is True
        assert result["item_type"] == "image"
        document_service.img_vector_store.remove_image.assert_called_once_with("img1")

    def test_remove_auto_detect_document(self, document_service):
        """auto検出でドキュメントを削除"""
        mock_docs = [{"document_id": "doc1", "document_name": "test.txt"}]
        document_service.doc_vector_store.list_documents = Mock(return_value=mock_docs)
        document_service.doc_vector_store.delete = Mock(return_value=3)

        result = document_service.remove_document("doc1", item_type="auto")

        assert result["success"] is True
        assert result["item_type"] == "document"

    def test_remove_auto_detect_image(self, document_service):
        """auto検出で画像を削除（ドキュメントが見つからない場合）"""
        # ドキュメントとしては見つからない
        document_service.doc_vector_store.list_documents = Mock(return_value=[])

        # 画像として見つかる
        mock_img = ImageDocument(
            id="img1",
            file_path="/path/to/image.jpg",
            file_name="image.jpg",
            image_type="jpg",
            caption="test",
            metadata={}
        )
        document_service.img_vector_store.get_image_by_id = Mock(return_value=mock_img)
        document_service.img_vector_store.remove_image = Mock(return_value=True)

        result = document_service.remove_document("img1", item_type="auto")

        assert result["success"] is True
        assert result["item_type"] == "image"

    def test_remove_not_found(self, document_service):
        """存在しないIDの削除"""
        document_service.doc_vector_store.list_documents = Mock(return_value=[])
        document_service.img_vector_store.get_image_by_id = Mock(return_value=None)

        result = document_service.remove_document("nonexistent", item_type="auto")

        assert result["success"] is False
        assert "見つかりませんでした" in result["message"]
        assert result["error"] == "NotFound"


class TestSearchDocuments:
    """search_documentsメソッドのテスト"""

    def test_search_documents_success(self, document_service):
        """ドキュメント検索成功"""
        # モックの検索結果
        mock_chunk = Chunk(
            content="test content",
            chunk_id="chunk1",
            document_id="doc1",
            chunk_index=0,
            start_char=0,
            end_char=100,
            metadata={"document_name": "test.txt", "document_id": "doc1", "chunk_index": 0}
        )
        from src.models.document import SearchResult
        mock_result = SearchResult(
            chunk=mock_chunk,
            score=0.95,
            document_name="test.txt",
            document_source="/path/to/test.txt"
        )

        document_service.embedding_generator.embed_query = Mock(return_value=[0.1] * 768)
        document_service.doc_vector_store.search = Mock(return_value=[mock_result])

        result = document_service.search_documents("test query", top_k=5)

        assert result["success"] is True
        assert result["query"] == "test query"
        assert result["count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "test content"
        assert result["results"][0]["score"] == 0.95


class TestSearchImages:
    """search_imagesメソッドのテスト"""

    def test_search_images_success(self, document_service):
        """画像検索成功"""
        # モックの検索結果
        from src.models.document import SearchResult, Chunk
        mock_chunk = Chunk(
            content="",
            chunk_id="img1",
            document_id="img1",
            chunk_index=0,
            start_char=0,
            end_char=0,
            metadata={"image_type": "jpg", "tags": [], "added_at": "2024-01-01"}
        )
        mock_result = SearchResult(
            chunk=mock_chunk,
            score=0.9,
            document_name="test.jpg",
            document_source="/path/to/test.jpg",
            rank=1,
            result_type="image",
            image_path=Path("/path/to/test.jpg"),
            caption="test image",
            metadata={"image_type": "jpg", "tags": [], "added_at": "2024-01-01"}
        )

        document_service.embedding_generator.embed_query = Mock(return_value=[0.1] * 768)
        document_service.img_vector_store.search_images = Mock(return_value=[mock_result])

        result = document_service.search_images("cat photo", top_k=3)

        assert result["success"] is True
        assert result["query"] == "cat photo"
        assert result["count"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["file_name"] == "test.jpg"
        assert result["results"][0]["score"] == 0.9

    def test_search_images_vector_store_error(self, document_service):
        """画像検索でVectorStoreError発生"""
        document_service.embedding_generator.embed_query = Mock(return_value=[0.1] * 768)
        document_service.img_vector_store.search_images = Mock(
            side_effect=VectorStoreError("No images found")
        )

        result = document_service.search_images("test query")

        assert result["success"] is False
        assert result["error"] == "VectorStoreError"
        assert "hint" in result
