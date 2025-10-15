"""ドキュメント処理のユニットテスト。

このモジュールはsrc/rag/document_processor.pyで定義されたDocumentProcessorクラスのテストを提供します。
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import io

from src.rag.document_processor import (
    DocumentProcessor,
    DocumentProcessorError,
    UnsupportedFileTypeError,
)
from src.models.document import Document
from src.utils.config import Config


@pytest.fixture
def config():
    """テスト用のConfig fixture。"""
    return Config(env_file=None)


@pytest.fixture
def processor(config):
    """テスト用のDocumentProcessor fixture。"""
    return DocumentProcessor(config)


@pytest.fixture
def fixtures_dir():
    """テストフィクスチャディレクトリのパス。"""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_txt_file(fixtures_dir):
    """サンプルTXTファイルのパス。"""
    return fixtures_dir / "sample.txt"


@pytest.fixture
def sample_md_file(fixtures_dir):
    """サンプルMDファイルのパス。"""
    return fixtures_dir / "sample.md"


@pytest.mark.unit
class TestDocumentProcessorFileSupport:
    """DocumentProcessor - ファイル形式サポートのテスト（3.1）。"""

    def test_is_supported_file_txt(self, processor):
        """is_supported_file()でTXTファイルが正しく判定される。"""
        txt_path = Path("/path/to/file.txt")
        assert processor.is_supported_file(txt_path) is True

    def test_is_supported_file_md(self, processor):
        """is_supported_file()でMDファイルが正しく判定される。"""
        md_path = Path("/path/to/file.md")
        assert processor.is_supported_file(md_path) is True

    def test_is_supported_file_pdf(self, processor):
        """is_supported_file()でPDFファイルが正しく判定される。"""
        pdf_path = Path("/path/to/file.pdf")
        assert processor.is_supported_file(pdf_path) is True

    def test_is_supported_file_case_insensitive(self, processor):
        """is_supported_file()で大文字拡張子も正しく判定される。"""
        assert processor.is_supported_file(Path("/path/to/FILE.TXT")) is True
        assert processor.is_supported_file(Path("/path/to/FILE.MD")) is True
        assert processor.is_supported_file(Path("/path/to/FILE.PDF")) is True

    def test_is_supported_file_unsupported(self, processor):
        """is_supported_file()でサポート外の形式がFalseを返す。"""
        assert processor.is_supported_file(Path("/path/to/file.docx")) is False
        assert processor.is_supported_file(Path("/path/to/file.jpg")) is False
        assert processor.is_supported_file(Path("/path/to/file.csv")) is False

    def test_load_txt_file(self, processor, sample_txt_file):
        """TXTファイルの読み込みが正常に動作する。"""
        document = processor.load_document(sample_txt_file)

        # 基本的なアサーション
        assert isinstance(document, Document)
        assert document.name == "sample.txt"
        assert document.doc_type == "txt"
        assert document.file_path == sample_txt_file.resolve()
        assert "テスト用のサンプルテキストファイル" in document.content
        assert len(document.content) > 0

        # メタデータの確認
        assert "file_size" in document.metadata
        assert "file_modified" in document.metadata
        assert document.metadata["encoding"] == "utf-8"

    def test_load_md_file(self, processor, sample_md_file):
        """MDファイルの読み込みが正常に動作する。"""
        document = processor.load_document(sample_md_file)

        # 基本的なアサーション
        assert isinstance(document, Document)
        assert document.name == "sample.md"
        assert document.doc_type == "md"
        assert document.file_path == sample_md_file.resolve()
        assert "# サンプルMarkdown" in document.content
        assert "## セクション1" in document.content
        assert len(document.content) > 0

        # メタデータの確認
        assert "file_size" in document.metadata
        assert document.metadata["encoding"] == "utf-8"

    def test_load_pdf_file(self, processor, tmp_path):
        """PDFファイルの読み込みが正常に動作する（PyPDF2を使用）。"""
        # 一時的なPDFファイルを作成
        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake pdf content")

        # PyPDF2のモックを使用
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "これはPDFのテキストです。"

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page, mock_page]

        # PyPDF2.PdfReaderをモック
        with patch("PyPDF2.PdfReader", return_value=mock_reader):
            document = processor.load_document(pdf_path)

        # アサーション
        assert isinstance(document, Document)
        assert document.name == "test.pdf"
        assert document.doc_type == "pdf"
        assert "これはPDFのテキストです。" in document.content
        assert document.metadata["encoding"] == "binary"

    def test_load_unsupported_file_type_raises_error(self, processor, tmp_path):
        """サポート外のファイル形式でUnsupportedFileTypeErrorがraiseされる。"""
        # サポート外の拡張子のファイルを作成
        unsupported_file = tmp_path / "test.docx"
        unsupported_file.write_text("test content")

        with pytest.raises(UnsupportedFileTypeError) as exc_info:
            processor.load_document(unsupported_file)

        assert "サポートされていないファイル形式" in str(exc_info.value)
        assert ".docx" in str(exc_info.value)

    def test_load_file_with_string_path(self, processor, sample_txt_file):
        """文字列パスでもファイルが読み込める。"""
        # 文字列パスを渡す
        document = processor.load_document(str(sample_txt_file))

        assert isinstance(document, Document)
        assert document.name == "sample.txt"
        assert isinstance(document.file_path, Path)
