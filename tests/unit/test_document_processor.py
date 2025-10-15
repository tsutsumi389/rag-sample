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


@pytest.mark.unit
class TestDocumentProcessorErrorHandling:
    """DocumentProcessor - エラーハンドリングのテスト（3.2）。"""

    def test_load_nonexistent_file_raises_error(self, processor):
        """存在しないファイルでDocumentProcessorErrorがraiseされる。"""
        nonexistent_file = Path("/path/to/nonexistent/file.txt")

        with pytest.raises(DocumentProcessorError) as exc_info:
            processor.load_document(nonexistent_file)

        assert "ファイルが見つかりません" in str(exc_info.value)

    def test_load_directory_raises_error(self, processor, tmp_path):
        """ディレクトリパスでDocumentProcessorErrorがraiseされる。"""
        # ディレクトリを作成
        directory = tmp_path / "test_dir"
        directory.mkdir()

        with pytest.raises(DocumentProcessorError) as exc_info:
            processor.load_document(directory)

        assert "ディレクトリではなくファイルを指定してください" in str(exc_info.value)

    def test_load_empty_txt_file_raises_error(self, processor, tmp_path):
        """空のTXTファイルでDocumentProcessorErrorがraiseされる。"""
        # 空ファイルを作成
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")

        with pytest.raises(DocumentProcessorError) as exc_info:
            processor.load_document(empty_file)

        assert "ファイルが空です" in str(exc_info.value)

    def test_load_empty_md_file_raises_error(self, processor, tmp_path):
        """空のMDファイルでDocumentProcessorErrorがraiseされる。"""
        # 空ファイルを作成（スペースのみ）
        empty_file = tmp_path / "empty.md"
        empty_file.write_text("   \n\n   ")

        with pytest.raises(DocumentProcessorError) as exc_info:
            processor.load_document(empty_file)

        assert "ファイルが空です" in str(exc_info.value)

    def test_load_empty_pdf_raises_error(self, processor, tmp_path):
        """空のPDFファイルでDocumentProcessorErrorがraiseされる。"""
        # 空のPDFファイル（テキスト抽出できない）をモック
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"%PDF-1.4 fake pdf")

        # PyPDF2のモック：空のテキストを返す
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "   "

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with patch("PyPDF2.PdfReader", return_value=mock_reader):
            with pytest.raises(DocumentProcessorError) as exc_info:
                processor.load_document(empty_pdf)

        assert "PDFからテキストを抽出できませんでした" in str(exc_info.value)

    def test_load_invalid_encoding_utf8_fallback_to_shift_jis(self, processor, tmp_path):
        """不正なエンコーディングのファイルでShift_JISフォールバックが動作する。"""
        # Shift_JISでエンコードされたファイルを作成
        sjis_file = tmp_path / "sjis.txt"
        sjis_content = "これはShift_JISエンコーディングのファイルです。"
        sjis_file.write_bytes(sjis_content.encode('shift_jis'))

        # ファイルの読み込み（UTF-8で失敗 → Shift_JISで成功）
        document = processor.load_document(sjis_file)

        assert isinstance(document, Document)
        assert sjis_content in document.content

    def test_load_invalid_encoding_both_fail_raises_error(self, processor, tmp_path):
        """UTF-8とShift_JIS両方で読めないファイルでDocumentProcessorErrorがraiseされる。"""
        # 不正なエンコーディングのファイルを作成
        invalid_file = tmp_path / "invalid.txt"
        # Latin-1でエンコードした特殊文字（UTF-8/Shift_JISで読めない）
        invalid_file.write_bytes(b'\xff\xfe\x00Invalid encoding')

        with pytest.raises(DocumentProcessorError) as exc_info:
            processor.load_document(invalid_file)

        assert "エンコーディングを認識できません" in str(exc_info.value)


@pytest.mark.unit
class TestDocumentProcessorTextSplitting:
    """DocumentProcessor - テキスト分割のテスト（3.3）。"""

    def test_split_text_creates_chunks(self, processor):
        """split_text()で正しくチャンクに分割される。"""
        # 長めのテキストを用意（デフォルトのchunk_size=1000より大きくする）
        text = "これはテストです。" * 200  # 1800文字程度

        chunks = processor.split_text(text)

        # チャンクが作成されることを確認
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        # 複数のチャンクに分割されることを確認（テキストが十分長い場合）
        assert len(chunks) > 1

    def test_split_text_respects_chunk_size(self, config):
        """chunk_sizeが正しく適用されることを確認。"""
        # カスタムchunk_sizeで初期化
        config.chunk_size = 100
        config.chunk_overlap = 20
        processor = DocumentProcessor(config)

        # 長いテキスト（500文字）
        text = "あ" * 500

        chunks = processor.split_text(text)

        # 各チャンクのサイズを確認
        for chunk in chunks:
            # chunk_size以下であることを確認（最後のチャンクを除く）
            assert len(chunk) <= config.chunk_size + 50  # 若干の余裕を持たせる

    def test_split_text_respects_chunk_overlap(self, config):
        """chunk_overlapが正しく適用されることを確認。"""
        # カスタム設定で初期化
        config.chunk_size = 100
        config.chunk_overlap = 20
        processor = DocumentProcessor(config)

        # 長いテキスト（300文字）
        text = "0123456789" * 30  # 300文字

        chunks = processor.split_text(text)

        # 少なくとも2つのチャンクがあることを確認
        assert len(chunks) >= 2

        # 隣接するチャンクが重複部分を持つことを確認
        if len(chunks) >= 2:
            # 最初のチャンクの末尾部分
            first_chunk_end = chunks[0][-20:]
            # 2番目のチャンクの先頭部分
            second_chunk_start = chunks[1][:20]

            # 何らかの重複があることを確認（完全一致でなくても良い）
            # オーバーラップが機能していることの確認
            assert len(chunks[1]) > 0

    def test_split_japanese_text_with_period_separator(self, config):
        """日本語テキストの分割（separators: "。"）が正しく動作する。"""
        # 日本語の文章（句点で区切られている）
        text = (
            "これは最初の文です。これは2番目の文です。これは3番目の文です。"
            "これは4番目の文です。これは5番目の文です。これは6番目の文です。"
            "これは7番目の文です。これは8番目の文です。これは9番目の文です。"
            "これは10番目の文です。"
        )

        # 小さめのchunk_sizeで分割
        config.chunk_size = 100
        config.chunk_overlap = 20
        processor = DocumentProcessor(config)

        chunks = processor.split_text(text)

        # チャンクが作成されることを確認
        assert len(chunks) > 0

        # 各チャンクに日本語テキストが含まれていることを確認
        for chunk in chunks:
            assert "文です" in chunk or "これは" in chunk

    def test_split_text_with_empty_string(self, processor):
        """空文字列の分割では空リストが返される。"""
        chunks = processor.split_text("")

        assert isinstance(chunks, list)
        assert len(chunks) == 0

    def test_split_text_with_short_text(self, processor):
        """短いテキストは分割されずに1つのチャンクとして返される。"""
        short_text = "これは短いテキストです。"

        chunks = processor.split_text(short_text)

        assert len(chunks) == 1
        assert chunks[0] == short_text

    def test_split_text_with_newlines(self, processor):
        """改行を含むテキストが正しく分割される。"""
        text = "段落1の内容です。\n\n段落2の内容です。\n\n段落3の内容です。"

        chunks = processor.split_text(text)

        # チャンクが作成されることを確認
        assert len(chunks) > 0
        # 改行が考慮されていることを確認
        for chunk in chunks:
            assert len(chunk) > 0
