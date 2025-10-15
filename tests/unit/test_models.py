"""データモデルのユニットテスト。

このモジュールはsrc/models/document.pyで定義されたデータモデルのテストを提供します。
"""

import pytest
from datetime import datetime
from pathlib import Path

from src.models.document import Document, Chunk, SearchResult, ChatMessage, ChatHistory


class TestDocument:
    """Documentクラスのテスト。"""

    def test_create_document_with_valid_data(self):
        """正常なDocumentインスタンスの作成。"""
        doc = Document(
            file_path=Path("/path/to/file.txt"),
            name="file.txt",
            content="これはテストコンテンツです。",
            doc_type="txt",
            source="/path/to/file.txt"
        )

        assert doc.file_path == Path("/path/to/file.txt")
        assert doc.name == "file.txt"
        assert doc.content == "これはテストコンテンツです。"
        assert doc.doc_type == "txt"
        assert doc.source == "/path/to/file.txt"
        assert isinstance(doc.timestamp, datetime)
        assert isinstance(doc.metadata, dict)

    def test_file_path_converts_to_path_object(self):
        """file_pathが自動的にPathオブジェクトに変換されることを確認。"""
        # 文字列を渡してもPathに変換される
        doc = Document(
            file_path="/path/to/file.txt",  # str型
            name="file.txt",
            content="テスト",
            doc_type="txt",
            source="/path/to/file.txt"
        )

        assert isinstance(doc.file_path, Path)
        assert doc.file_path == Path("/path/to/file.txt")

    def test_size_property_returns_content_length(self):
        """sizeプロパティが正しい文字数を返すことを確認。"""
        content = "これはテストです。日本語も含まれます。"
        doc = Document(
            file_path=Path("/path/to/file.txt"),
            name="file.txt",
            content=content,
            doc_type="txt",
            source="/path/to/file.txt"
        )

        assert doc.size == len(content)
        assert doc.size == 19  # 具体的な文字数を確認

    def test_metadata_defaults_to_empty_dict(self):
        """metadataのデフォルト値が空辞書であることを確認。"""
        doc = Document(
            file_path=Path("/path/to/file.txt"),
            name="file.txt",
            content="テスト",
            doc_type="txt",
            source="/path/to/file.txt"
        )

        assert doc.metadata == {}
        assert isinstance(doc.metadata, dict)

    def test_timestamp_is_automatically_set(self):
        """timestampが自動設定されることを確認。"""
        before = datetime.now()
        doc = Document(
            file_path=Path("/path/to/file.txt"),
            name="file.txt",
            content="テスト",
            doc_type="txt",
            source="/path/to/file.txt"
        )
        after = datetime.now()

        assert before <= doc.timestamp <= after
        assert isinstance(doc.timestamp, datetime)

    def test_custom_metadata_is_preserved(self):
        """カスタムメタデータが保持されることを確認。"""
        custom_metadata = {
            "author": "テスト太郎",
            "tags": ["test", "sample"],
            "version": 1
        }
        doc = Document(
            file_path=Path("/path/to/file.txt"),
            name="file.txt",
            content="テスト",
            doc_type="txt",
            source="/path/to/file.txt",
            metadata=custom_metadata
        )

        assert doc.metadata == custom_metadata
        assert doc.metadata["author"] == "テスト太郎"
        assert doc.metadata["tags"] == ["test", "sample"]

    def test_custom_timestamp_is_preserved(self):
        """カスタムタイムスタンプが保持されることを確認。"""
        custom_timestamp = datetime(2024, 1, 1, 12, 0, 0)
        doc = Document(
            file_path=Path("/path/to/file.txt"),
            name="file.txt",
            content="テスト",
            doc_type="txt",
            source="/path/to/file.txt",
            timestamp=custom_timestamp
        )

        assert doc.timestamp == custom_timestamp

    def test_empty_content_has_zero_size(self):
        """空のコンテンツのsizeが0であることを確認。"""
        doc = Document(
            file_path=Path("/path/to/file.txt"),
            name="file.txt",
            content="",
            doc_type="txt",
            source="/path/to/file.txt"
        )

        assert doc.size == 0
