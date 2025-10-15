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


class TestChunk:
    """Chunkクラスのテスト。"""

    def test_create_chunk_with_valid_data(self):
        """正常なChunkインスタンスの作成。"""
        chunk = Chunk(
            content="これはテストチャンクです。",
            chunk_id="doc123_chunk_0001",
            document_id="doc123",
            chunk_index=0,
            start_char=0,
            end_char=13
        )

        assert chunk.content == "これはテストチャンクです。"
        assert chunk.chunk_id == "doc123_chunk_0001"
        assert chunk.document_id == "doc123"
        assert chunk.chunk_index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 13
        assert isinstance(chunk.metadata, dict)

    def test_post_init_adds_metadata(self):
        """__post_init__でメタデータが正しく追加されることを確認。"""
        chunk = Chunk(
            content="テストコンテンツ",
            chunk_id="doc456_chunk_0002",
            document_id="doc456",
            chunk_index=1,
            start_char=100,
            end_char=108
        )

        # チャンク固有のメタデータが追加されている
        assert chunk.metadata['chunk_id'] == "doc456_chunk_0002"
        assert chunk.metadata['document_id'] == "doc456"
        assert chunk.metadata['chunk_index'] == 1
        assert chunk.metadata['start_char'] == 100
        assert chunk.metadata['end_char'] == 108
        assert chunk.metadata['size'] == 8

    def test_size_property_returns_content_length(self):
        """sizeプロパティが正しい文字数を返すことを確認。"""
        content = "これは日本語のテストチャンクです。"
        chunk = Chunk(
            content=content,
            chunk_id="doc789_chunk_0003",
            document_id="doc789",
            chunk_index=2,
            start_char=200,
            end_char=217
        )

        assert chunk.size == len(content)
        assert chunk.size == 17

    def test_metadata_includes_chunk_specific_info(self):
        """metadataにchunk固有の情報が含まれることを確認。"""
        chunk = Chunk(
            content="テスト",
            chunk_id="doc001_chunk_0000",
            document_id="doc001",
            chunk_index=0,
            start_char=0,
            end_char=3
        )

        # すべてのチャンク固有の情報がメタデータに含まれている
        required_keys = ['chunk_id', 'document_id', 'chunk_index', 'start_char', 'end_char', 'size']
        for key in required_keys:
            assert key in chunk.metadata

    def test_custom_metadata_is_preserved(self):
        """カスタムメタデータが保持され、追加のメタデータとマージされることを確認。"""
        custom_metadata = {
            "source_file": "example.txt",
            "author": "テスト太郎"
        }
        chunk = Chunk(
            content="テストチャンク",
            chunk_id="doc002_chunk_0001",
            document_id="doc002",
            chunk_index=1,
            start_char=50,
            end_char=57,
            metadata=custom_metadata.copy()
        )

        # カスタムメタデータが保持されている
        assert chunk.metadata["source_file"] == "example.txt"
        assert chunk.metadata["author"] == "テスト太郎"

        # チャンク固有のメタデータも追加されている
        assert chunk.metadata["chunk_id"] == "doc002_chunk_0001"
        assert chunk.metadata["document_id"] == "doc002"

    def test_empty_content_has_zero_size(self):
        """空のコンテンツのsizeが0であることを確認。"""
        chunk = Chunk(
            content="",
            chunk_id="doc003_chunk_0000",
            document_id="doc003",
            chunk_index=0,
            start_char=0,
            end_char=0
        )

        assert chunk.size == 0
        assert chunk.metadata['size'] == 0


class TestSearchResult:
    """SearchResultクラスのテスト。"""

    def test_create_search_result_with_valid_data(self):
        """正常なSearchResultインスタンスの作成。"""
        chunk = Chunk(
            content="検索結果のテストチャンク",
            chunk_id="doc_search_chunk_0001",
            document_id="doc_search",
            chunk_index=0,
            start_char=0,
            end_char=12
        )

        result = SearchResult(
            chunk=chunk,
            score=0.85,
            document_name="test_document.txt",
            document_source="/path/to/test_document.txt",
            rank=1
        )

        assert result.chunk == chunk
        assert result.score == 0.85
        assert result.document_name == "test_document.txt"
        assert result.document_source == "/path/to/test_document.txt"
        assert result.rank == 1
        assert isinstance(result.metadata, dict)

    def test_score_validation_raises_error_for_negative_score(self):
        """scoreが0未満の場合にValueErrorがraiseされることを確認。"""
        chunk = Chunk(
            content="テストチャンク",
            chunk_id="doc_invalid_chunk_0001",
            document_id="doc_invalid",
            chunk_index=0,
            start_char=0,
            end_char=7
        )

        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            SearchResult(
                chunk=chunk,
                score=-0.1,  # 負のスコア
                document_name="test.txt",
                document_source="/path/to/test.txt"
            )

    def test_score_validation_raises_error_for_score_above_one(self):
        """scoreが1より大きい場合にValueErrorがraiseされることを確認。"""
        chunk = Chunk(
            content="テストチャンク",
            chunk_id="doc_invalid2_chunk_0001",
            document_id="doc_invalid2",
            chunk_index=0,
            start_char=0,
            end_char=7
        )

        with pytest.raises(ValueError, match="Score must be between 0 and 1"):
            SearchResult(
                chunk=chunk,
                score=1.5,  # 1より大きいスコア
                document_name="test.txt",
                document_source="/path/to/test.txt"
            )

    def test_boundary_value_score_zero(self):
        """境界値テスト: score=0が正常に動作することを確認。"""
        chunk = Chunk(
            content="スコア0のチャンク",
            chunk_id="doc_zero_chunk_0001",
            document_id="doc_zero",
            chunk_index=0,
            start_char=0,
            end_char=8
        )

        result = SearchResult(
            chunk=chunk,
            score=0.0,  # 最小値
            document_name="test.txt",
            document_source="/path/to/test.txt",
            rank=5
        )

        assert result.score == 0.0
        assert result.rank == 5

    def test_boundary_value_score_one(self):
        """境界値テスト: score=1が正常に動作することを確認。"""
        chunk = Chunk(
            content="スコア1のチャンク",
            chunk_id="doc_one_chunk_0001",
            document_id="doc_one",
            chunk_index=0,
            start_char=0,
            end_char=8
        )

        result = SearchResult(
            chunk=chunk,
            score=1.0,  # 最大値
            document_name="test.txt",
            document_source="/path/to/test.txt",
            rank=1
        )

        assert result.score == 1.0
        assert result.rank == 1
