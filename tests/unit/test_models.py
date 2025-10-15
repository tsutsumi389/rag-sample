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


class TestChatMessage:
    """ChatMessageクラスのテスト。"""

    def test_create_chat_message_with_user_role(self):
        """正常なChatMessageインスタンスの作成（user）。"""
        message = ChatMessage(
            role="user",
            content="こんにちは、これはテストメッセージです。"
        )

        assert message.role == "user"
        assert message.content == "こんにちは、これはテストメッセージです。"
        assert isinstance(message.timestamp, datetime)
        assert isinstance(message.metadata, dict)
        assert message.metadata == {}

    def test_create_chat_message_with_assistant_role(self):
        """正常なChatMessageインスタンスの作成（assistant）。"""
        message = ChatMessage(
            role="assistant",
            content="お答えします。これは回答メッセージです。"
        )

        assert message.role == "assistant"
        assert message.content == "お答えします。これは回答メッセージです。"
        assert isinstance(message.timestamp, datetime)

    def test_create_chat_message_with_system_role(self):
        """正常なChatMessageインスタンスの作成（system）。"""
        message = ChatMessage(
            role="system",
            content="システムメッセージです。"
        )

        assert message.role == "system"
        assert message.content == "システムメッセージです。"
        assert isinstance(message.timestamp, datetime)

    def test_invalid_role_raises_value_error(self):
        """無効なroleでValueErrorがraiseされることを確認。"""
        with pytest.raises(ValueError, match="Role must be one of"):
            ChatMessage(
                role="invalid_role",
                content="これは無効な役割のメッセージです。"
            )

    def test_invalid_role_error_message_includes_valid_roles(self):
        """無効なroleのエラーメッセージに有効な役割が含まれることを確認。"""
        with pytest.raises(ValueError, match="'invalid_role'"):
            ChatMessage(
                role="invalid_role",
                content="テストメッセージ"
            )

    def test_to_dict_returns_correct_format(self):
        """to_dict()メソッドが正しい辞書を返すことを確認。"""
        message = ChatMessage(
            role="user",
            content="テストメッセージです。",
            metadata={"model": "test-model"}
        )

        result = message.to_dict()

        assert isinstance(result, dict)
        assert result == {
            'role': 'user',
            'content': 'テストメッセージです。'
        }
        # metadataやtimestampは含まれない
        assert 'metadata' not in result
        assert 'timestamp' not in result

    def test_to_dict_with_all_roles(self):
        """to_dict()が全ての有効な役割で機能することを確認。"""
        roles = ['user', 'assistant', 'system']

        for role in roles:
            message = ChatMessage(
                role=role,
                content=f"{role}のメッセージ"
            )
            result = message.to_dict()

            assert result['role'] == role
            assert result['content'] == f"{role}のメッセージ"

    def test_custom_metadata_is_preserved(self):
        """カスタムメタデータが保持されることを確認。"""
        custom_metadata = {
            "model": "llama3.2",
            "tokens": 42,
            "context": ["doc1", "doc2"]
        }
        message = ChatMessage(
            role="assistant",
            content="メタデータ付きメッセージ",
            metadata=custom_metadata
        )

        assert message.metadata == custom_metadata
        assert message.metadata["model"] == "llama3.2"
        assert message.metadata["tokens"] == 42

    def test_timestamp_is_automatically_set(self):
        """timestampが自動設定されることを確認。"""
        before = datetime.now()
        message = ChatMessage(
            role="user",
            content="タイムスタンプテスト"
        )
        after = datetime.now()

        assert before <= message.timestamp <= after
        assert isinstance(message.timestamp, datetime)

    def test_custom_timestamp_is_preserved(self):
        """カスタムタイムスタンプが保持されることを確認。"""
        custom_timestamp = datetime(2024, 6, 15, 10, 30, 0)
        message = ChatMessage(
            role="user",
            content="カスタムタイムスタンプテスト",
            timestamp=custom_timestamp
        )

        assert message.timestamp == custom_timestamp

    def test_empty_content_is_allowed(self):
        """空のコンテンツが許可されることを確認。"""
        message = ChatMessage(
            role="user",
            content=""
        )

        assert message.content == ""
        assert message.role == "user"


class TestChatHistory:
    """ChatHistoryクラスのテスト。"""

    def test_create_chat_history_with_default_values(self):
        """正常なChatHistoryインスタンスの作成。"""
        history = ChatHistory()

        assert isinstance(history.messages, list)
        assert len(history.messages) == 0
        assert history.max_messages is None

    def test_add_message_adds_to_history(self):
        """add_message()でメッセージが追加されることを確認。"""
        history = ChatHistory()

        history.add_message(role="user", content="こんにちは")
        history.add_message(role="assistant", content="こんにちは！何をお手伝いしましょうか？")

        assert len(history.messages) == 2
        assert history.messages[0].role == "user"
        assert history.messages[0].content == "こんにちは"
        assert history.messages[1].role == "assistant"
        assert history.messages[1].content == "こんにちは！何をお手伝いしましょうか？"

    def test_max_messages_limits_history(self):
        """max_messagesによる履歴制限が機能することを確認。"""
        history = ChatHistory(max_messages=3)

        # 5つのメッセージを追加
        history.add_message(role="user", content="メッセージ1")
        history.add_message(role="assistant", content="メッセージ2")
        history.add_message(role="user", content="メッセージ3")
        history.add_message(role="assistant", content="メッセージ4")
        history.add_message(role="user", content="メッセージ5")

        # 最大3件のみ保持される（最新のメッセージ）
        assert len(history.messages) == 3
        assert history.messages[0].content == "メッセージ3"
        assert history.messages[1].content == "メッセージ4"
        assert history.messages[2].content == "メッセージ5"

    def test_to_dicts_converts_all_messages(self):
        """to_dicts()で全メッセージが辞書リストに変換されることを確認。"""
        history = ChatHistory()

        history.add_message(role="user", content="質問です")
        history.add_message(role="assistant", content="回答です")
        history.add_message(role="system", content="システムメッセージ")

        result = history.to_dicts()

        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == {'role': 'user', 'content': '質問です'}
        assert result[1] == {'role': 'assistant', 'content': '回答です'}
        assert result[2] == {'role': 'system', 'content': 'システムメッセージ'}

    def test_clear_removes_all_messages(self):
        """clear()で履歴がクリアされることを確認。"""
        history = ChatHistory()

        # メッセージを追加
        history.add_message(role="user", content="テスト1")
        history.add_message(role="assistant", content="テスト2")
        assert len(history.messages) == 2

        # クリア
        history.clear()

        assert len(history.messages) == 0
        assert history.messages == []

    def test_len_returns_message_count(self):
        """__len__でメッセージ数が正しく返されることを確認。"""
        history = ChatHistory()

        assert len(history) == 0

        history.add_message(role="user", content="メッセージ1")
        assert len(history) == 1

        history.add_message(role="assistant", content="メッセージ2")
        assert len(history) == 2

        history.add_message(role="user", content="メッセージ3")
        assert len(history) == 3

    def test_add_message_with_metadata(self):
        """add_message()でメタデータ付きメッセージが追加されることを確認。"""
        history = ChatHistory()
        metadata = {"model": "llama3.2", "tokens": 100}

        history.add_message(
            role="assistant",
            content="メタデータ付き回答",
            metadata=metadata
        )

        assert len(history.messages) == 1
        assert history.messages[0].metadata == metadata
        assert history.messages[0].metadata["model"] == "llama3.2"

    def test_add_message_without_metadata(self):
        """add_message()でメタデータなしでメッセージが追加されることを確認。"""
        history = ChatHistory()

        history.add_message(role="user", content="メタデータなし")

        assert len(history.messages) == 1
        assert history.messages[0].metadata == {}

    def test_max_messages_with_exact_limit(self):
        """max_messagesちょうどの数のメッセージが正しく保持されることを確認。"""
        history = ChatHistory(max_messages=2)

        history.add_message(role="user", content="メッセージ1")
        history.add_message(role="assistant", content="メッセージ2")

        # ちょうど2件なので全て保持される
        assert len(history.messages) == 2
        assert history.messages[0].content == "メッセージ1"
        assert history.messages[1].content == "メッセージ2"

        # 3件目を追加すると古いものが削除される
        history.add_message(role="user", content="メッセージ3")
        assert len(history.messages) == 2
        assert history.messages[0].content == "メッセージ2"
        assert history.messages[1].content == "メッセージ3"

    def test_to_dicts_returns_empty_list_for_empty_history(self):
        """空の履歴でto_dicts()が空リストを返すことを確認。"""
        history = ChatHistory()

        result = history.to_dicts()

        assert isinstance(result, list)
        assert len(result) == 0
        assert result == []

    def test_add_message_preserves_timestamp(self):
        """add_message()でタイムスタンプが保持されることを確認。"""
        history = ChatHistory()

        before = datetime.now()
        history.add_message(role="user", content="タイムスタンプテスト")
        after = datetime.now()

        assert len(history.messages) == 1
        assert before <= history.messages[0].timestamp <= after

    def test_initialize_with_max_messages(self):
        """max_messagesを指定してインスタンス化できることを確認。"""
        history = ChatHistory(max_messages=5)

        assert history.max_messages == 5
        assert len(history.messages) == 0

    def test_max_messages_with_large_number(self):
        """大きなmax_messages値で正しく動作することを確認。"""
        history = ChatHistory(max_messages=100)

        # 10件のメッセージを追加
        for i in range(10):
            history.add_message(role="user", content=f"メッセージ{i+1}")

        # max_messages(100)より少ないので全て保持される
        assert len(history.messages) == 10
        assert history.messages[0].content == "メッセージ1"
        assert history.messages[9].content == "メッセージ10"
