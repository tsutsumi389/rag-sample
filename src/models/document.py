"""RAGアプリケーションのデータモデル。

このモジュールはアプリケーション全体で使用される中核的なデータ構造を定義します:
- Document: メタデータを含むソースドキュメントを表現
- Chunk: メタデータを含む分割されたテキストチャンクを表現
- SearchResult: 類似度スコアを含む検索結果を表現
- ChatMessage: 会話履歴内のチャットメッセージを表現
- ImageDocument: 画像ドキュメントを表現（マルチモーダルRAG用）
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

__all__ = [
    'Document',
    'Chunk',
    'SearchResult',
    'ChatMessage',
    'ChatHistory',
    'ImageDocument',
]


@dataclass
class Document:
    """メタデータを含むソースドキュメントを表現する。

    Attributes:
        file_path: ソースファイルへの絶対パス
        name: ドキュメント名（通常はファイル名）
        content: ドキュメントの全文テキストコンテンツ
        doc_type: ファイルタイプ/拡張子（例: 'txt', 'pdf', 'md'）
        source: ソース識別子（通常はファイルパス文字列）
        timestamp: 作成/追加タイムスタンプ
        metadata: 追加のメタデータ辞書
        document_type: ドキュメントのタイプ（'text' または 'image'）
        image_path: 画像の場合のファイルパス（テキストの場合はNone）
    """
    file_path: Path
    name: str
    content: str
    doc_type: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    document_type: str = 'text'  # 'text' または 'image'
    image_path: Path | None = None

    @property
    def size(self) -> int:
        """ドキュメントコンテンツの文字数を返す。"""
        return len(self.content)

    def __post_init__(self):
        """file_pathとimage_pathがPathオブジェクトであることを保証する。"""
        if not isinstance(self.file_path, Path):
            self.file_path = Path(self.file_path)
        if self.image_path and not isinstance(self.image_path, Path):
            self.image_path = Path(self.image_path)


@dataclass
class Chunk:
    """メタデータを含む分割されたテキストチャンクを表現する。

    Attributes:
        content: チャンクのテキストコンテンツ
        chunk_id: チャンクの一意識別子
        document_id: 親ドキュメントへの参照
        chunk_index: ドキュメント内のこのチャンクのインデックス（0始まり）
        start_char: 元のドキュメント内の開始文字位置
        end_char: 元のドキュメント内の終了文字位置
        metadata: 親ドキュメントのメタデータとチャンク固有の情報
    """
    content: str
    chunk_id: str
    document_id: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        """チャンクコンテンツの文字数を返す。"""
        return len(self.content)

    def __post_init__(self):
        """チャンク固有のメタデータを追加する。"""
        self.metadata.update({
            'chunk_id': self.chunk_id,
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'size': self.size,
        })


@dataclass
class SearchResult:
    """類似度スコアを含む検索結果を表現する。

    Attributes:
        chunk: マッチしたテキストチャンク
        score: 類似度スコア（通常はコサイン類似度、0-1の範囲）
        document_name: ソースドキュメントの名前
        document_source: ドキュメントのソースパス/識別子
        rank: 検索結果内のランキング位置（1始まり）
        metadata: チャンクからの追加メタデータ
        result_type: 検索結果のタイプ（'text' または 'image'）
        image_path: 画像検索結果の場合のファイルパス
        caption: 画像検索結果の場合の説明文
    """
    chunk: Chunk
    score: float
    document_name: str
    document_source: str
    rank: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    result_type: str = 'text'  # 'text' または 'image'
    image_path: Path | None = None
    caption: str | None = None

    def __post_init__(self):
        """スコアが有効な範囲内であることを保証する。"""
        if not 0 <= self.score <= 1:
            raise ValueError(f"Score must be between 0 and 1, got {self.score}")


@dataclass
class ChatMessage:
    """会話履歴内のチャットメッセージを表現する。

    Attributes:
        role: メッセージの役割（'user', 'assistant', または 'system'）
        content: メッセージコンテンツのテキスト
        timestamp: メッセージのタイムスタンプ
        metadata: 追加のメタデータ（例: 使用モデル、トークン数、コンテキスト）
    """
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """役割を検証する。"""
        valid_roles = {'user', 'assistant', 'system'}
        if self.role not in valid_roles:
            raise ValueError(
                f"Role must be one of {valid_roles}, got '{self.role}'"
            )

    def to_dict(self) -> dict[str, str]:
        """LLM API用の辞書形式に変換する。

        Returns:
            'role'と'content'キーを含む辞書
        """
        return {
            'role': self.role,
            'content': self.content,
        }


@dataclass
class ChatHistory:
    """チャットモード用の会話履歴を管理する。

    Attributes:
        messages: チャットメッセージのリスト
        max_messages: 保持する最大メッセージ数（None = 無制限）
    """
    messages: list[ChatMessage] = field(default_factory=list)
    max_messages: int | None = None

    def add_message(self, role: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """履歴に新しいメッセージを追加する。

        Args:
            role: メッセージの役割（'user', 'assistant', または 'system'）
            content: メッセージコンテンツ
            metadata: オプションのメタデータ辞書
        """
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)

        # max_messagesが設定されている場合は履歴を削減
        if self.max_messages and len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def to_dicts(self) -> list[dict[str, str]]:
        """すべてのメッセージをLLM API用の辞書形式に変換する。

        Returns:
            'role'と'content'キーを含む辞書のリスト
        """
        return [msg.to_dict() for msg in self.messages]

    def clear(self) -> None:
        """履歴からすべてのメッセージをクリアする。"""
        self.messages.clear()

    def __len__(self) -> int:
        """履歴内のメッセージ数を返す。"""
        return len(self.messages)


@dataclass
class ImageDocument:
    """画像ドキュメントを表現する（マルチモーダルRAG用）。

    Attributes:
        id: 画像の一意識別子
        file_path: 画像ファイルへの絶対パス
        file_name: 画像ファイル名
        image_type: 画像フォーマット（例: 'jpg', 'png', 'gif', 'webp'）
        caption: ビジョンモデルが生成した画像の説明文
        metadata: 追加のメタデータ辞書（タグ、元画像サイズなど）
        created_at: 作成/追加タイムスタンプ
        image_data: Base64エンコードされた画像データ（オプション）
    """
    id: str
    file_path: Path
    file_name: str
    image_type: str
    caption: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    image_data: str | None = None

    def __post_init__(self):
        """file_pathがPathオブジェクトであることを保証する。"""
        if not isinstance(self.file_path, Path):
            self.file_path = Path(self.file_path)

    @property
    def size_mb(self) -> float:
        """画像ファイルのサイズをMB単位で返す。"""
        if self.file_path.exists():
            return self.file_path.stat().st_size / (1024 * 1024)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """辞書形式に変換する（シリアライゼーション用）。

        Returns:
            画像ドキュメントの情報を含む辞書
        """
        return {
            'id': self.id,
            'file_path': str(self.file_path),
            'file_name': self.file_name,
            'image_type': self.image_type,
            'caption': self.caption,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'image_data': self.image_data,
        }
