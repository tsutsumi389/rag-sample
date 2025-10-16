"""ドキュメント処理モジュール

このモジュールはファイルの読み込み、テキスト分割、メタデータ付与を行います。
複数のファイル形式（TXT, MD, PDF）に対応し、LangChainのテキスト分割機能を使用します。
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import hashlib

from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..models.document import Document, Chunk
from ..utils.config import Config


class DocumentProcessorError(Exception):
    """ドキュメント処理エラー"""
    pass


class UnsupportedFileTypeError(DocumentProcessorError):
    """サポートされていないファイルタイプエラー"""
    pass


class DocumentProcessor:
    """ドキュメントの読み込みと処理を行うクラス

    ファイルの読み込み、テキスト分割、チャンク作成などの機能を提供します。

    Attributes:
        config: アプリケーション設定
        text_splitter: テキスト分割器
        supported_extensions: サポートされているファイル拡張子
    """

    # サポートされているファイル拡張子
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.pdf'}

    def __init__(self, config: Config):
        """ドキュメントプロセッサーの初期化

        Args:
            config: アプリケーション設定
        """
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def is_supported_file(self, file_path: Path) -> bool:
        """ファイルがサポートされている形式かチェック

        Args:
            file_path: チェックするファイルパス

        Returns:
            bool: サポートされている場合True
        """
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def load_document(self, file_path: str | Path) -> Document:
        """ファイルを読み込んでDocumentオブジェクトを作成

        Args:
            file_path: 読み込むファイルのパス

        Returns:
            Document: 読み込んだドキュメント

        Raises:
            DocumentProcessorError: ファイルが存在しない、または読み込みに失敗した場合
            UnsupportedFileTypeError: サポートされていないファイル形式の場合
        """
        # Path オブジェクトに変換
        path = Path(file_path) if isinstance(file_path, str) else file_path

        # ファイルの存在確認
        if not path.exists():
            raise DocumentProcessorError(f"ファイルが見つかりません: {path}")

        if not path.is_file():
            raise DocumentProcessorError(f"ディレクトリではなくファイルを指定してください: {path}")

        # ファイルタイプのチェック
        if not self.is_supported_file(path):
            raise UnsupportedFileTypeError(
                f"サポートされていないファイル形式です: {path.suffix}\n"
                f"サポート形式: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        # ファイル形式に応じて読み込み
        doc_type = path.suffix.lower().lstrip('.')

        if doc_type in ['txt', 'md']:
            content = self._load_text_file(path)
        elif doc_type == 'pdf':
            content = self._load_pdf_file(path)
        else:
            raise UnsupportedFileTypeError(f"未実装のファイル形式: {doc_type}")

        # Documentオブジェクトの作成
        document = Document(
            file_path=path.resolve(),
            name=path.name,
            content=content,
            doc_type=doc_type,
            source=str(path.resolve()),
            timestamp=datetime.now(),
            metadata={
                'file_size': path.stat().st_size,
                'file_modified': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                'encoding': 'utf-8' if doc_type in ['txt', 'md'] else 'binary',
            }
        )

        return document

    def _load_text_file(self, file_path: Path) -> str:
        """テキストファイル（TXT, MD）を読み込む

        Args:
            file_path: 読み込むファイルのパス

        Returns:
            str: ファイルの内容

        Raises:
            DocumentProcessorError: 読み込みに失敗した場合
        """
        try:
            # UTF-8で読み込みを試みる
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # UTF-8で読めない場合、他のエンコーディングを試す
            try:
                with open(file_path, 'r', encoding='shift_jis') as f:
                    content = f.read()
            except UnicodeDecodeError:
                raise DocumentProcessorError(
                    f"ファイルのエンコーディングを認識できません: {file_path}"
                )
        except Exception as e:
            raise DocumentProcessorError(
                f"ファイルの読み込みに失敗しました: {file_path}\n{str(e)}"
            )

        # 空ファイルのチェック
        if not content.strip():
            raise DocumentProcessorError(f"ファイルが空です: {file_path}")

        return content

    def _load_pdf_file(self, file_path: Path) -> str:
        """PDFファイルを読み込む

        Args:
            file_path: 読み込むPDFファイルのパス

        Returns:
            str: PDFから抽出されたテキスト

        Raises:
            DocumentProcessorError: 読み込みに失敗した場合
        """
        try:
            # PyPDF2を使用してPDFを読み込む
            import PyPDF2

            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)

                # 全ページのテキストを抽出
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(page_text)

                content = '\n\n'.join(text_parts)

        except ImportError:
            raise DocumentProcessorError(
                "PyPDF2がインストールされていません。\n"
                "PDFファイルを読み込むには 'pip install PyPDF2' を実行してください。"
            )
        except Exception as e:
            raise DocumentProcessorError(
                f"PDFファイルの読み込みに失敗しました: {file_path}\n{str(e)}"
            )

        # 空ファイルのチェック
        if not content.strip():
            raise DocumentProcessorError(
                f"PDFからテキストを抽出できませんでした: {file_path}"
            )

        return content

    def split_text(self, text: str) -> list[str]:
        """テキストをチャンクに分割

        Args:
            text: 分割するテキスト

        Returns:
            list[str]: 分割されたテキストのリスト
        """
        return self.text_splitter.split_text(text)

    def create_chunks(
        self,
        document: Document,
        document_id: Optional[str] = None
    ) -> list[Chunk]:
        """ドキュメントからチャンクを作成

        Args:
            document: チャンク化するドキュメント
            document_id: ドキュメントID（省略時は自動生成）

        Returns:
            list[Chunk]: 作成されたチャンクのリスト
        """
        # ドキュメントIDの生成（省略時）
        if document_id is None:
            document_id = self._generate_document_id(document)

        # テキストを分割
        text_chunks = self.split_text(document.content)

        # Chunkオブジェクトのリストを作成
        chunks: list[Chunk] = []
        current_position = 0

        for chunk_index, chunk_text in enumerate(text_chunks):
            # チャンクIDの生成
            chunk_id = self._generate_chunk_id(document_id, chunk_index)

            # 元のドキュメント内での位置を特定
            start_char = document.content.find(chunk_text, current_position)
            if start_char == -1:
                # 完全一致が見つからない場合は現在位置を使用
                start_char = current_position
            end_char = start_char + len(chunk_text)

            # 次の検索のために位置を更新
            current_position = end_char

            # メタデータの準備
            chunk_metadata = {
                **document.metadata,  # ドキュメントのメタデータを継承
                'document_name': document.name,
                'document_source': document.source,
                'document_type': document.doc_type,
                'timestamp': document.timestamp.isoformat(),
            }

            # Chunkオブジェクトの作成
            chunk = Chunk(
                content=chunk_text,
                chunk_id=chunk_id,
                document_id=document_id,
                chunk_index=chunk_index,
                start_char=start_char,
                end_char=end_char,
                metadata=chunk_metadata,
            )

            chunks.append(chunk)

        return chunks

    def process_document(
        self,
        file_path: str | Path,
        document_id: Optional[str] = None
    ) -> tuple[Document, list[Chunk]]:
        """ドキュメントの読み込みとチャンク化を一度に実行

        Args:
            file_path: 処理するファイルのパス
            document_id: ドキュメントID（省略時は自動生成）

        Returns:
            tuple[Document, list[Chunk]]: ドキュメントとチャンクのタプル

        Raises:
            DocumentProcessorError: 処理に失敗した場合
        """
        # ドキュメントの読み込み
        document = self.load_document(file_path)

        # チャンクの作成
        chunks = self.create_chunks(document, document_id)

        return document, chunks

    @staticmethod
    def _generate_document_id(document: Document) -> str:
        """ドキュメントから一意のIDを生成

        Args:
            document: ドキュメント

        Returns:
            str: 生成されたドキュメントID（SHA-256ハッシュの先頭16文字）
        """
        # ファイルパスとタイムスタンプからハッシュを生成
        hash_input = f"{document.source}_{document.timestamp.isoformat()}"
        hash_object = hashlib.sha256(hash_input.encode())
        return hash_object.hexdigest()[:16]

    @staticmethod
    def _generate_chunk_id(document_id: str, chunk_index: int) -> str:
        """チャンクIDを生成

        Args:
            document_id: ドキュメントID
            chunk_index: チャンクインデックス

        Returns:
            str: 生成されたチャンクID
        """
        return f"{document_id}_chunk_{chunk_index:04d}"
