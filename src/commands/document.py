"""ドキュメント管理コマンドモジュール

このモジュールはドキュメントの追加、削除、一覧表示、クリアなどの
ドキュメント管理コマンドを提供します。
"""

import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..rag.vector_store import VectorStore, VectorStoreError
from ..rag.document_processor import (
    DocumentProcessor,
    DocumentProcessorError,
    UnsupportedFileTypeError,
)
from ..rag.embeddings import EmbeddingGenerator, EmbeddingError
from ..utils.config import get_config

logger = logging.getLogger(__name__)
console = Console()


class DocumentCommandError(Exception):
    """ドキュメントコマンドのエラー"""
    pass


@click.command('add')
@click.argument('file_path', type=click.Path(exists=True))
@click.option(
    '--document-id',
    '-d',
    type=str,
    default=None,
    help='ドキュメントIDを指定（省略時は自動生成）',
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='詳細情報を表示',
)
def add_command(file_path: str, document_id: Optional[str], verbose: bool):
    """ドキュメントをベクトルストアに追加

    ファイルを読み込み、テキストチャンクに分割し、
    埋め込みを生成してベクトルストアに保存します。

    Args:
        file_path: 追加するファイルのパス
        document_id: ドキュメントID（省略時は自動生成）
        verbose: 詳細情報を表示するか
    """
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # 設定の読み込み
            config = get_config()

            # コンポーネントの初期化
            task = progress.add_task("初期化中...", total=None)
            vector_store = VectorStore(config)
            vector_store.initialize()

            document_processor = DocumentProcessor(config)
            embedding_generator = EmbeddingGenerator(config)
            progress.update(task, description="初期化完了")

            # ドキュメントの処理
            progress.update(task, description=f"ドキュメントを読み込み中: {Path(file_path).name}")
            document, chunks = document_processor.process_document(
                file_path, document_id
            )

            if verbose:
                console.print(f"\n[cyan]ドキュメント情報:[/cyan]")
                console.print(f"  名前: {document.name}")
                console.print(f"  タイプ: {document.doc_type}")
                console.print(f"  サイズ: {document.size:,} 文字")
                console.print(f"  チャンク数: {len(chunks)}")

            # 埋め込みの生成
            progress.update(task, description=f"{len(chunks)}個のチャンクの埋め込みを生成中...")
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = embedding_generator.embed_documents(chunk_texts)

            # ベクトルストアに追加
            progress.update(task, description="ベクトルストアに追加中...")
            vector_store.add_documents(chunks, embeddings)

            progress.update(task, description="完了", completed=True)

        # 成功メッセージ
        console.print(
            f"\n[green]✓[/green] ドキュメント '{document.name}' を正常に追加しました"
        )
        console.print(f"  ドキュメントID: {chunks[0].document_id}")
        console.print(f"  チャンク数: {len(chunks)}")
        console.print(f"  総サイズ: {document.size:,} 文字")

    except UnsupportedFileTypeError as e:
        console.print(f"[red]✗ エラー:[/red] {e}", style="bold red")
        raise click.Abort()

    except DocumentProcessorError as e:
        console.print(f"[red]✗ ドキュメント処理エラー:[/red] {e}", style="bold red")
        if verbose:
            logger.exception("ドキュメント処理エラーの詳細")
        raise click.Abort()

    except EmbeddingError as e:
        console.print(f"[red]✗ 埋め込み生成エラー:[/red] {e}", style="bold red")
        console.print("\n[yellow]ヒント:[/yellow]")
        console.print("  1. Ollamaが起動しているか確認してください")
        console.print("  2. 埋め込みモデルがインストールされているか確認してください")
        console.print("     $ ollama pull nomic-embed-text")
        if verbose:
            logger.exception("埋め込み生成エラーの詳細")
        raise click.Abort()

    except VectorStoreError as e:
        console.print(f"[red]✗ ベクトルストアエラー:[/red] {e}", style="bold red")
        if verbose:
            logger.exception("ベクトルストアエラーの詳細")
        raise click.Abort()

    except Exception as e:
        console.print(f"[red]✗ 予期しないエラー:[/red] {e}", style="bold red")
        if verbose:
            logger.exception("予期しないエラーの詳細")
        raise click.Abort()


@click.command('remove')
@click.argument('document_id', type=str)
@click.option(
    '--yes',
    '-y',
    is_flag=True,
    help='確認をスキップ',
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='詳細情報を表示',
)
def remove_command(document_id: str, yes: bool, verbose: bool):
    """ドキュメントをベクトルストアから削除

    指定されたドキュメントIDのドキュメントとそのすべてのチャンクを削除します。

    Args:
        document_id: 削除するドキュメントのID
        yes: 確認をスキップするか
        verbose: 詳細情報を表示するか
    """
    try:
        # 設定の読み込み
        config = get_config()

        # ベクトルストアの初期化
        vector_store = VectorStore(config)
        vector_store.initialize()

        # ドキュメントの存在確認
        documents = vector_store.list_documents()
        target_doc = None
        for doc in documents:
            if doc['document_id'] == document_id:
                target_doc = doc
                break

        if not target_doc:
            console.print(
                f"[yellow]警告:[/yellow] ドキュメントID '{document_id}' が見つかりませんでした",
                style="bold yellow"
            )
            if verbose:
                console.print("\n利用可能なドキュメントID:")
                for doc in documents:
                    console.print(f"  - {doc['document_id']}")
            raise click.Abort()

        # 削除確認
        if not yes:
            console.print(f"\n削除するドキュメント:")
            console.print(f"  名前: {target_doc['document_name']}")
            console.print(f"  ソース: {target_doc['source']}")
            console.print(f"  チャンク数: {target_doc['chunk_count']}")

            if not click.confirm("\n本当に削除しますか?", default=False):
                console.print("[yellow]削除をキャンセルしました[/yellow]")
                return

        # 削除実行
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("削除中...", total=None)
            deleted_count = vector_store.delete(document_id=document_id)
            progress.update(task, description="完了", completed=True)

        # 成功メッセージ
        console.print(
            f"\n[green]✓[/green] ドキュメント '{target_doc['document_name']}' を削除しました"
        )
        console.print(f"  削除されたチャンク数: {deleted_count}")

    except VectorStoreError as e:
        console.print(f"[red]✗ ベクトルストアエラー:[/red] {e}", style="bold red")
        if verbose:
            logger.exception("ベクトルストアエラーの詳細")
        raise click.Abort()

    except Exception as e:
        console.print(f"[red]✗ 予期しないエラー:[/red] {e}", style="bold red")
        if verbose:
            logger.exception("予期しないエラーの詳細")
        raise click.Abort()


@click.command('list')
@click.option(
    '--limit',
    '-l',
    type=int,
    default=None,
    help='表示するドキュメント数の上限',
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='詳細情報を表示',
)
def list_command(limit: Optional[int], verbose: bool):
    """ベクトルストア内のドキュメント一覧を表示

    登録されているすべてのドキュメントの情報を表形式で表示します。

    Args:
        limit: 表示するドキュメント数の上限
        verbose: 詳細情報を表示するか
    """
    try:
        # 設定の読み込み
        config = get_config()

        # ベクトルストアの初期化
        vector_store = VectorStore(config)
        vector_store.initialize()

        # ドキュメント一覧の取得
        documents = vector_store.list_documents(limit=limit)

        if not documents:
            console.print("[yellow]ベクトルストアにドキュメントがありません[/yellow]")
            console.print("\nドキュメントを追加するには:")
            console.print("  $ rag-cli add <file_path>")
            return

        # テーブルの作成
        table = Table(title=f"ドキュメント一覧 (総数: {len(documents)})")
        table.add_column("ドキュメント名", style="cyan", no_wrap=True)
        table.add_column("タイプ", style="magenta")
        table.add_column("チャンク数", justify="right", style="green")
        table.add_column("サイズ", justify="right", style="blue")

        if verbose:
            table.add_column("ドキュメントID", style="dim")
            table.add_column("ソース", style="dim")

        # ドキュメント情報を追加
        for doc in documents:
            row = [
                doc['document_name'],
                doc['doc_type'].upper(),
                str(doc['chunk_count']),
                f"{doc['total_size']:,} 字",
            ]

            if verbose:
                row.extend([
                    doc['document_id'],
                    doc['source'],
                ])

            table.add_row(*row)

        # テーブルの表示
        console.print()
        console.print(table)

        # 統計情報の表示
        if verbose:
            total_chunks = sum(doc['chunk_count'] for doc in documents)
            total_size = sum(doc['total_size'] for doc in documents)

            console.print(f"\n[cyan]統計情報:[/cyan]")
            console.print(f"  総ドキュメント数: {len(documents)}")
            console.print(f"  総チャンク数: {total_chunks}")
            console.print(f"  総サイズ: {total_size:,} 文字")

            # コレクション情報の表示
            collection_info = vector_store.get_collection_info()
            console.print(f"\n[cyan]コレクション情報:[/cyan]")
            console.print(f"  コレクション名: {collection_info['collection_name']}")
            console.print(f"  保存場所: {collection_info['persist_directory']}")

    except VectorStoreError as e:
        console.print(f"[red]✗ ベクトルストアエラー:[/red] {e}", style="bold red")
        if verbose:
            logger.exception("ベクトルストアエラーの詳細")
        raise click.Abort()

    except Exception as e:
        console.print(f"[red]✗ 予期しないエラー:[/red] {e}", style="bold red")
        if verbose:
            logger.exception("予期しないエラーの詳細")
        raise click.Abort()


@click.command('clear')
@click.option(
    '--yes',
    '-y',
    is_flag=True,
    help='確認をスキップ',
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='詳細情報を表示',
)
def clear_command(yes: bool, verbose: bool):
    """ベクトルストア内のすべてのドキュメントを削除

    警告: この操作は取り消せません。すべてのドキュメントとチャンクが削除されます。

    Args:
        yes: 確認をスキップするか
        verbose: 詳細情報を表示するか
    """
    try:
        # 設定の読み込み
        config = get_config()

        # ベクトルストアの初期化
        vector_store = VectorStore(config)
        vector_store.initialize()

        # ドキュメント数の取得
        document_count = vector_store.get_document_count()

        if document_count == 0:
            console.print("[yellow]ベクトルストアは既に空です[/yellow]")
            return

        # ドキュメント情報の表示
        if verbose:
            documents = vector_store.list_documents()
            console.print(f"\n削除されるドキュメント数: {len(documents)}")
            console.print(f"削除されるチャンク数: {document_count}")

        # 削除確認
        if not yes:
            console.print(
                f"\n[red bold]警告:[/red bold] "
                f"すべてのドキュメント（{document_count}チャンク）を削除します"
            )
            console.print("[yellow]この操作は取り消せません[/yellow]")

            confirmation = click.prompt(
                "\n続行するには 'DELETE' と入力してください",
                type=str,
                default="",
            )

            if confirmation != "DELETE":
                console.print("[yellow]削除をキャンセルしました[/yellow]")
                return

        # 削除実行
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("すべてのドキュメントを削除中...", total=None)
            vector_store.clear()
            progress.update(task, description="完了", completed=True)

        # 成功メッセージ
        console.print(
            f"\n[green]✓[/green] すべてのドキュメントを削除しました"
        )
        console.print(f"  削除されたチャンク数: {document_count}")

    except VectorStoreError as e:
        console.print(f"[red]✗ ベクトルストアエラー:[/red] {e}", style="bold red")
        if verbose:
            logger.exception("ベクトルストアエラーの詳細")
        raise click.Abort()

    except Exception as e:
        console.print(f"[red]✗ 予期しないエラー:[/red] {e}", style="bold red")
        if verbose:
            logger.exception("予期しないエラーの詳細")
        raise click.Abort()
