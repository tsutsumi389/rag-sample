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
from ..rag.vision_embeddings import VisionEmbeddings, VisionEmbeddingError
from ..rag.image_processor import ImageProcessor, ImageProcessorError
from ..utils.config import get_config

logger = logging.getLogger(__name__)
console = Console()


class DocumentCommandError(Exception):
    """ドキュメントコマンドのエラー"""
    pass


# 画像ファイル拡張子の定義
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}


def _is_image_file(file_path: str) -> bool:
    """ファイルが画像かどうかを拡張子から判定

    Args:
        file_path: チェックするファイルのパス

    Returns:
        画像ファイルの場合True
    """
    return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS


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
    '--caption',
    '-c',
    type=str,
    default=None,
    help='画像のキャプション（画像ファイルの場合のみ、省略時は自動生成）',
)
@click.option(
    '--tags',
    '-t',
    type=str,
    default=None,
    help='カンマ区切りのタグ（画像ファイルの場合のみ、例: tag1,tag2,tag3）',
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='詳細情報を表示',
)
def add_command(file_path: str, document_id: Optional[str], caption: Optional[str], tags: Optional[str], verbose: bool):
    """ドキュメントまたは画像をベクトルストアに追加

    ファイルの拡張子から自動的に画像かドキュメントかを判定し、
    適切な処理を行います。

    - 画像ファイル（.jpg, .png, .gif等）: ビジョン埋め込みを生成
    - ドキュメントファイル（.txt, .md, .pdf等）: テキストチャンクに分割して埋め込みを生成
    - ディレクトリ: 内部のファイルを分類して画像とドキュメントを別々に処理

    Args:
        file_path: 追加するファイルまたはディレクトリのパス
        document_id: ドキュメントID（省略時は自動生成）
        caption: 画像のキャプション（画像ファイルの場合のみ）
        tags: カンマ区切りのタグ（画像ファイルの場合のみ）
        verbose: 詳細情報を表示するか
    """
    path = Path(file_path)

    # ディレクトリの場合は内部のファイルを分類して処理
    if path.is_dir():
        _add_from_directory(file_path, document_id, caption, tags, verbose)
        return

    # ファイルの場合は拡張子で判定
    if _is_image_file(file_path):
        _add_image_file(file_path, caption, tags, verbose)
    else:
        _add_document_file(file_path, document_id, verbose)


def _add_document_file(file_path: str, document_id: Optional[str], verbose: bool):
    """テキストドキュメントファイルを追加

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


def _add_image_file(image_path: str, caption: Optional[str], tags: Optional[str], verbose: bool):
    """画像ファイルを追加

    Args:
        image_path: 追加する画像ファイルのパス
        caption: 画像のキャプション（省略時は自動生成）
        tags: カンマ区切りのタグ
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

            vision_embeddings = VisionEmbeddings(config)
            image_processor = ImageProcessor(vision_embeddings, config)
            progress.update(task, description="初期化完了")

            # タグの処理
            tag_list = [tag.strip() for tag in tags.split(',')] if tags else []

            # 画像の読み込み
            path = Path(image_path)
            progress.update(task, description=f"画像を読み込み中: {path.name}")
            image = image_processor.load_image(str(path), caption=caption, tags=tag_list)

            if verbose:
                console.print(f"\n[cyan]画像情報:[/cyan]")
                console.print(f"  ファイル名: {image.file_name}")
                console.print(f"  タイプ: {image.image_type.upper()}")
                console.print(f"  キャプション: {image.caption[:50]}...")

            # 埋め込みの生成
            progress.update(task, description="画像の埋め込みを生成中...")
            embeddings = vision_embeddings.embed_images([image.file_path])

            # ベクトルストアに追加
            progress.update(task, description="ベクトルストアに追加中...")
            image_ids = vector_store.add_images([image], embeddings)

            progress.update(task, description="完了", completed=True)

        # 成功メッセージ
        console.print(f"\n[green]✓[/green] 画像 '{image.file_name}' を正常に追加しました")
        if verbose:
            console.print(f"  画像ID: {image_ids[0]}")

    except ImageProcessorError as e:
        console.print(f"[red]✗ 画像処理エラー:[/red] {e}", style="bold red")
        if verbose:
            logger.exception("画像処理エラーの詳細")
        raise click.Abort()

    except VisionEmbeddingError as e:
        console.print(f"[red]✗ ビジョン埋め込みエラー:[/red] {e}", style="bold red")
        console.print("\n[yellow]ヒント:[/yellow]")
        console.print("  1. Ollamaが起動しているか確認してください")
        console.print("  2. ビジョンモデルがインストールされているか確認してください")
        console.print("     $ ollama pull llava")
        if verbose:
            logger.exception("ビジョン埋め込みエラーの詳細")
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


def _add_from_directory(directory_path: str, document_id: Optional[str], caption: Optional[str], tags: Optional[str], verbose: bool):
    """ディレクトリから画像とドキュメントを一括追加

    ディレクトリ内のファイルを走査し、拡張子から画像とドキュメントを分類して
    それぞれ適切に処理します。

    Args:
        directory_path: ディレクトリのパス
        document_id: ドキュメントID（ドキュメントファイルの場合、各ファイルで自動生成される）
        caption: 画像のキャプション（画像ファイルの場合、各画像で自動生成される）
        tags: カンマ区切りのタグ（画像ファイルの場合のみ）
        verbose: 詳細情報を表示するか
    """
    path = Path(directory_path)

    # ディレクトリ内のファイルを収集
    all_files = []
    for file_path in path.rglob('*'):
        if file_path.is_file():
            all_files.append(file_path)

    if not all_files:
        console.print(f"[yellow]警告:[/yellow] {path} にファイルが見つかりませんでした")
        return

    # 画像ファイルとドキュメントファイルに分類
    image_files = [f for f in all_files if _is_image_file(str(f))]
    document_files = [f for f in all_files if not _is_image_file(str(f))]

    # 結果の表示
    console.print(f"\n[cyan]ディレクトリ分析結果:[/cyan]")
    console.print(f"  総ファイル数: {len(all_files)}")
    console.print(f"  画像ファイル: {len(image_files)}個")
    console.print(f"  ドキュメントファイル: {len(document_files)}個")

    # 画像ファイルの処理
    if image_files:
        console.print(f"\n[bold cyan]画像ファイルを処理中...[/bold cyan]")
        _add_images_from_directory(directory_path, caption, tags, verbose)

    # ドキュメントファイルの処理
    processed_docs = 0
    skipped_docs = 0
    if document_files:
        console.print(f"\n[bold cyan]ドキュメントファイルを処理中...[/bold cyan]")
        for doc_file in document_files:
            try:
                console.print(f"\n処理中: {doc_file.name}")
                _add_document_file(str(doc_file), None, verbose)
                processed_docs += 1
            except (UnsupportedFileTypeError, DocumentProcessorError) as e:
                console.print(f"  [yellow]スキップ:[/yellow] {e}")
                skipped_docs += 1
                continue
            except Exception as e:
                console.print(f"  [red]エラー:[/red] {e}")
                skipped_docs += 1
                if verbose:
                    logger.exception(f"ファイル処理エラー: {doc_file}")
                continue

    # 完了メッセージ
    console.print(f"\n[green]✓[/green] ディレクトリの処理が完了しました")
    if image_files:
        console.print(f"  画像: {len(image_files)}個を追加")
    if document_files:
        console.print(f"  ドキュメント: {processed_docs}個を追加")
        if skipped_docs > 0:
            console.print(f"  スキップ: {skipped_docs}個")


def _add_images_from_directory(directory_path: str, caption: Optional[str], tags: Optional[str], verbose: bool):
    """ディレクトリから画像を一括追加

    Args:
        directory_path: 画像ディレクトリのパス
        caption: 画像のキャプション（ディレクトリ一括追加では使用されず、各画像で自動生成される）
        tags: カンマ区切りのタグ
        verbose: 詳細情報を表示するか

    Note:
        caption パラメータはインターフェース統一のため存在しますが、
        ディレクトリからの一括追加では各画像のキャプションが自動生成されます。
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

            vision_embeddings = VisionEmbeddings(config)
            image_processor = ImageProcessor(vision_embeddings, config)
            progress.update(task, description="初期化完了")

            # タグの処理
            tag_list = [tag.strip() for tag in tags.split(',')] if tags else []

            # ディレクトリから画像を読み込み
            path = Path(directory_path)
            progress.update(task, description=f"ディレクトリから画像を読み込み中: {path.name}")
            images = image_processor.load_images_from_directory(
                str(path),
                auto_caption=True,
                tags=tag_list
            )

            if not images:
                console.print(f"[yellow]警告:[/yellow] {path} に画像ファイルが見つかりませんでした")
                return

            if verbose:
                console.print(f"\n[cyan]画像情報:[/cyan]")
                console.print(f"  読み込み画像数: {len(images)}")
                for img in images[:5]:  # 最初の5件のみ表示
                    console.print(f"    - {img.file_name} ({img.image_type.upper()})")
                if len(images) > 5:
                    console.print(f"    ... 他 {len(images) - 5} 件")

            # 埋め込みの生成
            progress.update(task, description=f"{len(images)}個の画像の埋め込みを生成中...")
            image_paths = [img.file_path for img in images]
            embeddings = vision_embeddings.embed_images(image_paths)

            # ベクトルストアに追加
            progress.update(task, description="ベクトルストアに追加中...")
            image_ids = vector_store.add_images(images, embeddings)

            progress.update(task, description="完了", completed=True)

        # 成功メッセージ
        console.print(f"\n[green]✓[/green] {len(images)}個の画像を正常に追加しました")
        if verbose and len(images) > 0:
            console.print(f"  最初の画像ID: {image_ids[0]}")
            console.print(f"  追加された画像数: {len(image_ids)}")

    except ImageProcessorError as e:
        console.print(f"[red]✗ 画像処理エラー:[/red] {e}", style="bold red")
        if verbose:
            logger.exception("画像処理エラーの詳細")
        raise click.Abort()

    except VisionEmbeddingError as e:
        console.print(f"[red]✗ ビジョン埋め込みエラー:[/red] {e}", style="bold red")
        console.print("\n[yellow]ヒント:[/yellow]")
        console.print("  1. Ollamaが起動しているか確認してください")
        console.print("  2. ビジョンモデルがインストールされているか確認してください")
        console.print("     $ ollama pull llava")
        if verbose:
            logger.exception("ビジョン埋め込みエラーの詳細")
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
@click.argument('item_id', type=str)
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
def remove_command(item_id: str, yes: bool, verbose: bool):
    """ドキュメントまたは画像をベクトルストアから削除

    指定されたIDのドキュメントまたは画像を自動判定して削除します。
    IDからドキュメントと画像の両方を検索し、見つかった方を削除します。

    Args:
        item_id: 削除するドキュメントIDまたは画像ID
        yes: 確認をスキップするか
        verbose: 詳細情報を表示するか
    """
    try:
        # 設定の読み込み
        config = get_config()

        # ベクトルストアの初期化
        vector_store = VectorStore(config)
        vector_store.initialize()

        # まずドキュメントとして検索
        documents = vector_store.list_documents()
        target_doc = None
        for doc in documents:
            if doc['document_id'] == item_id:
                target_doc = doc
                break

        # ドキュメントとして見つかった場合
        if target_doc:
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
                deleted_count = vector_store.delete(document_id=item_id)
                progress.update(task, description="完了", completed=True)

            # 成功メッセージ
            console.print(
                f"\n[green]✓[/green] ドキュメント '{target_doc['document_name']}' を削除しました"
            )
            console.print(f"  削除されたチャンク数: {deleted_count}")
            return

        # ドキュメントとして見つからなければ画像として検索
        image = vector_store.get_image_by_id(item_id)

        if image:
            # 削除確認
            if not yes:
                console.print(f"\n削除する画像:")
                console.print(f"  ファイル名: {image['file_name']}")
                console.print(f"  パス: {image['file_path']}")
                caption_preview = image['caption'][:50] + "..." if len(image['caption']) > 50 else image['caption']
                console.print(f"  キャプション: {caption_preview}")

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
                success = vector_store.remove_image(item_id)
                progress.update(task, description="完了", completed=True)

            if success:
                console.print(f"\n[green]✓[/green] 画像 '{image['file_name']}' を削除しました")
            else:
                console.print(f"\n[yellow]警告:[/yellow] 画像の削除に失敗しました")
            return

        # どちらにも見つからなかった場合
        console.print(
            f"[yellow]警告:[/yellow] ID '{item_id}' が見つかりませんでした",
            style="bold yellow"
        )
        console.print("\nドキュメントにも画像にも該当するIDが存在しません")
        if verbose:
            console.print("\n利用可能なドキュメントID:")
            for doc in documents:
                console.print(f"  - {doc['document_id']}")
            console.print("\n利用可能な画像IDを確認するには:")
            console.print("  $ rag list --type image -v")
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


def _list_documents(limit: Optional[int], verbose: bool):
    """テキストドキュメント一覧を表示

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
            console.print("  $ rag add <file_path>")
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


def _list_images(limit: Optional[int], format: str, verbose: bool):
    """画像一覧を表示

    Args:
        limit: 表示する画像数の上限
        format: 出力フォーマット（table または json）
        verbose: 詳細情報を表示するか
    """
    try:
        # 設定の読み込み
        config = get_config()

        # ベクトルストアの初期化
        vector_store = VectorStore(config)
        vector_store.initialize()

        # 画像一覧の取得
        images = vector_store.list_images(limit=limit)

        if not images:
            console.print("[yellow]ベクトルストアに画像がありません[/yellow]")
            console.print("\n画像を追加するには:")
            console.print("  $ rag add <image_path>")
            return

        # JSON形式で出力
        if format == 'json':
            import json
            output = []
            for img in images:
                output.append({
                    'id': img.id,
                    'file_name': img.file_name,
                    'file_path': str(img.file_path),
                    'image_type': img.image_type,
                    'caption': img.caption,
                    'tags': img.metadata.get('tags', []),
                    'created_at': img.created_at.isoformat(),
                })
            console.print(json.dumps(output, ensure_ascii=False, indent=2))
            return

        # テーブル形式で出力
        table = Table(title=f"画像一覧 (総数: {len(images)})")
        table.add_column("ファイル名", style="cyan", no_wrap=True)
        table.add_column("タイプ", style="magenta")
        table.add_column("キャプション", style="green")

        if verbose:
            table.add_column("画像ID", style="dim")
            table.add_column("タグ", style="yellow")
            table.add_column("作成日時", style="dim")

        # 画像情報を追加
        for img in images:
            # キャプションを短縮
            caption = img.caption
            if len(caption) > 50:
                caption = caption[:47] + "..."

            row = [
                img.file_name,
                img.image_type.upper(),
                caption,
            ]

            if verbose:
                # tagsは文字列化されたリストかもしれないので適切に処理
                tags = img.metadata.get('tags', [])
                if isinstance(tags, list):
                    tags_str = ", ".join(str(tag) for tag in tags)
                else:
                    tags_str = str(tags) if tags else ""
                row.extend([
                    img.id[:8] + "...",  # IDの最初の8文字のみ表示
                    tags_str if tags_str else "-",
                    img.created_at.isoformat()[:19],  # 秒まで表示
                ])

            table.add_row(*row)

        # テーブルの表示
        console.print()
        console.print(table)

        # 統計情報の表示
        if verbose:
            console.print(f"\n[cyan]統計情報:[/cyan]")
            console.print(f"  総画像数: {len(images)}")

            # 画像タイプの集計
            type_counts = {}
            for img in images:
                img_type = img.image_type
                type_counts[img_type] = type_counts.get(img_type, 0) + 1

            console.print(f"  画像タイプ別:")
            for img_type, count in type_counts.items():
                console.print(f"    {img_type.upper()}: {count}")

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


def _list_all(limit: Optional[int], verbose: bool):
    """ドキュメントと画像の両方を表示

    Args:
        limit: 表示する項目数の上限（それぞれに適用）
        verbose: 詳細情報を表示するか
    """
    console.print("[bold cyan]テキストドキュメント:[/bold cyan]")
    _list_documents(limit, verbose)

    console.print("\n[bold cyan]画像:[/bold cyan]")
    _list_images(limit, 'table', verbose)


@click.command('list')
@click.option(
    '--limit',
    '-l',
    type=int,
    default=None,
    help='表示する項目数の上限',
)
@click.option(
    '--type',
    '-t',
    type=click.Choice(['all', 'text', 'image']),
    default='all',
    help='表示するタイプ（all: すべて, text: テキストのみ, image: 画像のみ）',
)
@click.option(
    '--format',
    '-f',
    type=click.Choice(['table', 'json']),
    default='table',
    help='出力フォーマット（画像表示時のみ有効）',
)
@click.option(
    '--verbose',
    '-v',
    is_flag=True,
    help='詳細情報を表示',
)
def list_command(limit: Optional[int], type: str, format: str, verbose: bool):
    """ベクトルストア内のドキュメントと画像の一覧を表示

    登録されているドキュメントと画像の情報を表形式で表示します。
    --type オプションでテキストのみ、画像のみに絞り込むことができます。

    Args:
        limit: 表示する項目数の上限
        type: 表示するタイプ（all/text/image）
        format: 出力フォーマット（table/json、画像表示時のみ有効）
        verbose: 詳細情報を表示するか
    """
    # タイプに基づいて適切な関数を呼び出し
    if type == 'image':
        _list_images(limit, format, verbose)
    elif type == 'text':
        _list_documents(limit, verbose)
    else:  # type == 'all'
        _list_all(limit, verbose)


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


# =============================================================================
# 画像管理コマンド
# =============================================================================


@click.command('clear-images')
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
def clear_images_command(yes: bool, verbose: bool):
    """ベクトルストア内のすべての画像を削除

    警告: この操作は取り消せません。すべての画像が削除されます。

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

        # 画像一覧の取得
        images = vector_store.list_images()

        if not images:
            console.print("[yellow]ベクトルストアに画像がありません[/yellow]")
            return

        image_count = len(images)

        # 画像情報の表示
        if verbose:
            console.print(f"\n削除される画像数: {image_count}")

        # 削除確認
        if not yes:
            console.print(
                f"\n[red bold]警告:[/red bold] "
                f"すべての画像（{image_count}個）を削除します"
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
            task = progress.add_task("すべての画像を削除中...", total=None)

            # すべての画像を個別に削除
            deleted_count = 0
            for image in images:
                if vector_store.remove_image(image['id']):
                    deleted_count += 1

            progress.update(task, description="完了", completed=True)

        # 成功メッセージ
        console.print(f"\n[green]✓[/green] すべての画像を削除しました")
        console.print(f"  削除された画像数: {deleted_count}")

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
