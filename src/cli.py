"""RAG CLI エントリーポイント

このモジュールはRAGアプリケーションのメインCLIエントリーポイントです。
Clickを使用して各種コマンドを統合し、Richによる見やすい出力を提供します。
"""

import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from .commands.config import config_command, init_command, status_command
from .commands.document import (
    add_command,
    add_image_command,
    clear_command,
    clear_images_command,
    list_command,
    list_images_command,
    remove_command,
    remove_image_command,
)
from .commands.query import chat, query, search, search_images
from .utils.config import ConfigError, get_config

# Richコンソールの初期化
console = Console()

# バージョン情報
__version__ = "0.1.0"


def setup_logging(verbose: bool = False):
    """ロギングの設定

    Args:
        verbose: 詳細ログを出力するか
    """
    try:
        config = get_config()
        log_level = logging.DEBUG if verbose else getattr(logging, config.log_level.upper())
    except ConfigError:
        # 設定が読み込めない場合はデフォルト値を使用
        log_level = logging.DEBUG if verbose else logging.INFO

    # ロギングハンドラの設定
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console,
                show_time=True,
                show_path=verbose,
                markup=True,
                rich_tracebacks=True,
            )
        ],
    )


@click.group(
    name="rag",
    invoke_without_command=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="詳細ログを出力",
)
@click.option(
    "--version",
    is_flag=True,
    help="バージョン情報を表示",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, version: bool):
    """RAG (Retrieval-Augmented Generation) CLI Application

    ローカルドキュメントと画像をベクトルデータベースに保存し、
    自然言語での質問に対して関連情報を検索・回答を生成します。

    \b
    主要機能:
      • ドキュメント管理 (add, remove, list, clear)
      • 画像管理 (add-image, remove-image, list-images, clear-images)
      • 質問応答 (query)
      • ドキュメント検索 (search)
      • 対話モード (chat)
      • システム管理 (init, status, config)

    \b
    使用例:
      $ rag init                      # システムの初期化
      $ rag add sample.txt            # ドキュメントの追加
      $ rag add-image ./images        # 画像の追加
      $ rag query "RAGとは何ですか？"    # 質問応答
      $ rag search "機械学習"          # ドキュメント検索
      $ rag chat                      # 対話モード開始

    詳細は各コマンドのヘルプを参照してください:
      $ rag <command> --help
    """
    # ロギングの設定
    setup_logging(verbose)

    # バージョン情報の表示
    if version:
        console.print(f"[bold cyan]RAG CLI[/bold cyan] version [bold]{__version__}[/bold]")
        console.print("\nPowered by:")
        console.print("  • Python 3.13+")
        console.print("  • Ollama (Local LLM)")
        console.print("  • LangChain")
        console.print("  • ChromaDB")
        sys.exit(0)

    # サブコマンドが指定されていない場合はヘルプを表示
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


# ドキュメント管理コマンドの登録
cli.add_command(add_command, name="add")
cli.add_command(remove_command, name="remove")
cli.add_command(list_command, name="list")
cli.add_command(clear_command, name="clear")

# 画像管理コマンドの登録
cli.add_command(add_image_command, name="add-image")
cli.add_command(list_images_command, name="list-images")
cli.add_command(remove_image_command, name="remove-image")
cli.add_command(clear_images_command, name="clear-images")

# 検索・質問コマンドの登録
cli.add_command(query, name="query")
cli.add_command(search, name="search")
cli.add_command(search_images, name="search-images")
cli.add_command(chat, name="chat")

# 設定・管理コマンドの登録
cli.add_command(init_command, name="init")
cli.add_command(status_command, name="status")
cli.add_command(config_command, name="config")


def main():
    """メイン関数

    CLIアプリケーションのエントリーポイント。
    パッケージインストール時に実行可能コマンドとして登録されます。
    """
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]処理が中断されました[/yellow]")
        sys.exit(130)  # 128 + SIGINT (2)
    except Exception as e:
        console.print(f"\n[bold red]予期しないエラーが発生しました:[/bold red] {e}")
        logging.exception("予期しないエラー")
        sys.exit(1)


if __name__ == "__main__":
    main()
