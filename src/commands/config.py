"""設定・管理コマンドモジュール

このモジュールはRAGアプリケーションの初期化、ステータス確認、設定変更などの
管理機能を提供するCLIコマンドを実装します。
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..rag.embeddings import OllamaEmbeddingGenerator
from ..rag.vector_store import VectorStore
from ..utils.config import Config, ConfigError, get_config

console = Console()


@click.command("init")
@click.option(
    "--force",
    is_flag=True,
    help="既存のデータベースを削除して初期化"
)
def init_command(force: bool):
    """RAGシステムの初期化

    ChromaDBデータベースを初期化し、必要なディレクトリを作成します。
    Ollamaサーバーへの接続確認も行います。
    """
    try:
        console.print("\n[bold cyan]RAGシステムを初期化しています...[/bold cyan]\n")

        # 設定の読み込み
        try:
            config = get_config()
        except ConfigError as e:
            console.print(f"[bold red]設定エラー:[/bold red] {str(e)}")
            sys.exit(1)

        # ChromaDBディレクトリの確認
        chroma_path = config.get_chroma_path()

        if chroma_path.exists() and not force:
            console.print(
                f"[yellow]警告:[/yellow] データベースディレクトリが既に存在します: {chroma_path}"
            )
            if not click.confirm("既存のデータベースを使用しますか?"):
                console.print("[yellow]初期化をキャンセルしました[/yellow]")
                sys.exit(0)

        # VectorStoreの初期化
        vector_store = VectorStore(config)
        vector_store.initialize()

        if force:
            console.print("[yellow]既存データをクリアしています...[/yellow]")
            vector_store.clear()

        # Ollamaサーバーの接続確認
        console.print("\n[cyan]Ollamaサーバーへの接続を確認中...[/cyan]")

        try:
            embeddings = OllamaEmbeddingGenerator(config)
            # テスト用の埋め込み生成
            test_embedding = embeddings.embed_query("test")

            if test_embedding:
                console.print("[green]✓[/green] Ollamaサーバーに接続しました")
                console.print(
                    f"  - 埋め込みモデル: [bold]{config.ollama_embedding_model}[/bold]"
                )
                console.print(
                    f"  - LLMモデル: [bold]{config.ollama_llm_model}[/bold]"
                )
        except Exception as e:
            console.print(f"[yellow]警告:[/yellow] Ollamaサーバーへの接続に失敗しました")
            console.print(f"  エラー: {str(e)}")
            console.print(
                "\n  Ollamaサーバーが起動していることを確認してください:"
            )
            console.print(f"  $ ollama serve")
            console.print(
                f"\n  必要なモデルがダウンロードされていることを確認してください:"
            )
            console.print(f"  $ ollama pull {config.ollama_llm_model}")
            console.print(f"  $ ollama pull {config.ollama_embedding_model}")

        # 初期化完了メッセージ
        console.print("\n[bold green]✓ RAGシステムの初期化が完了しました![/bold green]\n")

        # 設定情報の表示
        _display_config_info(config, vector_store)

        console.print("\n[dim]次のコマンドを使用してドキュメントを追加できます:[/dim]")
        console.print("  [cyan]rag add <file_path>[/cyan]\n")

    except Exception as e:
        console.print(f"\n[bold red]エラー:[/bold red] {str(e)}")
        sys.exit(1)


@click.command("status")
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="詳細情報を表示"
)
def status_command(verbose: bool):
    """RAGシステムのステータス表示

    データベースの状態、ドキュメント数、設定情報などを表示します。
    """
    try:
        # 設定の読み込み
        try:
            config = get_config()
        except ConfigError as e:
            console.print(f"[bold red]設定エラー:[/bold red] {str(e)}")
            sys.exit(1)

        console.print("\n[bold cyan]RAGシステムのステータス[/bold cyan]\n")

        # データベース状態の確認
        chroma_path = config.get_chroma_path()

        if not chroma_path.exists():
            console.print(
                "[yellow]データベースが初期化されていません[/yellow]"
            )
            console.print("\n次のコマンドで初期化してください:")
            console.print("  [cyan]rag init[/cyan]\n")
            sys.exit(0)

        # VectorStoreの初期化
        vector_store = VectorStore(config)
        vector_store.initialize()

        # コレクション情報の取得
        collection_info = vector_store.get_collection_info()
        documents = vector_store.list_documents()

        # ステータステーブルの作成
        status_table = Table(show_header=False, box=None, padding=(0, 2))
        status_table.add_column("項目", style="cyan")
        status_table.add_column("値", style="white")

        # データベース情報
        status_table.add_row(
            "データベースパス",
            str(chroma_path)
        )
        status_table.add_row(
            "コレクション名",
            collection_info['collection_name']
        )
        status_table.add_row(
            "登録ドキュメント数",
            f"[bold]{collection_info['unique_documents']}[/bold] 個"
        )
        status_table.add_row(
            "総チャンク数",
            f"[bold]{collection_info['total_chunks']}[/bold] 個"
        )

        console.print(Panel(status_table, title="データベース情報", border_style="blue"))

        # Ollama接続状態の確認
        console.print()
        ollama_table = Table(show_header=False, box=None, padding=(0, 2))
        ollama_table.add_column("項目", style="cyan")
        ollama_table.add_column("値", style="white")

        ollama_table.add_row("サーバーURL", config.ollama_base_url)
        ollama_table.add_row("LLMモデル", config.ollama_llm_model)
        ollama_table.add_row("埋め込みモデル", config.ollama_embedding_model)

        # Ollamaへの接続確認
        try:
            embeddings = OllamaEmbeddingGenerator(config)
            embeddings.embed_query("test")
            ollama_status = "[green]✓ 接続OK[/green]"
        except Exception:
            ollama_status = "[red]✗ 接続失敗[/red]"

        ollama_table.add_row("接続状態", ollama_status)

        console.print(Panel(ollama_table, title="Ollama設定", border_style="blue"))

        # 詳細情報の表示
        if verbose:
            console.print()
            _display_config_info(config, vector_store, show_documents=True)

        console.print()

    except Exception as e:
        console.print(f"\n[bold red]エラー:[/bold red] {str(e)}")
        sys.exit(1)


@click.command("config")
@click.argument("action", type=click.Choice(["show", "set", "reset"]))
@click.argument("key", required=False)
@click.argument("value", required=False)
def config_command(action: str, key: Optional[str], value: Optional[str]):
    """設定の表示・変更

    ACTIONS:
        show   - 現在の設定を表示
        set    - 設定値を変更
        reset  - 設定をデフォルト値にリセット

    EXAMPLES:
        rag config show
        rag config set CHUNK_SIZE 1500
        rag config reset
    """
    try:
        # 設定の読み込み
        try:
            config = get_config()
        except ConfigError as e:
            console.print(f"[bold red]設定エラー:[/bold red] {str(e)}")
            sys.exit(1)

        if action == "show":
            _show_config(config)

        elif action == "set":
            if not key or not value:
                console.print(
                    "[bold red]エラー:[/bold red] set コマンドには KEY と VALUE が必要です"
                )
                console.print("\n使用例:")
                console.print("  [cyan]rag config set CHUNK_SIZE 1500[/cyan]")
                sys.exit(1)

            _set_config(key, value)

        elif action == "reset":
            if click.confirm("すべての設定をデフォルト値にリセットしますか?"):
                _reset_config()
            else:
                console.print("[yellow]リセットをキャンセルしました[/yellow]")

    except Exception as e:
        console.print(f"\n[bold red]エラー:[/bold red] {str(e)}")
        sys.exit(1)


def _show_config(config: Config):
    """現在の設定を表示"""
    console.print("\n[bold cyan]現在の設定[/bold cyan]\n")

    config_dict = config.to_dict()

    # 設定テーブルの作成
    table = Table(show_header=True, box=None, padding=(0, 2))
    table.add_column("設定項目", style="cyan", no_wrap=True)
    table.add_column("現在値", style="white")
    table.add_column("デフォルト値", style="dim")

    # 各設定項目の表示
    defaults = {
        "ollama_base_url": Config.DEFAULT_OLLAMA_BASE_URL,
        "ollama_llm_model": Config.DEFAULT_OLLAMA_LLM_MODEL,
        "ollama_embedding_model": Config.DEFAULT_OLLAMA_EMBEDDING_MODEL,
        "chroma_persist_directory": Config.DEFAULT_CHROMA_PERSIST_DIRECTORY,
        "chunk_size": Config.DEFAULT_CHUNK_SIZE,
        "chunk_overlap": Config.DEFAULT_CHUNK_OVERLAP,
        "log_level": Config.DEFAULT_LOG_LEVEL,
    }

    for key, current_value in config_dict.items():
        default_value = defaults.get(key, "")

        # 値が変更されている場合は強調表示
        if str(current_value) != str(default_value):
            current_value_str = f"[bold yellow]{current_value}[/bold yellow]"
        else:
            current_value_str = str(current_value)

        table.add_row(
            key.upper(),
            current_value_str,
            str(default_value)
        )

    console.print(table)

    console.print("\n[dim]設定を変更するには:[/dim]")
    console.print("  [cyan]rag config set <KEY> <VALUE>[/cyan]")
    console.print("\n[dim].env ファイルを直接編集することもできます[/dim]\n")


def _set_config(key: str, value: str):
    """設定値を変更"""
    console.print(f"\n[cyan]設定を変更しています...[/cyan]")
    console.print(f"  {key.upper()} = {value}\n")

    # .envファイルのパス
    env_path = Path(".env")

    # .envファイルの読み込み
    env_lines = []
    key_found = False

    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith(key.upper() + "="):
                    env_lines.append(f"{key.upper()}={value}\n")
                    key_found = True
                else:
                    env_lines.append(line)

    # キーが見つからなかった場合は追加
    if not key_found:
        env_lines.append(f"{key.upper()}={value}\n")

    # .envファイルに書き込み
    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(env_lines)

    console.print("[green]✓[/green] 設定を保存しました")

    # 設定の再読み込みとバリデーション
    try:
        get_config(reload=True)
        console.print("[green]✓[/green] 設定の検証に成功しました\n")
    except ConfigError as e:
        console.print(f"\n[bold red]警告:[/bold red] 設定値が不正です: {str(e)}")
        console.print("設定ファイルを確認してください\n")
        sys.exit(1)


def _reset_config():
    """設定をデフォルト値にリセット"""
    console.print("\n[cyan]設定をリセットしています...[/cyan]\n")

    env_path = Path(".env")

    if env_path.exists():
        # .envファイルのバックアップ
        backup_path = Path(".env.backup")
        import shutil
        shutil.copy(env_path, backup_path)
        console.print(f"[dim]バックアップを作成しました: {backup_path}[/dim]")

        # .envファイルを削除
        env_path.unlink()

    # デフォルトの.envファイルを作成
    default_env = f"""# Ollama設定
OLLAMA_BASE_URL={Config.DEFAULT_OLLAMA_BASE_URL}
OLLAMA_LLM_MODEL={Config.DEFAULT_OLLAMA_LLM_MODEL}
OLLAMA_EMBEDDING_MODEL={Config.DEFAULT_OLLAMA_EMBEDDING_MODEL}

# ChromaDB設定
CHROMA_PERSIST_DIRECTORY={Config.DEFAULT_CHROMA_PERSIST_DIRECTORY}

# チャンク設定
CHUNK_SIZE={Config.DEFAULT_CHUNK_SIZE}
CHUNK_OVERLAP={Config.DEFAULT_CHUNK_OVERLAP}

# ログ設定
LOG_LEVEL={Config.DEFAULT_LOG_LEVEL}
"""

    with open(env_path, "w", encoding="utf-8") as f:
        f.write(default_env)

    console.print("[green]✓[/green] 設定をデフォルト値にリセットしました\n")

    # リセット後の設定を表示
    get_config(reload=True)
    _show_config(get_config())


def _display_config_info(
    config: Config,
    vector_store: VectorStore,
    show_documents: bool = False
):
    """設定情報を表示（共通ヘルパー関数）"""
    # 設定テーブル
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("項目", style="cyan")
    config_table.add_column("値", style="white")

    config_table.add_row("データベースパス", str(config.get_chroma_path()))
    config_table.add_row("チャンクサイズ", f"{config.chunk_size} 文字")
    config_table.add_row("チャンクオーバーラップ", f"{config.chunk_overlap} 文字")
    config_table.add_row("ログレベル", config.log_level)

    console.print(Panel(config_table, title="設定情報", border_style="blue"))

    # ドキュメント一覧の表示
    if show_documents:
        documents = vector_store.list_documents()

        if documents:
            console.print()
            doc_table = Table(title="登録ドキュメント", box=None)
            doc_table.add_column("ID", style="cyan", no_wrap=True)
            doc_table.add_column("ドキュメント名", style="white")
            doc_table.add_column("タイプ", style="yellow")
            doc_table.add_column("チャンク数", style="magenta", justify="right")

            for doc in documents:
                doc_table.add_row(
                    doc['document_id'][:8] + "...",
                    doc['document_name'],
                    doc['doc_type'],
                    str(doc['chunk_count'])
                )

            console.print(doc_table)
