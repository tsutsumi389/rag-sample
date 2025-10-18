"""検索・質問応答コマンドモジュール

このモジュールはquery、search、chatコマンドの実装を提供します。
RAGエンジンを使用してドキュメントの検索、質問応答、対話モードを実行します。
"""

import logging
import sys
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from ..rag.engine import RAGEngine, RAGEngineError, create_rag_engine
from ..rag.vector_store import VectorStore, VectorStoreError
from ..rag.vision_embeddings import VisionEmbeddings, VisionEmbeddingError
from ..utils.config import get_config

logger = logging.getLogger(__name__)
console = Console()


@click.command()
@click.argument("question", type=str)
@click.option(
    "--n-results",
    "-n",
    type=int,
    default=5,
    help="検索する関連ドキュメントの最大数（デフォルト: 5）"
)
@click.option(
    "--no-sources",
    is_flag=True,
    help="情報源の表示を省略"
)
@click.option(
    "--filter",
    "-f",
    type=str,
    default=None,
    help="メタデータフィルタ（JSON形式、例: '{\"document_type\": \"pdf\"}'）"
)
def query(
    question: str,
    n_results: int,
    no_sources: bool,
    filter: Optional[str]
) -> None:
    """質問に対して回答を生成します

    指定された質問に対してベクトルデータベースから関連ドキュメントを検索し、
    LLMを使用して回答を生成します。

    Args:
        question: ユーザーの質問文
        n_results: 検索する関連ドキュメントの最大数
        no_sources: 情報源の表示を省略するフラグ
        filter: メタデータフィルタ（JSON形式）

    Example:
        $ rag query "RAGとは何ですか？"
        $ rag query "Pythonの特徴を教えて" -n 3
        $ rag query "設定方法は？" --no-sources
    """
    try:
        # メタデータフィルタのパース
        where_filter = None
        if filter:
            import json
            try:
                where_filter = json.loads(filter)
            except json.JSONDecodeError as e:
                console.print(
                    f"[bold red]エラー:[/bold red] フィルタのJSON形式が不正です: {str(e)}",
                    style="red"
                )
                sys.exit(1)

        # RAGエンジンの初期化
        with console.status("[bold green]RAGエンジンを初期化中..."):
            config = get_config()
            engine = create_rag_engine(config=config)
            engine.initialize()

        # 質問の表示
        console.print(Panel(
            f"[bold cyan]質問:[/bold cyan] {question}",
            border_style="cyan"
        ))

        # 質問応答の実行
        with console.status(
            f"[bold green]関連ドキュメントを検索中（最大{n_results}件）..."
        ):
            result = engine.query(
                question=question,
                n_results=n_results,
                where=where_filter,
                include_sources=not no_sources
            )

        # 回答の表示
        console.print()
        console.print(Panel(
            Markdown(result["answer"]),
            title="[bold green]回答",
            border_style="green",
            padding=(1, 2)
        ))

        # コンテキスト情報の表示
        console.print(
            f"\n[dim]使用したコンテキスト: {result['context_count']}件[/dim]"
        )

        # 情報源の表示
        if not no_sources and "sources" in result and result["sources"]:
            console.print("\n[bold]参照元ドキュメント:[/bold]")

            sources_table = Table(show_header=True, header_style="bold magenta")
            sources_table.add_column("ドキュメント名", style="cyan")
            sources_table.add_column("ソース", style="dim")
            sources_table.add_column("類似度", justify="right", style="green")

            for source in result["sources"]:
                sources_table.add_row(
                    source["name"],
                    source["source"],
                    f"{source['score']:.4f}"
                )

            console.print(sources_table)

        logger.info(f"質問応答が完了しました: '{question[:50]}'")

    except RAGEngineError as e:
        console.print(f"[bold red]エラー:[/bold red] {str(e)}", style="red")
        logger.error(f"質問応答エラー: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]処理が中断されました[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(
            f"[bold red]予期しないエラーが発生しました:[/bold red] {str(e)}",
            style="red"
        )
        logger.exception("予期しないエラー")
        sys.exit(1)


@click.command()
@click.argument("query_text", type=str)
@click.option(
    "--n-results",
    "-n",
    type=int,
    default=10,
    help="検索する結果の最大数（デフォルト: 10）"
)
@click.option(
    "--filter",
    "-f",
    type=str,
    default=None,
    help="メタデータフィルタ（JSON形式、例: '{\"document_type\": \"pdf\"}'）"
)
@click.option(
    "--show-content",
    is_flag=True,
    help="検索結果の内容を表示"
)
def search(
    query_text: str,
    n_results: int,
    filter: Optional[str],
    show_content: bool
) -> None:
    """ドキュメントを検索します

    指定されたクエリに類似したドキュメントチャンクを検索し、
    類似度スコアと共に表示します。

    Args:
        query_text: 検索クエリ文字列
        n_results: 検索する結果の最大数
        filter: メタデータフィルタ（JSON形式）
        show_content: 検索結果の内容を表示するフラグ

    Example:
        $ rag search "機械学習"
        $ rag search "Python" -n 5
        $ rag search "設定" --show-content
    """
    try:
        # メタデータフィルタのパース
        where_filter = None
        if filter:
            import json
            try:
                where_filter = json.loads(filter)
            except json.JSONDecodeError as e:
                console.print(
                    f"[bold red]エラー:[/bold red] フィルタのJSON形式が不正です: {str(e)}",
                    style="red"
                )
                sys.exit(1)

        # RAGエンジンの初期化
        with console.status("[bold green]RAGエンジンを初期化中..."):
            config = get_config()
            engine = create_rag_engine(config=config)
            engine.initialize()

        # 検索の実行
        with console.status(
            f"[bold green]'{query_text}'を検索中（最大{n_results}件）..."
        ):
            search_results = engine.retrieve(
                query=query_text,
                n_results=n_results,
                where=where_filter
            )

        # 結果の表示
        if not search_results:
            console.print(
                "[yellow]検索結果が見つかりませんでした。[/yellow]"
            )
            return

        console.print(
            f"\n[bold green]検索結果: {len(search_results)}件[/bold green]\n"
        )

        # 検索結果をテーブル表示
        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("#", justify="right", style="cyan", width=4)
        results_table.add_column("ドキュメント名", style="green")
        results_table.add_column("類似度", justify="right", style="yellow")

        if show_content:
            results_table.add_column("内容（抜粋）", style="dim", max_width=60)

        for i, result in enumerate(search_results, 1):
            row_data = [
                str(i),
                result.document_name,
                f"{result.score:.4f}"
            ]

            if show_content:
                # 内容を最初の100文字に制限
                content_preview = result.chunk.content[:100]
                if len(result.chunk.content) > 100:
                    content_preview += "..."
                row_data.append(content_preview)

            results_table.add_row(*row_data)

        console.print(results_table)

        # 詳細情報を表示（--show-contentが指定された場合）
        if show_content:
            console.print("\n[bold]検索結果の詳細:[/bold]\n")

            for i, result in enumerate(search_results, 1):
                console.print(Panel(
                    f"[bold cyan]ソース:[/bold cyan] {result.document_source}\n"
                    f"[bold cyan]類似度:[/bold cyan] {result.score:.4f}\n\n"
                    f"{result.chunk.content}",
                    title=f"[bold green]{i}. {result.document_name}",
                    border_style="blue",
                    padding=(1, 2)
                ))

        logger.info(f"検索が完了しました: '{query_text[:50]}' -> {len(search_results)}件")

    except RAGEngineError as e:
        console.print(f"[bold red]エラー:[/bold red] {str(e)}", style="red")
        logger.error(f"検索エラー: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]検索が中断されました[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(
            f"[bold red]予期しないエラーが発生しました:[/bold red] {str(e)}",
            style="red"
        )
        logger.exception("予期しないエラー")
        sys.exit(1)


@click.command()
@click.option(
    "--n-results",
    "-n",
    type=int,
    default=3,
    help="各質問で検索する関連ドキュメントの数（デフォルト: 3）"
)
@click.option(
    "--no-sources",
    is_flag=True,
    help="情報源の表示を省略"
)
def chat(n_results: int, no_sources: bool) -> None:
    """対話モードで質問応答を行います

    対話形式で連続して質問を行うことができます。
    会話履歴を保持し、コンテキストを考慮した回答を生成します。
    'exit'、'quit'、'q'で終了します。

    Args:
        n_results: 各質問で検索する関連ドキュメントの数
        no_sources: 情報源の表示を省略するフラグ

    Example:
        $ rag chat
        $ rag chat -n 5
        $ rag chat --no-sources
    """
    try:
        # RAGエンジンの初期化
        with console.status("[bold green]RAGエンジンを初期化中..."):
            config = get_config()
            engine = create_rag_engine(config=config)
            engine.initialize()

        # チャットモード開始のメッセージ
        console.print(Panel(
            "[bold green]対話モードを開始しました[/bold green]\n\n"
            "質問を入力してEnterキーを押してください。\n"
            "終了するには 'exit'、'quit'、または 'q' を入力してください。\n"
            "履歴をクリアするには 'clear' を入力してください。",
            title="RAG チャット",
            border_style="green"
        ))

        console.print()

        # チャットループ
        while True:
            try:
                # ユーザー入力の取得
                user_input = console.input("[bold cyan]あなた:[/bold cyan] ").strip()

                # 終了コマンドのチェック
                if user_input.lower() in ["exit", "quit", "q"]:
                    console.print("\n[green]チャットを終了します。[/green]")
                    break

                # 履歴クリアコマンド
                if user_input.lower() == "clear":
                    engine.clear_chat_history()
                    console.print(
                        "[yellow]チャット履歴をクリアしました。[/yellow]\n"
                    )
                    continue

                # 空入力のスキップ
                if not user_input:
                    continue

                # 質問応答の実行
                with console.status("[bold green]回答を生成中..."):
                    result = engine.chat(
                        message=user_input,
                        n_results=n_results,
                        include_sources=not no_sources
                    )

                # 回答の表示
                console.print(
                    f"\n[bold green]アシスタント:[/bold green]\n{result['answer']}\n"
                )

                # コンテキスト情報の表示
                console.print(
                    f"[dim]（コンテキスト: {result['context_count']}件 | "
                    f"履歴: {result['history_length']}メッセージ）[/dim]"
                )

                # 情報源の表示（簡易版）
                if not no_sources and "sources" in result and result["sources"]:
                    sources_text = ", ".join([
                        f"{s['name']} ({s['score']:.3f})"
                        for s in result["sources"][:3]
                    ])
                    console.print(f"[dim]参照: {sources_text}[/dim]")

                console.print()  # 空行を追加

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Ctrl+Cが押されました。[/yellow]")
                console.print(
                    "[yellow]終了するには 'exit' を入力してください。[/yellow]\n"
                )
                continue

        logger.info("チャットモードが終了しました")

    except RAGEngineError as e:
        console.print(f"\n[bold red]エラー:[/bold red] {str(e)}", style="red")
        logger.error(f"チャットエラー: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]チャットが中断されました[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(
            f"\n[bold red]予期しないエラーが発生しました:[/bold red] {str(e)}",
            style="red"
        )
        logger.exception("予期しないエラー")
        sys.exit(1)


@click.command()
@click.argument("query_text", type=str)
@click.option(
    "--top-k",
    "-k",
    type=int,
    default=5,
    help="検索する画像の最大数（デフォルト: 5）"
)
@click.option(
    "--show-path",
    is_flag=True,
    help="画像のフルパスを表示"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="詳細情報を表示"
)
def search_images(
    query_text: str,
    top_k: int,
    show_path: bool,
    verbose: bool
) -> None:
    """テキストクエリで画像を検索します

    指定されたテキストクエリに類似した画像を検索し、
    類似度スコアと共に表示します。

    Args:
        query_text: 検索クエリ文字列
        top_k: 検索する画像の最大数
        show_path: 画像のフルパスを表示するフラグ
        verbose: 詳細情報を表示するフラグ

    Example:
        $ rag search-images "犬の写真"
        $ rag search-images "sunset landscape" -k 10
        $ rag search-images "人物" --show-path -v
    """
    try:
        # 設定の読み込みとコンポーネントの初期化
        with console.status("[bold green]初期化中..."):
            config = get_config()
            vector_store = VectorStore(config)
            vector_store.initialize()
            
            vision_embeddings = VisionEmbeddings(config)

        # クエリの表示
        console.print(Panel(
            f"[bold cyan]検索クエリ:[/bold cyan] {query_text}",
            border_style="cyan"
        ))

        # クエリの埋め込み生成
        with console.status(
            "[bold green]クエリの埋め込みを生成中..."
        ):
            # テキストクエリから埋め込みを生成
            # vision_embeddingsではテキストからの埋め込み生成をサポートしていないため、
            # テキスト埋め込みを使用してキャプションベースの検索を行う
            from ..rag.embeddings import EmbeddingGenerator
            embedding_generator = EmbeddingGenerator(config)
            query_embedding = embedding_generator.embed_query(query_text)

        # 画像検索の実行
        with console.status(
            f"[bold green]'{query_text}'で画像を検索中（最大{top_k}件）..."
        ):
            search_results = vector_store.search_images(
                query_embedding=query_embedding,
                top_k=top_k
            )

        # 結果の表示
        if not search_results:
            console.print(
                "[yellow]検索結果が見つかりませんでした。[/yellow]"
            )
            console.print("\n[dim]ヒント:[/dim]")
            console.print("  • 画像を追加してください: rag add <image_path>")
            console.print("  • 異なる検索クエリを試してください")
            return

        console.print(
            f"\n[bold green]検索結果: {len(search_results)}件[/bold green]\n"
        )

        # 検索結果をテーブル表示
        results_table = Table(show_header=True, header_style="bold magenta")
        results_table.add_column("#", justify="right", style="cyan", width=4)
        results_table.add_column("ファイル名", style="green")
        results_table.add_column("タイプ", style="magenta", width=8)
        results_table.add_column("類似度", justify="right", style="yellow", width=10)
        results_table.add_column("キャプション", style="dim", max_width=50)

        if show_path:
            results_table.add_column("パス", style="blue", max_width=40)

        for i, result in enumerate(search_results, 1):
            # キャプションを短縮
            caption = result['caption']
            if len(caption) > 50:
                caption = caption[:47] + "..."

            row_data = [
                str(i),
                result['file_name'],
                result['image_type'].upper(),
                f"{result['score']:.4f}",
                caption
            ]

            if show_path:
                row_data.append(result['file_path'])

            results_table.add_row(*row_data)

        console.print(results_table)

        # 詳細情報の表示
        if verbose:
            console.print("\n[bold]検索結果の詳細:[/bold]\n")

            for i, result in enumerate(search_results, 1):
                panel_content = (
                    f"[bold cyan]ファイル:[/bold cyan] {result['file_name']}\n"
                    f"[bold cyan]タイプ:[/bold cyan] {result['image_type'].upper()}\n"
                    f"[bold cyan]パス:[/bold cyan] {result['file_path']}\n"
                    f"[bold cyan]類似度:[/bold cyan] {result['score']:.4f}\n"
                    f"[bold cyan]作成日時:[/bold cyan] {result['created_at']}\n\n"
                    f"[bold cyan]キャプション:[/bold cyan]\n{result['caption']}"
                )

                # タグがあれば表示
                if result.get('tags'):
                    tags_str = ", ".join(result['tags'])
                    panel_content += f"\n\n[bold cyan]タグ:[/bold cyan] {tags_str}"

                console.print(Panel(
                    panel_content,
                    title=f"[bold green]{i}. {result['file_name']}",
                    border_style="blue",
                    padding=(1, 2)
                ))

        # 統計情報
        if verbose:
            console.print(f"\n[cyan]統計情報:[/cyan]")
            console.print(f"  検索結果数: {len(search_results)}")
            if search_results:
                avg_score = sum(r['score'] for r in search_results) / len(search_results)
                console.print(f"  平均類似度: {avg_score:.4f}")
                console.print(f"  最高類似度: {search_results[0]['score']:.4f}")
                console.print(f"  最低類似度: {search_results[-1]['score']:.4f}")

        logger.info(f"画像検索が完了しました: '{query_text[:50]}' -> {len(search_results)}件")

    except VisionEmbeddingError as e:
        console.print(f"[bold red]エラー:[/bold red] {str(e)}", style="red")
        console.print("\n[yellow]ヒント:[/yellow]")
        console.print("  1. Ollamaが起動しているか確認してください")
        console.print("  2. ビジョンモデルがインストールされているか確認してください")
        console.print("     $ ollama pull llava")
        logger.error(f"ビジョン埋め込みエラー: {str(e)}")
        sys.exit(1)

    except VectorStoreError as e:
        console.print(f"[bold red]エラー:[/bold red] {str(e)}", style="red")
        logger.error(f"ベクトルストアエラー: {str(e)}")
        sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]検索が中断されました[/yellow]")
        sys.exit(0)

    except Exception as e:
        console.print(
            f"[bold red]予期しないエラーが発生しました:[/bold red] {str(e)}",
            style="red"
        )
        if verbose:
            logger.exception("予期しないエラー")
        sys.exit(1)
