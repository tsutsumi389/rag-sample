"""RAGエンジンモジュール

このモジュールは検索・質問応答のコアロジックを実装します。
コンテキスト検索、プロンプト生成、LLMによる回答生成、チャット履歴管理などの機能を提供します。
"""

import logging
from typing import Optional

from langchain_ollama import OllamaLLM

from ..models.document import ChatHistory, SearchResult
from ..utils.config import Config, get_config
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGEngineError(Exception):
    """RAGエンジンエラー"""
    pass


class RAGEngine:
    """RAG（Retrieval-Augmented Generation）エンジンクラス

    ベクトルストアから関連ドキュメントを検索し、
    LLMを使用して質問に対する回答を生成します。
    チャット形式での対話にも対応しています。

    Attributes:
        config: アプリケーション設定
        vector_store: ベクトルストアインスタンス
        embedding_generator: 埋め込み生成器インスタンス
        llm: LangChain OllamaLLMインスタンス
        chat_history: チャット履歴（チャットモード用）
    """

    # デフォルトのプロンプトテンプレート
    DEFAULT_SYSTEM_PROMPT = """あなたは親切で知識豊富なアシスタントです。
与えられたコンテキスト情報に基づいて、ユーザーの質問に正確に答えてください。
コンテキストに情報がない場合は、正直にそう伝えてください。"""

    DEFAULT_QA_TEMPLATE = """コンテキスト情報:
{context}

質問: {question}

上記のコンテキスト情報に基づいて質問に答えてください。
コンテキストに関連情報がない場合は、「提供された情報では回答できません」と答えてください。

回答:"""

    def __init__(
        self,
        config: Optional[Config] = None,
        vector_store: Optional[VectorStore] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        llm_model: Optional[str] = None,
        max_chat_history: Optional[int] = 10
    ):
        """RAGエンジンの初期化

        Args:
            config: アプリケーション設定（省略時はデフォルト設定を使用）
            vector_store: ベクトルストアインスタンス（省略時は新規作成）
            embedding_generator: 埋め込み生成器（省略時は新規作成）
            llm_model: LLMモデル名（省略時は設定ファイルの値を使用）
            max_chat_history: チャット履歴の最大メッセージ数

        Raises:
            RAGEngineError: 初期化に失敗した場合
        """
        self.config = config or get_config()

        # ベクトルストアの設定
        self.vector_store = vector_store or VectorStore(self.config)

        # 埋め込み生成器の設定
        self.embedding_generator = embedding_generator or EmbeddingGenerator(
            self.config
        )

        # LLMの設定
        llm_model_name = llm_model or self.config.ollama_llm_model
        try:
            self.llm = OllamaLLM(
                model=llm_model_name,
                base_url=self.config.ollama_base_url
            )
            logger.info(f"LLMを初期化しました: {llm_model_name}")
        except Exception as e:
            raise RAGEngineError(
                f"LLMの初期化に失敗しました: {str(e)}\n"
                f"Ollamaが{self.config.ollama_base_url}で起動しており、"
                f"モデル'{llm_model_name}'が利用可能か確認してください。"
            ) from e

        # チャット履歴の初期化
        self.chat_history = ChatHistory(max_messages=max_chat_history)

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        where: Optional[dict] = None
    ) -> list[SearchResult]:
        """クエリに関連するドキュメントを検索

        Args:
            query: 検索クエリ文字列
            n_results: 返す結果の最大数
            where: メタデータフィルタ（例: {"document_id": "doc123"}）

        Returns:
            SearchResultオブジェクトのリスト（類似度の高い順）

        Raises:
            RAGEngineError: 検索に失敗した場合
        """
        if not query.strip():
            raise RAGEngineError("検索クエリが空です")

        try:
            logger.info(f"ドキュメントを検索中: '{query[:50]}...'")

            # クエリを埋め込みベクトルに変換
            query_embedding = self.embedding_generator.embed_query(query)

            # ベクトルストアで検索
            search_results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=n_results,
                where=where
            )

            logger.info(f"{len(search_results)}件の関連ドキュメントを取得しました")
            return search_results

        except Exception as e:
            error_msg = f"ドキュメントの検索に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise RAGEngineError(error_msg) from e

    def generate_answer(
        self,
        question: str,
        context_results: list[SearchResult],
        system_prompt: Optional[str] = None,
        qa_template: Optional[str] = None,
        include_sources: bool = True
    ) -> dict:
        """検索結果を使用してLLMで回答を生成

        Args:
            question: ユーザーの質問
            context_results: 検索結果のリスト
            system_prompt: システムプロンプト（省略時はデフォルト）
            qa_template: Q&Aプロンプトテンプレート（省略時はデフォルト）
            include_sources: 回答に情報源を含めるか

        Returns:
            回答情報を含む辞書:
                - answer: 生成された回答テキスト
                - sources: 情報源のリスト（include_sources=Trueの場合）
                - context_count: 使用したコンテキストの数

        Raises:
            RAGEngineError: 回答生成に失敗した場合
        """
        if not question.strip():
            raise RAGEngineError("質問が空です")

        try:
            # コンテキストの準備
            if not context_results:
                logger.warning("コンテキストが空です。一般的な回答を生成します。")
                context_text = "関連する情報が見つかりませんでした。"
            else:
                # 検索結果からテキストを抽出
                context_parts = []
                for i, result in enumerate(context_results, 1):
                    context_parts.append(
                        f"[{i}] {result.document_name}\n{result.chunk.content}\n"
                    )
                context_text = "\n".join(context_parts)

            # プロンプトの構築
            template = qa_template or self.DEFAULT_QA_TEMPLATE
            prompt = template.format(context=context_text, question=question)

            logger.debug(f"生成プロンプト:\n{prompt[:200]}...")

            # LLMで回答を生成
            logger.info("LLMで回答を生成中...")
            answer = self.llm.invoke(prompt)

            # 結果の構築
            result = {
                "answer": answer,
                "context_count": len(context_results)
            }

            # 情報源の追加
            if include_sources and context_results:
                sources = []
                seen_sources = set()

                for search_result in context_results:
                    source_id = search_result.document_source
                    if source_id not in seen_sources:
                        sources.append({
                            "name": search_result.document_name,
                            "source": search_result.document_source,
                            "score": search_result.score
                        })
                        seen_sources.add(source_id)

                result["sources"] = sources

            logger.info(f"回答を生成しました（コンテキスト数: {len(context_results)}）")
            return result

        except Exception as e:
            error_msg = f"回答の生成に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise RAGEngineError(error_msg) from e

    def query(
        self,
        question: str,
        n_results: int = 5,
        where: Optional[dict] = None,
        include_sources: bool = True
    ) -> dict:
        """質問に対して検索と回答生成を一度に実行

        Args:
            question: ユーザーの質問
            n_results: 検索する結果の最大数
            where: メタデータフィルタ
            include_sources: 回答に情報源を含めるか

        Returns:
            回答情報を含む辞書（generate_answerと同じ形式）

        Raises:
            RAGEngineError: 処理に失敗した場合
        """
        # 関連ドキュメントを検索
        search_results = self.retrieve(
            query=question,
            n_results=n_results,
            where=where
        )

        # 回答を生成
        result = self.generate_answer(
            question=question,
            context_results=search_results,
            include_sources=include_sources
        )

        return result

    def chat(
        self,
        message: str,
        n_results: int = 3,
        where: Optional[dict] = None,
        include_sources: bool = True
    ) -> dict:
        """チャット形式で質問応答を実行（履歴を保持）

        Args:
            message: ユーザーのメッセージ
            n_results: 検索する結果の最大数
            where: メタデータフィルタ
            include_sources: 回答に情報源を含めるか

        Returns:
            回答情報を含む辞書:
                - answer: 生成された回答テキスト
                - sources: 情報源のリスト（include_sources=Trueの場合）
                - context_count: 使用したコンテキストの数
                - history_length: 現在の履歴メッセージ数

        Raises:
            RAGEngineError: 処理に失敗した場合
        """
        # ユーザーメッセージを履歴に追加
        self.chat_history.add_message(role="user", content=message)

        # 関連ドキュメントを検索
        search_results = self.retrieve(
            query=message,
            n_results=n_results,
            where=where
        )

        # コンテキストの準備
        if not search_results:
            context_text = "関連する情報が見つかりませんでした。"
        else:
            context_parts = []
            for i, result in enumerate(search_results, 1):
                context_parts.append(
                    f"[{i}] {result.document_name}\n{result.chunk.content}\n"
                )
            context_text = "\n".join(context_parts)

        # チャット履歴を含むプロンプトを構築
        prompt_parts = [self.DEFAULT_SYSTEM_PROMPT]

        # 過去の会話履歴を追加（直近の数ターンのみ）
        if len(self.chat_history) > 1:
            prompt_parts.append("\n過去の会話:")
            for msg in self.chat_history.messages[:-1]:  # 最後のメッセージ（現在）を除く
                prompt_parts.append(f"{msg.role}: {msg.content}")

        # 現在の質問とコンテキスト
        prompt_parts.append(f"\nコンテキスト情報:\n{context_text}")
        prompt_parts.append(f"\n質問: {message}")
        prompt_parts.append(
            "\n上記のコンテキスト情報と会話履歴に基づいて質問に答えてください。\n\n回答:"
        )

        prompt = "\n".join(prompt_parts)

        try:
            # LLMで回答を生成
            logger.info("チャットモードで回答を生成中...")
            answer = self.llm.invoke(prompt)

            # アシスタントメッセージを履歴に追加
            self.chat_history.add_message(
                role="assistant",
                content=answer,
                metadata={"context_count": len(search_results)}
            )

            # 結果の構築
            result = {
                "answer": answer,
                "context_count": len(search_results),
                "history_length": len(self.chat_history)
            }

            # 情報源の追加
            if include_sources and search_results:
                sources = []
                seen_sources = set()

                for search_result in search_results:
                    source_id = search_result.document_source
                    if source_id not in seen_sources:
                        sources.append({
                            "name": search_result.document_name,
                            "source": search_result.document_source,
                            "score": search_result.score
                        })
                        seen_sources.add(source_id)

                result["sources"] = sources

            logger.info(
                f"チャット回答を生成しました "
                f"（履歴: {len(self.chat_history)}メッセージ）"
            )
            return result

        except Exception as e:
            error_msg = f"チャット回答の生成に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise RAGEngineError(error_msg) from e

    def clear_chat_history(self) -> None:
        """チャット履歴をクリア"""
        self.chat_history.clear()
        logger.info("チャット履歴をクリアしました")

    def get_chat_history(self) -> list[dict[str, str]]:
        """チャット履歴を取得

        Returns:
            チャットメッセージの辞書のリスト
        """
        return self.chat_history.to_dicts()

    def initialize(self) -> None:
        """RAGエンジンの初期化（ベクトルストアの初期化）

        Raises:
            RAGEngineError: 初期化に失敗した場合
        """
        try:
            logger.info("RAGエンジンを初期化中...")
            self.vector_store.initialize()
            logger.info("RAGエンジンの初期化が完了しました")
        except Exception as e:
            error_msg = f"RAGエンジンの初期化に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise RAGEngineError(error_msg) from e

    def get_status(self) -> dict:
        """RAGエンジンのステータス情報を取得

        Returns:
            ステータス情報の辞書:
                - llm_model: 使用中のLLMモデル名
                - embedding_model: 使用中の埋め込みモデル名
                - vector_store_info: ベクトルストア情報
                - chat_history_length: チャット履歴メッセージ数
        """
        try:
            vector_store_info = self.vector_store.get_collection_info()
        except Exception:
            vector_store_info = {"error": "ベクトルストアが初期化されていません"}

        return {
            "llm_model": self.config.ollama_llm_model,
            "embedding_model": self.config.ollama_embedding_model,
            "vector_store_info": vector_store_info,
            "chat_history_length": len(self.chat_history)
        }

    def __enter__(self):
        """コンテキストマネージャーのエントリ"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキストマネージャーの終了"""
        self.vector_store.close()
        return False


# 便利関数: デフォルト設定でRAGエンジンを作成
def create_rag_engine(
    config: Optional[Config] = None,
    llm_model: Optional[str] = None
) -> RAGEngine:
    """RAGエンジンを作成

    Args:
        config: アプリケーション設定（省略時はデフォルト設定を使用）
        llm_model: LLMモデル名（省略時は設定ファイルの値を使用）

    Returns:
        RAGEngine: RAGエンジンインスタンス

    Example:
        >>> engine = create_rag_engine()
        >>> engine.initialize()
        >>> result = engine.query("質問内容")
    """
    return RAGEngine(config=config, llm_model=llm_model)
