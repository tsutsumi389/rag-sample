"""マルチモーダルRAGエンジンモジュール

このモジュールはテキストと画像を統合的に扱うマルチモーダルRAGエンジンを実装します。
テキストと画像の両方を含む検索、画像を含む質問応答、マルチモーダルチャット機能を提供します。
"""

import base64
import logging
from pathlib import Path
from typing import Optional

import ollama

from ..models.document import ChatHistory, ChatMessage, SearchResult
from ..utils.config import Config, get_config
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore
from .vision_embeddings import VisionEmbeddings

logger = logging.getLogger(__name__)


class MultimodalRAGEngineError(Exception):
    """マルチモーダルRAGエンジンエラー"""
    pass


class MultimodalRAGEngine:
    """テキストと画像を統合的に扱うRAGエンジン

    マルチモーダルLLM（Gemma3など）を使用して、テキストと画像の両方を
    理解し、質問に答えます。画像検索、マルチモーダル検索、
    画像を含む質問応答をサポートします。

    Attributes:
        config: アプリケーション設定
        vector_store: ベクトルストアインスタンス
        text_embeddings: テキスト埋め込み生成器
        vision_embeddings: ビジョン埋め込み生成器
        llm_model: マルチモーダルLLMモデル名
        ollama_base_url: Ollama APIのベースURL
        ollama_client: Ollama Clientインスタンス
        chat_history: チャット履歴（チャットモード用）
    """

    # デフォルトのプロンプトテンプレート
    DEFAULT_SYSTEM_PROMPT = """あなたは親切で知識豊富なアシスタントです。
テキストと画像の両方を理解し、ユーザーの質問に正確に答えてください。
与えられたコンテキスト情報（テキストと画像）に基づいて回答してください。
コンテキストに情報がない場合は、正直にそう伝えてください。"""

    DEFAULT_QA_TEMPLATE = """コンテキスト情報:
{context}

質問: {question}

上記のコンテキスト情報（テキストと画像を含む）に基づいて質問に答えてください。
コンテキストに関連情報がない場合は、「提供された情報では回答できません」と答えてください。

回答:"""

    def __init__(
        self,
        config: Optional[Config] = None,
        vector_store: Optional[VectorStore] = None,
        text_embeddings: Optional[EmbeddingGenerator] = None,
        vision_embeddings: Optional[VisionEmbeddings] = None,
        llm_model: Optional[str] = None,
        ollama_base_url: Optional[str] = None,
        max_chat_history: Optional[int] = 10
    ):
        """マルチモーダルRAGエンジンの初期化

        Args:
            config: アプリケーション設定（省略時はデフォルト設定を使用）
            vector_store: ベクトルストアインスタンス（省略時は新規作成）
            text_embeddings: テキスト埋め込み生成器（省略時は新規作成）
            vision_embeddings: ビジョン埋め込み生成器（省略時は新規作成）
            llm_model: マルチモーダルLLMモデル名（省略時は設定ファイルの値を使用）
            ollama_base_url: Ollama APIのベースURL（省略時は設定ファイルの値を使用）
            max_chat_history: チャット履歴の最大メッセージ数

        Raises:
            MultimodalRAGEngineError: 初期化に失敗した場合
        """
        self.config = config or get_config()

        # ベクトルストアの設定
        self.vector_store = vector_store or VectorStore(self.config)

        # テキスト埋め込み生成器の設定
        self.text_embeddings = text_embeddings or EmbeddingGenerator(self.config)

        # ビジョン埋め込み生成器の設定
        self.vision_embeddings = vision_embeddings or VisionEmbeddings(self.config)

        # マルチモーダルLLMの設定
        self.llm_model = llm_model or getattr(
            self.config, 'ollama_multimodal_llm_model', 'gemma3'
        )
        self.ollama_base_url = ollama_base_url or self.config.ollama_base_url

        try:
            self.ollama_client = ollama.Client(host=self.ollama_base_url)
            # モデルの存在確認
            self._verify_model()
            logger.info(
                f"MultimodalRAGEngine initialized with model '{self.llm_model}' "
                f"at {self.ollama_base_url}"
            )
        except Exception as e:
            raise MultimodalRAGEngineError(
                f"Failed to initialize Ollama client: {e}\n"
                f"Make sure Ollama is running at {self.ollama_base_url} and "
                f"model '{self.llm_model}' is available (run: ollama pull {self.llm_model})."
            ) from e

        # チャット履歴の初期化
        self.chat_history = ChatHistory(max_messages=max_chat_history)

    def _verify_model(self) -> None:
        """マルチモーダルモデルの存在を確認する

        Raises:
            MultimodalRAGEngineError: モデルが利用できない場合
        """
        try:
            models = self.ollama_client.list()
            model_list = models.get('models', [])
            model_names = []
            for model in model_list:
                name = model.get('name') or model.get('model')
                if name:
                    # タグ付きの完全な名前も、タグなしのベース名も保存
                    model_names.append(name)
                    model_names.append(name.split(':')[0])

            # self.llm_modelも同様にタグあり/なし両方でチェック
            llm_model_base = self.llm_model.split(':')[0]
            if self.llm_model not in model_names and llm_model_base not in model_names:
                raise MultimodalRAGEngineError(
                    f"Multimodal LLM model '{self.llm_model}' is not available. "
                    f"Please run: ollama pull {self.llm_model}"
                )
        except ollama.ResponseError as e:
            raise MultimodalRAGEngineError(
                f"Failed to verify model '{self.llm_model}': {e}"
            ) from e

    def _encode_image_base64(self, image_path: str | Path) -> str:
        """画像ファイルをBase64エンコードする

        Args:
            image_path: 画像ファイルのパス

        Returns:
            str: Base64エンコードされた画像データ

        Raises:
            MultimodalRAGEngineError: 画像の読み込みまたはエンコードに失敗した場合
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            with open(path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            raise MultimodalRAGEngineError(
                f"Failed to encode image '{image_path}': {e}"
            ) from e

    def search_images(
        self,
        query: str,
        top_k: int = 5,
        collection_name: str = "images"
    ) -> list[SearchResult]:
        """テキストクエリで画像を検索

        テキストクエリを画像埋め込み空間に変換し、類似する画像を検索します。

        Args:
            query: 検索クエリ文字列
            top_k: 返す結果の最大数
            collection_name: 検索対象のコレクション名

        Returns:
            SearchResultオブジェクトのリスト（類似度の高い順）

        Raises:
            MultimodalRAGEngineError: 検索に失敗した場合
        """
        if not query.strip():
            raise MultimodalRAGEngineError("検索クエリが空です")

        try:
            logger.info(f"画像を検索中: '{query[:50]}...'")

            # テキストクエリを画像埋め込み空間に変換
            # Note: ビジョンモデルはテキストからも埋め込みを生成できる必要があります
            # （llavaやbakllavaはテキスト入力もサポート）
            # ここではテキスト埋め込みを使用する代替実装も可能
            query_embedding = self.text_embeddings.embed_query(query)

            # ベクトルストアで画像を検索
            search_results = self.vector_store.search_images(
                query_embedding=query_embedding,
                top_k=top_k,
                collection_name=collection_name
            )

            logger.info(f"{len(search_results)}件の画像を取得しました")
            return search_results

        except Exception as e:
            error_msg = f"画像の検索に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise MultimodalRAGEngineError(error_msg) from e

    def search_multimodal(
        self,
        query: str,
        top_k: int = 5,
        text_weight: Optional[float] = None,
        image_weight: Optional[float] = None
    ) -> list[SearchResult]:
        """テキストと画像の両方を検索してマージ

        テキストと画像の両方のコレクションから検索し、
        重み付けしたスコアで統合します。

        Args:
            query: 検索クエリ文字列
            top_k: 返す結果の最大数（テキストと画像の合計）
            text_weight: テキスト検索結果の重み（0.0-1.0）
            image_weight: 画像検索結果の重み（0.0-1.0）

        Returns:
            SearchResultオブジェクトのリスト（スコアの高い順にマージ）

        Raises:
            MultimodalRAGEngineError: 検索に失敗した場合
        """
        if not query.strip():
            raise MultimodalRAGEngineError("検索クエリが空です")

        # デフォルトの重みを設定
        if text_weight is None:
            text_weight = getattr(
                self.config, 'multimodal_search_text_weight', 0.5
            )
        if image_weight is None:
            image_weight = getattr(
                self.config, 'multimodal_search_image_weight', 0.5
            )

        try:
            logger.info(
                f"マルチモーダル検索を実行中: '{query[:50]}...' "
                f"(text_weight={text_weight}, image_weight={image_weight})"
            )

            # テキストクエリの埋め込みを生成
            query_embedding = self.text_embeddings.embed_query(query)

            # テキストコレクションを検索
            text_results = []
            try:
                text_results = self.vector_store.search(
                    query_embedding=query_embedding,
                    n_results=top_k
                )
                # テキスト結果のスコアに重みを適用
                for result in text_results:
                    result.score *= text_weight
                    result.result_type = 'text'
                logger.info(f"{len(text_results)}件のテキスト検索結果を取得")
            except Exception as e:
                logger.warning(f"テキスト検索に失敗: {e}")

            # 画像コレクションを検索
            image_results = []
            try:
                image_results = self.vector_store.search_images(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    collection_name="images"
                )
                # 画像結果のスコアに重みを適用
                for result in image_results:
                    result.score *= image_weight
                    result.result_type = 'image'
                logger.info(f"{len(image_results)}件の画像検索結果を取得")
            except Exception as e:
                logger.warning(f"画像検索に失敗: {e}")

            # 結果をマージしてスコア順にソート
            all_results = text_results + image_results
            all_results.sort(key=lambda x: x.score, reverse=True)

            # top_k件まで返す
            final_results = all_results[:top_k]

            logger.info(
                f"マルチモーダル検索完了: {len(final_results)}件の結果 "
                f"(テキスト: {len([r for r in final_results if r.result_type == 'text'])}, "
                f"画像: {len([r for r in final_results if r.result_type == 'image'])})"
            )

            return final_results

        except Exception as e:
            error_msg = f"マルチモーダル検索に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise MultimodalRAGEngineError(error_msg) from e

    def query_with_images(
        self,
        query: str,
        image_paths: Optional[list[str | Path]] = None,
        n_results: int = 5,
        chat_history: Optional[list[ChatMessage]] = None,
        include_sources: bool = True
    ) -> dict:
        """画像を含む質問に対して回答を生成

        テキストクエリと画像（オプション）を使用して、
        マルチモーダルLLMで回答を生成します。

        Args:
            query: ユーザーの質問
            image_paths: 質問に添付する画像のパスリスト（オプション）
            n_results: 検索する結果の最大数
            chat_history: 会話履歴のリスト（オプション）
            include_sources: 回答に情報源を含めるか

        Returns:
            回答情報を含む辞書:
                - answer: 生成された回答テキスト
                - sources: 情報源のリスト（include_sources=Trueの場合）
                - context_count: 使用したコンテキストの数
                - images_used: 使用した画像の数

        Raises:
            MultimodalRAGEngineError: 処理に失敗した場合
        """
        if not query.strip():
            raise MultimodalRAGEngineError("質問が空です")

        try:
            logger.info(f"マルチモーダルクエリを実行中: '{query[:50]}...'")

            # マルチモーダル検索で関連コンテンツを取得
            search_results = self.search_multimodal(
                query=query,
                top_k=n_results
            )

            # コンテキストの準備
            context_parts = []
            context_images = []

            if not search_results:
                context_parts.append("関連する情報が見つかりませんでした。")
            else:
                # 検索結果からテキストと画像を抽出
                for i, result in enumerate(search_results, 1):
                    if result.result_type == 'text':
                        context_parts.append(
                            f"[テキスト {i}] {result.document_name}\n{result.chunk.content}\n"
                        )
                    elif result.result_type == 'image':
                        context_parts.append(
                            f"[画像 {i}] {result.document_name}\n"
                            f"説明: {result.caption or 'N/A'}\n"
                        )
                        # 画像パスを収集
                        if result.image_path and Path(result.image_path).exists():
                            context_images.append(str(result.image_path))

            context_text = "\n".join(context_parts)

            # ユーザーが提供した画像を追加
            user_images = []
            if image_paths:
                for img_path in image_paths:
                    path = Path(img_path)
                    if path.exists():
                        user_images.append(str(path))
                    else:
                        logger.warning(f"画像ファイルが見つかりません: {img_path}")

            # すべての画像を統合
            all_images = user_images + context_images

            # プロンプトの構築
            prompt_parts = []

            # チャット履歴がある場合は追加
            if chat_history:
                prompt_parts.append("過去の会話:")
                for msg in chat_history:
                    prompt_parts.append(f"{msg.role}: {msg.content}")
                prompt_parts.append("")

            # コンテキストと質問
            prompt_parts.append(f"コンテキスト情報:\n{context_text}")
            prompt_parts.append(f"\n質問: {query}")
            prompt_parts.append(
                "\n上記のコンテキスト情報と画像に基づいて質問に答えてください。\n\n回答:"
            )

            prompt = "\n".join(prompt_parts)

            logger.debug(
                f"マルチモーダルプロンプト生成: "
                f"テキスト長={len(prompt)}, 画像数={len(all_images)}"
            )

            # Ollamaのchat APIを使用して回答を生成
            logger.info("マルチモーダルLLMで回答を生成中...")

            # メッセージの構築
            message_content = {
                'role': 'user',
                'content': prompt
            }

            # 画像がある場合は追加
            if all_images:
                message_content['images'] = all_images

            response = self.ollama_client.chat(
                model=self.llm_model,
                messages=[message_content]
            )

            answer = response.get('message', {}).get('content', '').strip()

            if not answer:
                raise MultimodalRAGEngineError("LLMからの応答が空です")

            # 結果の構築
            result = {
                "answer": answer,
                "context_count": len(search_results),
                "images_used": len(all_images)
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
                            "score": search_result.score,
                            "type": search_result.result_type
                        })
                        seen_sources.add(source_id)

                result["sources"] = sources

            logger.info(
                f"マルチモーダル回答を生成しました "
                f"（コンテキスト数: {len(search_results)}, 画像数: {len(all_images)}）"
            )
            return result

        except Exception as e:
            error_msg = f"マルチモーダルクエリの実行に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise MultimodalRAGEngineError(error_msg) from e

    def chat_multimodal(
        self,
        message: str,
        image_paths: Optional[list[str | Path]] = None,
        n_results: int = 3,
        include_sources: bool = True
    ) -> dict:
        """マルチモーダルチャット（履歴を保持）

        チャット形式で画像を含む質問応答を実行します。

        Args:
            message: ユーザーのメッセージ
            image_paths: メッセージに添付する画像のパスリスト（オプション）
            n_results: 検索する結果の最大数
            include_sources: 回答に情報源を含めるか

        Returns:
            回答情報を含む辞書:
                - answer: 生成された回答テキスト
                - sources: 情報源のリスト（include_sources=Trueの場合）
                - context_count: 使用したコンテキストの数
                - images_used: 使用した画像の数
                - history_length: 現在の履歴メッセージ数

        Raises:
            MultimodalRAGEngineError: 処理に失敗した場合
        """
        # ユーザーメッセージを履歴に追加
        metadata = {}
        if image_paths:
            metadata['image_paths'] = [str(p) for p in image_paths]

        self.chat_history.add_message(
            role="user",
            content=message,
            metadata=metadata
        )

        # マルチモーダルクエリを実行
        result = self.query_with_images(
            query=message,
            image_paths=image_paths,
            n_results=n_results,
            chat_history=self.chat_history.messages[:-1],  # 最後のメッセージ（現在）を除く
            include_sources=include_sources
        )

        # アシスタントメッセージを履歴に追加
        self.chat_history.add_message(
            role="assistant",
            content=result['answer'],
            metadata={
                "context_count": result.get('context_count', 0),
                "images_used": result.get('images_used', 0)
            }
        )

        # 履歴の長さを追加
        result['history_length'] = len(self.chat_history)

        logger.info(
            f"マルチモーダルチャット回答を生成しました "
            f"（履歴: {len(self.chat_history)}メッセージ）"
        )

        return result

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
        """マルチモーダルRAGエンジンの初期化（ベクトルストアの初期化）

        Raises:
            MultimodalRAGEngineError: 初期化に失敗した場合
        """
        try:
            logger.info("マルチモーダルRAGエンジンを初期化中...")
            self.vector_store.initialize()
            logger.info("マルチモーダルRAGエンジンの初期化が完了しました")
        except Exception as e:
            error_msg = f"マルチモーダルRAGエンジンの初期化に失敗しました: {str(e)}"
            logger.error(error_msg)
            raise MultimodalRAGEngineError(error_msg) from e

    def get_status(self) -> dict:
        """マルチモーダルRAGエンジンのステータス情報を取得

        Returns:
            ステータス情報の辞書:
                - multimodal_llm_model: 使用中のマルチモーダルLLMモデル名
                - text_embedding_model: 使用中のテキスト埋め込みモデル名
                - vision_embedding_model: 使用中のビジョン埋め込みモデル名
                - vector_store_info: ベクトルストア情報
                - chat_history_length: チャット履歴メッセージ数
        """
        try:
            vector_store_info = self.vector_store.get_collection_info()
        except Exception:
            vector_store_info = {"error": "ベクトルストアが初期化されていません"}

        return {
            "multimodal_llm_model": self.llm_model,
            "text_embedding_model": self.config.ollama_embedding_model,
            "vision_embedding_model": self.vision_embeddings.model_name,
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

    def __repr__(self) -> str:
        """文字列表現"""
        return (
            f"MultimodalRAGEngine("
            f"llm_model='{self.llm_model}', "
            f"vision_model='{self.vision_embeddings.model_name}', "
            f"base_url='{self.ollama_base_url}'"
            f")"
        )


# 便利関数: デフォルト設定でマルチモーダルRAGエンジンを作成
def create_multimodal_rag_engine(
    config: Optional[Config] = None,
    llm_model: Optional[str] = None
) -> MultimodalRAGEngine:
    """マルチモーダルRAGエンジンを作成

    Args:
        config: アプリケーション設定（省略時はデフォルト設定を使用）
        llm_model: マルチモーダルLLMモデル名（省略時は設定ファイルの値を使用）

    Returns:
        MultimodalRAGEngine: マルチモーダルRAGエンジンインスタンス

    Example:
        >>> engine = create_multimodal_rag_engine()
        >>> engine.initialize()
        >>> result = engine.query_with_images("この画像について説明して", ["image.jpg"])
    """
    return MultimodalRAGEngine(config=config, llm_model=llm_model)
