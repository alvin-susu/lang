import re

import gradio
from gradio.components.chatbot import ChatMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import PromptTemplate

from common.common_prompt_template import template_with_history
from common.common_utils import model_to_llm
from common.db_utils import get_vectordb
from serve.params.char_params import ChatParams


class QaChainSelf:
    """"
    不带历史记录的问答链
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - appid：星火需要输入
    - api_key：所有模型都需要
    - Spark_api_secret：星火秘钥
    - Wenxin_secret_key：文心秘钥
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥（智谱或者OpenAI）
    - template：可以自定义提示模板，没有输入则使用默认的提示模板default_template_rq
    """

    # 基于召回结果和 query 结合起来构建的 prompt使用的默认提示模版
    def __init__(self, chat_params: ChatParams):
        self.model = chat_params.model
        self.temperature = chat_params.temperature
        self.top_k = chat_params.top_k
        self.knowledge_file_path = chat_params.knowledge_file_path
        self.persist_path = chat_params.vector_db_path
        self.appid = chat_params.appid
        self.api_key = chat_params.api_key
        self.Spark_api_secret = chat_params.Spark_api_secret
        self.Wenxin_secret_key = chat_params.Wenxin_secret_key
        self.embedding_model_name = chat_params.embedding_model_name
        self.embedding_key = chat_params.embedding_key
        self.template = chat_params.prompt_template
        self.is_user_history = chat_params.is_use_history
        self.chat_history = chat_params.chat_history
        self.vectordb = get_vectordb(
            self.knowledge_file_path,
            self.persist_path,
            self.embedding_model_name,
            self.embedding_key
        )

        self.llm = model_to_llm(
            self.model,
            self.temperature,
            self.appid,
            self.api_key,
            self.Spark_api_secret,
            self.Wenxin_secret_key
        )
        self.retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={'k': self.top_k}
        )

        # 自定义的QA链
        self.QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=chat_params.input_variables,
            template=self.template,
        )
        no_history_documents_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.QA_CHAIN_PROMPT
        )
        self.qa_chain = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=no_history_documents_chain
        )

        # 带有上下文的自定义的QA链
        self.QA_CHAIN_PROMPT_WITH_HISTORY = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=template_with_history,
        )
        with_history_documents_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.QA_CHAIN_PROMPT_WITH_HISTORY
        )
        self.qa_chain_with_history = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=with_history_documents_chain
        )

    def answer(self, question: str, temperature=None, top_k=4):
        """
        核心方法 调用问答链
        :param question: 问题
        :param temperature: 温度系数
        :param top_k: topK
        :return: answer
        """
        if len(question) == 0:
            raise Exception("问题不能为空，请输入问题后提问")

        if temperature is None:
            temperature = self.temperature

        if top_k is None:
            top_k = self.top_k

        # 需要带历史记录则将is_user_history置为True
        print(f"提问的question为{question}")
        if self.is_user_history:
            print(f"使用历史记录回答问题")
            result = self.qa_chain_with_history.invoke(
                {"question": question, "context": "context"})
        else:
            print(f"不使用历史记录回答问题")
            result = self.qa_chain.invoke({"question": question, "input": question, "context": "context"})
        print(f"模型回答的结果为:{result}")
        user_message = gradio.ChatMessage(role="user", content=result['question'])
        assistant_message = gradio.ChatMessage(role="assistant", content=result['answer'])
        chatbot_response = [user_message, assistant_message]
        return chatbot_response

    def clear_history(self):
        return self.chat_history.clear()

    def change_history_length(self, history_round_num: int = 1):
        return self.chat_history[len(self.chat_history) - history_round_num:]

