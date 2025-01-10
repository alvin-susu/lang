from common.common_constants import OPEN_API_STR, KNOWLEDGE_DB_PATH, VECTOR_DB_PATH
from qa_chain.qa_chain_self import QaChainSelf
from serve.params.char_params import ChatParams


class ModelComponent:
    """
    存储问答对象
    """

    def __init__(self):
        self.chain = {}

    def chat(self,
             question: str,
             chat_history: list = None,
             model: str = OPEN_API_STR,
             embedding: str = OPEN_API_STR,
             temperature: float = 0.0,
             top_k: int = 4,
             is_user_history: bool = False,
             knowledge_file_path: str = KNOWLEDGE_DB_PATH,
             vector_db_path: str = VECTOR_DB_PATH,
             history_len: int = 3):
        print(
            f"chat函数 入参: question: {question},"
            f" chat_history: {chat_history},"
            f" model: {model},"
            f" embedding: {embedding},"
            f" temperature: {temperature},"
            f" top_k: {top_k},"
            f" is_user_history: {is_user_history},"
            f" knowledge_file_path: {knowledge_file_path},"
            f" vector_db_path: {vector_db_path},"
            f"history_len: {history_len}")

        if question is None or len(question) == 0:
            return "please input question", chat_history

        try:
            if (model, embedding) not in self.chain:
                chat_params = ChatParams(
                    question=question,
                    model=model,
                    temperature=temperature,
                    top_k=top_k,
                    is_use_history=is_user_history,
                    chat_history=chat_history,
                    knowledge_file_path=knowledge_file_path,
                    vector_db_path=vector_db_path,
                )
                qa_chain = QaChainSelf(chat_params)
                print(f"qa_chain: {qa_chain}")
                answer = qa_chain.answer(question=question, temperature=temperature, top_k=top_k)
                print("answer: ", answer)
                print(type(answer))
                return answer
        except Exception as e:
            print(f"出现异常:{str(e)}")
            return str(e), chat_history

    def clear_history(self):
        if len(self.chain) > 0:
            for item in self.chain.values():
                item.clear_history()
