from langchain_openai import OpenAIEmbeddings

from common.common_constants import KNOWLEDGE_DB_PATH, VECTOR_DB_PATH
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from common.db_utils import get_vectordb

if __name__ == '__main__':
    # vector_db = create_db(KNOWLEDGE_DB_PATH,
    #           VECTOR_DB_PATH,
    #           "openai")
    # search = vector_db.similarity_search("agent-tutorial")
    # print(search)

    vector_db = get_vectordb(KNOWLEDGE_DB_PATH, VECTOR_DB_PATH)
    print(f"数据库加载成功，存储向量数量: {vector_db._collection.count()}")
    search = vector_db.similarity_search("什么是LLM")
    print(search)
    # # 配置路径
    # persist_path = r"D:\git_hub_project\Chat_with_Datawhale_langchain\vector_db\chroma"
    # embedding_model = OpenAIEmbeddings()
    #
    # # 加载数据库
    # try:
    #     vectordb = Chroma(persist_directory=persist_path, embedding_function=embedding_model)
    #     print(f"数据库加载成功，存储向量数量: {vectordb._collection.count()}")
    # except Exception as e:
    #     print("加载数据库时出错:", str(e))

    # 测试生成嵌入
    # text = ["测试向量生成"]
    # embeddings = get_embedding(embedding_model_name="openai")
    # embeddings = embeddings.embed_documents(text)
    # print("生成的嵌入:", embeddings)
