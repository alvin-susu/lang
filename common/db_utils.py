import os

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.common_constants import KNOWLEDGE_DB_PATH, VECTOR_DB_PATH
from common.common_utils import get_embedding


def load_knowledge_db(path, embeddings):
    """
    该函数用于加载向量数据库。
    :param path: 要加载的向量数据库路径。
    :param embeddings: 向量数据库使用的 embedding 模型。
    :return vectordb: 加载的数据库。
    """
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    return vectordb


def create_db(know_files_path: str = KNOWLEDGE_DB_PATH,
              persist_path: str = VECTOR_DB_PATH,
              embeddings_model: str = "openai"):
    """
    加载PDF文件，切分文档，生产文档的嵌入向量，创建向量数据库
    :param persist_path: 保存db的路径你b'v
    :param know_files_path: 知识库路径
    :param embeddings_model: 生产嵌入的模型
    :return: db
    """
    if know_files_path is None:
        return "无法加载知识库文件，请检查路径是否正确！"

    # 加载文件生产loaders
    loaders = file_loader(know_files_path)
    # 通过 loaders 加载文档
    docs = [doc for loader in loaders for doc in loader.load()]
    # 切分文档
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    # 切分器切分文档
    split_docs = text_spliter.split_documents(docs)
    # 生成文本嵌入
    embeddings = get_embedding(embeddings_model)
    # 显式生成嵌入
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_path,
    )
    print(f"向量数据库创建完成，存储路径: {persist_path}")
    print(f"数据库中存储的向量数量: {vectordb._collection.count()}")

    return vectordb


def get_vectordb(file_path: str = None, persist_path: str = None, embedding_model_name="openai",
                 embedding_model_key: str = None):
    """
    返回向量数据库对象
    :param file_path: 知识库路径
    :param persist_path: 持久化数据库路径
    :param embedding_model_name: embedding模型的名称
    :param embedding_model_key: 调用模型所需要的key
    :return:
    """
    embeddings = get_embedding(embedding_model_name=embedding_model_name, embedding_model_key=embedding_model_key)
    if os.path.exists(persist_path):
        contents = os.listdir(persist_path)
        if len(contents) == 0:
            print(f"向量数据库不存在,需要新建向量数据库,路径为:{persist_path}")
            create_db(file_path, persist_path, embedding_model_name)
            vectordb = load_knowledge_db(persist_path, embeddings)
            print(f"向量数据库创建完毕")
        else:
            print(f"向量数据库已存在,加载路径为:{persist_path}")
            vectordb = load_knowledge_db(persist_path, embeddings)
            print(f"向量数据库加载完毕")
    else:
        print(f"向量数据库不存在,需要新建向量数据库,默认路径为:{persist_path}")
        create_db(file_path, persist_path, embedding_model_name)
        vectordb = load_knowledge_db(persist_path, embeddings)
        print(f"向量数据库使用默认路径创建完毕")

    return vectordb
