import os
import re

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyMuPDFLoader, UnstructuredFileLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.common_constants import KNOWLEDGE_DB_PATH, VECTOR_DB_PATH
from llm.wen_xin_llm import WenXin_LLM
from llm.spark_llm import Spark_LLM
from llm.zhi_pu_llm import ZhiPu_LLM
from common.common_enums import GptModelName, ErnieBotModelName, SparkModelName, ChatGlmModelName
from dotenv import load_dotenv, find_dotenv


def model_to_llm(model_name: str = None, temperature: float = 0.0, appid: str = None, api_key: str = None,
                 spark_api_secret: str = None, wenxin_secret_key: str = None):
    """
    根据模型名称转为LLM模型
    :param model_name: 模型名称 具体参考 common_enums.py
    :param temperature: 温度系数
    :param appid: 星火需要输入
    :param api_key: 调用api的密钥
    :param spark_api_secret: 星火api密钥
    :param wenxin_secret_key: 文心一言密钥
    :return: LLM模型
    """
    if model_name in [item.value for item in GptModelName]:
        api_key, = __get_api_keys_for_model("openai", api_key)
        return ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key)

    elif model_name in [item.value for item in ErnieBotModelName]:
        api_key, wenxin_secret_key = __get_api_keys_for_model("wenxin", api_key, (wenxin_secret_key,))
        return WenXin_LLM(model_name=model_name, temperature=temperature, api_key=api_key, secret_key=wenxin_secret_key)

    elif model_name in [item.value for item in SparkModelName]:
        api_key, appid, spark_api_secret = __get_api_keys_for_model("spark", api_key, (appid, spark_api_secret))
        return Spark_LLM(model_name=model_name, temperature=temperature, appid=appid, api_secret=spark_api_secret,
                         api_key=api_key)

    elif model_name in [item.value for item in ChatGlmModelName]:
        api_key, = __get_api_keys_for_model("zhipuai", api_key)
        # return ZhiPu_LLM(model_name=model, zhipuai_api_key=api_key, temperature=temperature)

    else:
        raise ValueError(f"Model {model_name} is not supported!!!")


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
            create_db(file_path, persist_path, embedding_model_name)
            vectordb = load_knowledge_db(persist_path, embeddings)
        else:
            vectordb = load_knowledge_db(persist_path, embeddings)
    else:
        create_db(file_path, persist_path, embedding_model_name)
        vectordb = load_knowledge_db(persist_path, embeddings)

    return vectordb


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


def get_embedding(embedding_model_name: str, embedding_model_key: str = None):
    """
    加载模型的Embedding
    :param embedding_model_name: 加载对应模型的Embedding
    :param embedding_model_key: 模型的key
    :return: Embedding
    """
    if embedding_model_name == 'm3e':
        print("加载 M3E 模型...")
        return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    if embedding_model_key is None:
        embedding_model_key = __parse_llm_api_key(embedding_model_name)
    if embedding_model_name == "openai":
        print("加载 OpenAI 模型...")
        return OpenAIEmbeddings()
    elif embedding_model_name == "zhipuai":
        print("加载 ZhipuAI 模型...")
        return ZhipuAIEmbeddings(api_key=embedding_model_key)
    else:
        raise ValueError(f"embedding {embedding_model_name} not support ")

def __get_api_keys_for_model(model_name: str, api_key: str, additional_keys: tuple = None):
    """
    根据环境变量获取模型对应的key
    :param model_name: 模型名称
    :param api_key:
    :param additional_keys:
    :return:
    """
    if api_key is None:
        keys = __parse_llm_api_key(model_name)
        if additional_keys:
            return keys[:len(additional_keys)]  # Extract the needed number of keys
        return keys
    return (api_key,) + additional_keys if additional_keys else (api_key,)


def __parse_llm_api_key(model: str):
    env_file = None
    """
    通过 model 和 env_file 的来解析平台参数
    """
    if env_file is None:
        _ = load_dotenv(find_dotenv())
        env_file = dict(os.environ)
    if model == "openai":
        return env_file["OPENAI_API_KEY"]
    elif model == "wenxin":
        return env_file["wenxin_api_key"], env_file["wenxin_secret_key"]
    elif model == "spark":
        return env_file["spark_api_key"], env_file["spark_appid"], env_file["spark_api_secret"]
    elif model == "zhipuai":
        return get_from_dict_or_env(env_file, "zhipuai_api_key", "ZHIPUAI_API_KEY")
    else:
        raise ValueError(f"model{model} not support!!!")


def file_loader(dir_path):
    """
    生产文件加载器
    :param dir_path: 加载文件的上层目录
    :return: loaders 加载器
    """
    # 检查目录是否存在
    if not os.path.isdir(dir_path):
        raise ValueError(f"指定的路径不是一个有效的目录: {dir_path}")

    # 查询目录下所有文件名
    files_name = os.listdir(dir_path)

    loaders = []
    # 获取文档类型
    for file_name in files_name:
        # 构建完整文件加载路径
        full_path = os.path.join(dir_path, file_name)

        if os.path.isfile(full_path) is False:
            # 只加载目录下的文件
            continue

        # 后缀全部转小写 统一格式
        file_type = file_name.split(".")[-1].lower()

        # 如果加载失败则全部失败，若想要不阻断可自行添加try catch
        if file_type == "pdf":
            loaders.append(PyMuPDFLoader(full_path))
        elif file_type == 'md':
            pattern = r"不存在|风控"
            match = re.search(pattern, full_path)
            if not match:
                loaders.append(UnstructuredMarkdownLoader(full_path))
        elif file_type == 'txt':
            loaders.append(UnstructuredFileLoader(full_path))

    return loaders


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
