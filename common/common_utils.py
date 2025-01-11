import os
import re

from dotenv import load_dotenv, find_dotenv
from gradio.components.chatbot import ChatMessage
from langchain_community.document_loaders import UnstructuredMarkdownLoader, PyMuPDFLoader, UnstructuredFileLoader
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from common.common_constants import OPEN_API_STR, WEN_XIN_STR, SPARK_STR, ZHI_PU_AI_STR
from common.common_enums import GptModelName, ErnieBotModelName, SparkModelName, ChatGlmModelName
from llm.call_llm import get_completion
from llm.spark_llm import Spark_LLM
from llm.wen_xin_llm import WenXin_LLM


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
        print("加载openaiLLM模型")
        api_key, = __get_api_keys_for_model("openai", api_key)
        chat_open_ai = ChatOpenAI(model_name=model_name, temperature=temperature, openai_api_key=api_key)
        print("openaiLLM模型加载完毕")
        return chat_open_ai

    elif model_name in [item.value for item in ErnieBotModelName]:
        print("model_to_llm model_name为ErnieBotModelName")
        api_key, wenxin_secret_key = __get_api_keys_for_model("wenxin", api_key, (wenxin_secret_key,))
        wenxin_llm = WenXin_LLM(model_name=model_name, temperature=temperature, api_key=api_key,
                                secret_key=wenxin_secret_key)
        print("wenxinLLM模型加载完毕")
        return wenxin_llm

    elif model_name in [item.value for item in SparkModelName]:
        print("model_to_llm model_name为SparkModelName")
        api_key, appid, spark_api_secret = __get_api_keys_for_model("spark", api_key, (appid, spark_api_secret))
        spark_llm = Spark_LLM(model_name=model_name, temperature=temperature, appid=appid, api_secret=spark_api_secret,
                              api_key=api_key)
        print("sparkLLM模型加载完毕")
        return spark_llm

    elif model_name in [item.value for item in ChatGlmModelName]:
        print("model_to_llm model_name为ChatGlmModelName")
        api_key, = __get_api_keys_for_model("zhipuai", api_key)
        # return ZhiPu_LLM(model_name=model, zhipuai_api_key=api_key, temperature=temperature)

    else:
        raise ValueError(f"Model {model_name} is not supported!!!")


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


def __get_api_keys_for_model(model_name: str, api_key: str, additional_keys: tuple = None):
    """
    根据环境变量获取模型对应的key
    :param model_name: 模型名称
    :param api_key:
    :param additional_keys:
    :return:
    """
    if api_key is None:
        key = __parse_llm_api_key(model_name)
        if additional_keys:
            return key[:len(additional_keys)]  # Extract the needed number of key
        return (api_key,)
    return (api_key,) + additional_keys if additional_keys else (api_key,)


def __parse_llm_api_key(model: str):
    env_file = None
    """
    通过 model 和 env_file 的来解析平台参数
    """
    if env_file is None:
        _ = load_dotenv(find_dotenv())
        env_file = dict(os.environ)
    if model == OPEN_API_STR:
        key = env_file["OPENAI_API_KEY"]
    elif model == WEN_XIN_STR:
        key = env_file["wenxin_api_key"], env_file["wenxin_secret_key"]
    elif model == SPARK_STR:
        key = env_file["spark_api_key"], env_file["spark_appid"], env_file["spark_api_secret"]
    elif model == ZHI_PU_AI_STR:
        key = get_from_dict_or_env(env_file, "zhipuai_api_key", "ZHIPUAI_API_KEY")
    else:
        raise ValueError(f"model{model} not support!!!")

    print("从环境变量获取key完成")
    return key


def format_chat_prompt(message, chat_history):
    prompt = ""
    for chat_item in chat_history:
        print(f"chat_item是:{chat_item}")
        if chat_item['role'] == 'user':
            prompt = f"{prompt} \nUser:{chat_item['content']}"
        elif chat_item['role'] == 'assistant':
            prompt = f"{prompt} \nAssistant:{chat_item['content']}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt


def respond(message, chat_history, llm, history_len=3, temperature=0.1, max_tokens=2048):
    """
    该函数用于生成机器人的回复。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    "": 空字符串表示没有内容需要显示在界面上，可以替换为真正的机器人回复。
    chat_history: 更新后的聊天历史记录
    """
    print(
        f"respond函数入参: message:{message}, chat_history:{chat_history}, history_len:{history_len}, temperature:{temperature}, max_tokens:{max_tokens}")
    if message is None or len(message) < 1:
        return None
    try:
        # 限制 history 的记忆长度
        chat_history = chat_history[-history_len:] if history_len > 0 else []
        # 调用上面的函数，将用户的消息和聊天历史记录格式化为一个 prompt。
        print(f"取出的最近{history_len}条消息为:{chat_history}")
        formatted_prompt = format_chat_prompt(message=message, chat_history=chat_history)
        print(f"格式化后的消息为{formatted_prompt}")
        # 使用llm对象的predict方法生成机器人的回复（注意：llm对象在此代码中并未定义）。
        response = get_completion(
            formatted_prompt, llm, temperature=temperature, max_tokens=max_tokens)
        print(f"回答为:{response}")
        return [ChatMessage(role="user", content=message), ChatMessage(role=response[0], content=response[1])]
    except Exception as e:
        raise e

