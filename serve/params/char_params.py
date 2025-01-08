# 定义数据模型结构体 用于post请求
from typing import List

from pydantic import BaseModel, StrictStr, Field, field_validator

from common.common_prompt_template import default_template
from common.common_enums import GptModelName


class ChatParams(BaseModel):
    # 提示词 StrictStr表示不可为None的字符串
    prompt: StrictStr
    # 使用模型 默认为gpt3.5
    model: str = GptModelName.GPT_3_5_TURBO
    # 温度系数 默认值为0.1
    temperature: float = 0.1
    # 是否使用历史对话
    is_use_history: bool = False
    # API_KRY
    api_key: str = None
    # Secret_Key
    secret_key: str = None
    # access_token
    access_token: str = None
    # APPID
    appid: str = None
    # APISecret
    Spark_api_secret: str = None
    # Wenxin_Secret_key
    Wenxin_secret_key: str = None
    # 数据库路径
    vector_db_path: str = "../vector_db/chroma"
    # 源文件路径
    source_file_path: str = "../knowledge"
    # 提示词模板 默认为基础模板
    prompt_template: str = default_template
    # 模板中需要替换的变量
    input_variables: List[str] = Field(default=["context", "question"],
                                       description="A list of variables for the prompt template.")
    # Embedding
    embedding: str = "m3e"
    # top k
    top_k: int = 5
    # embedding_key
    embedding_key: str = None

    @classmethod
    def validate_input_variables(cls, value):
        if not value:
            raise ValueError("The input_variables field cannot be an empty list.")
        return value