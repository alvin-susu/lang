from enum import Enum


class GptModelName(Enum):
    GPT_3_5_TURBO = 'gpt-3.5-turbo'
    GPT_3_5_TURBO_16K_0613 = 'gpt-3.5-turbo-16k-0613'
    GPT_3_5_TURBO_0613 = 'gpt-3.5-turbo-0613'
    GPT_4 = 'gpt-4'
    GPT_4_32K = 'gpt-4-32k'


class ErnieBotModelName(Enum):
    ERNIE_BOT = 'ERNIE-Bot'
    ERNIE_BOT_4 = 'ERNIE-Bot-4'
    ERNIE_BOT_TURBO = 'ERNIE-Bot-turbo'


class SparkModelName(Enum):
    SPARK_1_5 = 'Spark-1.5'
    SPARK_2_0 = 'Spark-2.0'


class ChatGlmModelName(Enum):
    CHAT_GLM_PRO = 'chatglm_pro'
    CHAT_GLM_STD = 'chatglm_std'
    CHAT_GLM_LITE = 'chatglm_lite'
