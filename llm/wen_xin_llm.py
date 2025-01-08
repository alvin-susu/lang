import json
from typing import Optional, List, Any

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun

from llm.self_llm import Self_LLM


def get_access_token(api_key: str, secret_key: str):
    """
    使用 API Key，Secret Key 获取access_token，替换下列示例中的应用API Key、应用Secret Key
    """
    # 指定网址
    url = f"https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={api_key}&client_secret={secret_key}"
    # 设置 POST 访问
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    # 通过 POST 访问获取账户对应的 access_token
    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json().get("access_token")


class WenXin_LLM(Self_LLM):
    # 文心大模型的自定义 LLM
    # URL
    url: str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token={}"
    # Secret_Key
    secret_key: str = None
    # access_token
    access_token: str = None

    def init_access_token(self):
        try:
            if self.api_key is not None and self.secret_key is not None:
                # 两个 Key 均非空才可以获取 access_token
                self.access_token = get_access_token(self.api_key, self.secret_key)
            else:
                raise Exception("获取 access_token 失败，请检查 Key")
        except Exception as e:
            raise Exception("获取 access_token 失败")

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 如果 access_token 为空，初始化 access_token
        if self.access_token is None:
            self.init_access_token()
        # API 调用 url
        url = self.url.format(self.access_token)
        # 配置 POST 参数
        payload = json.dumps({
            "messages": [
                {
                    "role": "user",  # user prompt
                    "content": "{}".format(prompt)  # 输入的 prompt
                }
            ],
            'temperature': self.temperature
        })
        headers = {
            'Content-Type': 'application/json'
        }
        # 发起请求
        response = requests.request("POST", url, headers=headers, data=payload, timeout=self.request_timeout)
        if response.status_code == 200:
            # 返回的是一个 Json 字符串
            js = json.loads(response.text)
            # print(js)
            return js["result"]
        else:
            return "请求失败"

    @property
    def _llm_type(self) -> str:
        return "WenXin"