import gradio as gr

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from common.common_constants import AIGC_LOGO_PATH, DATAWHALE_LOGO_PATH, AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH
from common.db_utils import create_db
from common.common_utils import respond
from serve.web.component import ModelComponent

LLM_MODEL_DICT = {
    "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
    "wenxin": ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"],
    "xinhuo": ["Spark-1.5", "Spark-2.0"],
    "zhipuai": ["chatglm_pro", "chatglm_std", "chatglm_lite"]
}
INIT_LLM = "gpt-3.5-turbo"
EMBEDDING_MODEL_LIST = ['zhipuai', 'openai', 'm3e']
INIT_EMBEDDING_MODEL = "openai"
LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()), [])

model_center = ModelComponent()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        gr.Image(value=AIGC_LOGO_PATH,
                 scale=1,
                 min_width=10,
                 show_label=False,
                 container=False)

        with gr.Column(scale=2):
            gr.Markdown(
                """
                <h1><center>动手学大模型应用开发</center></h1>
                <center>LLM-UNIVERSE</center>
                """
            )
            gr.Image(value=DATAWHALE_LOGO_PATH,
                     scale=1,
                     min_width=10,
                     show_label=False,
                     show_download_button=False,
                     container=False)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                type="messages",
                height=400,
                show_copy_button=True,
                show_share_button=True,
                avatar_images=(AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH))
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")
            with gr.Row():
                db_with_his_btn = gr.Button("Chat db with history")
                db_wo_his_btn = gr.Button("Chat db without history")
                llm_btn = gr.Button("Chat with llm")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot],
                    value="Clear console")
        with gr.Column(scale=1):
            file = gr.File(label='请选择知识库目录',
                           file_count='directory')
            with gr.Row():
                init_db = gr.Button("知识库文件向量化")
            model_argument = gr.Accordion(label="参数配置", open=False)
            with model_argument:
                # 温控系数滑动条
                temperature = gr.Slider(0,
                                        1,
                                        value=0.01,
                                        step=0.01,
                                        label="llm temperature",
                                        interactive=True)
                # topK滑动条
                top_k = gr.Slider(1,
                                  10,
                                  value=3,
                                  step=1,
                                  label="vector db search top k",
                                  interactive=True)
                # 对话历史长度滑动条
                history_len = gr.Slider(0,
                                        5,
                                        value=3,
                                        step=1,
                                        label="history length",
                                        interactive=True)

                model_select = gr.Accordion("模型选择")
                with model_select:
                    llm = gr.Dropdown(
                        LLM_MODEL_LIST,
                        label="large language model",
                        value=INIT_LLM,
                        interactive=True)
                    embeddings = gr.Dropdown(EMBEDDING_MODEL_LIST,
                                             label="Embedding model",
                                             value=INIT_EMBEDDING_MODEL)
        # 设置初始化向量数据库按钮的点击事件，点击时，调用create_db函数，并传入用户的文件和希望使用的embedding模型
        init_db.click(create_db,
                      inputs=[file, embeddings],
                      outputs=[msg])

        # 设置按钮的点击事件。当点击时，调用上面定义的 chain 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_with_his_btn.click(model_center.chat,
                              inputs=[msg, chatbot, llm, embeddings, temperature, top_k, history_len],
                              outputs=chatbot)
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.chat,
                            inputs=[msg, chatbot, llm, embeddings, temperature, top_k],
                            outputs=chatbot)
        # 设置按钮的点击事件。当点击时，调用上面定义的 respond 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        llm_btn.click(respond,
                      inputs=[msg, chatbot, llm, history_len, temperature],
                      outputs=chatbot, show_progress="minimal")

        # 设置文本框的提交事件（即按下Enter键时）。功能与上面的 llm_btn 按钮点击事件相同。
        msg.submit(respond,
                   inputs=[msg, chatbot, llm, history_len, temperature],
                   outputs=[msg, chatbot], show_progress="hidden")
        # 点击后清空后端存储的聊天记录
        clear.click(model_center.clear_history)
    gr.Markdown("""提醒：<br>
        1. 使用时请先上传自己的知识文件，不然将会解析项目自带的知识库。
        2. 初始化数据库时间可能较长，请耐心等待。
        3. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
        """)

gr.close_all()
demo.launch(debug=True)
