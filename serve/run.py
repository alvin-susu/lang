import uvicorn
from fastapi import FastAPI, HTTPException

from qa_chain.qa_chain_self import QaChainSelf
from serve.params.char_params import ChatParams

app = FastAPI() # 创建 api 对象

@app.post("/chat")
async def chat(chat_params: ChatParams):
    if chat_params is None:
        raise HTTPException(status_code=400, detail="Invalid input.")

    if chat_params.embedding_key is None:
        chat_params.embedding_key = chat_params.api_key

    # 确定调用的链条
    if chat_params.is_use_history is False:
        chain = QaChainSelf(chat_params)
        response = chain.answer(question=chat_params.question)
        return response
    else:
        return "API暂不支持即时链条"

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8889)