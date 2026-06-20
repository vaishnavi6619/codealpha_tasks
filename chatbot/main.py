from fastapi import FastAPI
from fastapi.responses import FileResponse
from service.chatbot import ChatBot
from model.models import ChatRequest, ChatResponse

app = FastAPI()
bot = ChatBot()
print("App loaded")

@app.get("/")
def home():
    return FileResponse("chatbot_ui.html")


@app.post("/chat")
def chat(chatReq: ChatRequest):
    response: ChatResponse = bot.chat(chatReq)

    return response


@app.delete("/chat")
def clear_history():
    bot.clear()