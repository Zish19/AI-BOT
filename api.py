from fastapi import FastAPI
from agent import create_agent, chat

app = FastAPI()
agent = create_agent()

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat")
def chat_api(message: str):
    return {"response": chat(message, agent)}

