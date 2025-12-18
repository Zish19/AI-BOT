"""
agent.py
--------
Python 3.13 + Streamlit Cloud SAFE
Groq + Tavily (no LangChain agents)
"""

import os
import requests
from datetime import datetime, timedelta

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults

# ===============================
# ENV CHECK
# ===============================
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY missing")

if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY missing")

# ===============================
# LLM
# ===============================
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=1024,
)

tavily = TavilySearchResults(max_results=3)

# ===============================
# TOOLS (MANUAL)
# ===============================
def get_time():
    now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    return now.strftime("%Y-%m-%d %H:%M:%S (IST)")


def get_weather(city: str):
    try:
        r = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
        return r.text
    except Exception:
        return "Weather unavailable."

# ===============================
# ROUTER CHAT
# ===============================
def chat(user_input: str) -> str:
    text = user_input.lower()

    # Tool routing (simple & stable)
    if "time" in text or "date" in text:
        return get_time()

    if "weather" in text:
        city = text.replace("weather", "").strip() or "Delhi"
        return get_weather(city)

    if "search" in text or "news" in text:
        results = tavily.invoke({"query": user_input})
        return results[0]["content"] if results else "No results found."

    # Default: LLM
    response = llm.invoke(user_input)
    return response.content

