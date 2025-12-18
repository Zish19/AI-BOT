"""
agent.py
--------
Stable Groq + Tavily Agent (NO tool parser crash)
"""

import os
import requests
from datetime import datetime, timedelta

from langchain.agents import initialize_agent, AgentType
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

# ===============================
# ENV CHECK
# ===============================
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY missing")

if not os.getenv("TAVILY_API_KEY"):
    raise RuntimeError("TAVILY_API_KEY missing")

# ===============================
# TOOLS
# ===============================
@tool
def get_current_datetime() -> str:
    """Get current time in IST."""
    now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    return now.strftime("%Y-%m-%d %H:%M:%S (IST)")


@tool
def get_weather(city: str) -> str:
    """Get weather info."""
    try:
        r = requests.get(f"https://wttr.in/{city}?format=3", timeout=10)
        return r.text
    except Exception:
        return "Weather service unavailable."

tools = [
    get_current_datetime,
    get_weather,
    TavilySearchResults(max_results=3),
]

# ===============================
# AGENT
# ===============================
def create_agent():
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024,
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.OPENAI_FUNCTIONS,  # ✅ IMPORTANT
        verbose=False,
        handle_parsing_errors=True,
    )

    return agent

# ===============================
# CHAT
# ===============================
def chat(user_input: str, agent):
    try:
        response = agent.run(user_input)
        return response
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

