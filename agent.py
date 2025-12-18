"""
agent.py
--------
LangChain Agent using Groq + Tavily
Works on Streamlit Cloud
"""

import os
import requests
from datetime import datetime, timedelta

from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq

# ===============================
# ENV VALIDATION (CRITICAL)
# ===============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing (check Streamlit Secrets)")

if not TAVILY_API_KEY:
    raise RuntimeError("TAVILY_API_KEY missing (check Streamlit Secrets)")

# ===============================
# TIMEZONE SUPPORT (IST)
# ===============================
try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# ===============================
# TOOLS
# ===============================
@tool
def get_current_datetime() -> str:
    """Return current date & time in IST."""
    if ZoneInfo:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    else:
        now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    return now.strftime("%Y-%m-%d %H:%M:%S (IST)")


@tool
def get_weather(city: str) -> str:
    """Get weather using wttr.in (no API key needed)."""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        r = requests.get(url, timeout=10)
        data = r.json()["current_condition"][0]
        return (
            f"Weather in {city}: {data['weatherDesc'][0]['value']}, "
            f"{data['temp_C']}°C (feels like {data['FeelsLikeC']}°C)"
        )
    except Exception:
        return "Unable to fetch weather right now."

# ===============================
# MEMORY
# ===============================
chat_history = []

# ===============================
# AGENT CREATION
# ===============================
def create_agent():
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1024,
    )

    tavily = TavilySearchResults(max_results=3)

    tools = [get_current_datetime, get_weather, tavily]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant with tools."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
    )

# ===============================
# CHAT FUNCTION
# ===============================
def chat(user_input: str, executor):
    global chat_history

    history_msgs = []
    for role, msg in chat_history:
        if role == "human":
            history_msgs.append(HumanMessage(content=msg))
        else:
            history_msgs.append(AIMessage(content=msg))

    result = executor.invoke(
        {"input": user_input, "chat_history": history_msgs}
    )

    output = result.get("output", "No response.")

    chat_history.append(("human", user_input))
    chat_history.append(("assistant", output))
    chat_history = chat_history[-20:]

    return output

