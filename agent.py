"""
Agent.py
---------
LangChain Agent using:
- Groq (LLM)
- Tavily (Web Search)
- Custom tools (datetime, weather)
Compatible with Streamlit Cloud
"""

# ===============================
# Imports
# ===============================
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
# ENVIRONMENT VALIDATION
# ===============================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("❌ GROQ_API_KEY missing. Add it to Streamlit Secrets.")

if not TAVILY_API_KEY:
    raise RuntimeError("❌ TAVILY_API_KEY missing. Add it to Streamlit Secrets.")

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
    """Return current date and time in Indian Standard Time (IST)."""
    try:
        if ZoneInfo:
            now = datetime.now(ZoneInfo("Asia/Kolkata"))
        else:
            now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        return now.strftime("%Y-%m-%d %H:%M:%S (IST)")
    except Exception as e:
        return f"Error fetching time: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """Get current weather for a city using wttr.in (no API key required)."""
    try:
        url = f"https://wttr.in/{city}?format=j1"
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return f"Could not fetch weather for {city}"

        data = res.json()
        current = data["current_condition"][0]
        desc = current["weatherDesc"][0]["value"]
        temp = current["temp_C"]
        feels = current["FeelsLikeC"]
        humidity = current["humidity"]

        return (
            f"Weather in {city}: {desc}, "
            f"Temperature: {temp}°C (feels like {feels}°C), "
            f"Humidity: {humidity}%"
        )
    except Exception as e:
        return f"Weather error: {str(e)}"

# ===============================
# GLOBAL CHAT MEMORY (lightweight)
# ===============================
chat_history = []

# ===============================
# AGENT CREATION
# ===============================
def create_agent():
    """Create and return the AgentExecutor."""

    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",  # ✅ Groq-supported
        temperature=0.7,
        max_tokens=1024,
        timeout=30,
        max_retries=2,
    )

    tavily_tool = TavilySearchResults(
        max_results=3,
        search_depth="basic",
        include_answer=True,
        include_raw_content=False,
    )

    tools = [get_current_datetime, get_weather, tavily_tool]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant.
You have access to tools for:
- Current date & time (IST)
- Weather lookup
- Web search (Tavily)

Use tools when required.
Be clear, concise, and conversational.
""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    return executor

# ===============================
# CHAT FUNCTION (USED BY STREAMLIT)
# ===============================
def chat(user_input: str, agent_executor):
    """Run one chat turn and update memory."""
    global chat_history

    formatted_history = []
    for role, content in chat_history:
        if role == "human":
            formatted_history.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_history.append(AIMessage(content=content))

    try:
        result = agent_executor.invoke(
            {
                "input": user_input,
                "chat_history": formatted_history,
            }
        )

        output = result.get("output", "No response generated.")

    except Exception as e:
        output = f"⚠️ Error: {str(e)}"

    chat_history.append(("human", user_input))
    chat_history.append(("assistant", output))

    # keep last 10 exchanges only
    chat_history = chat_history[-20:]

    return output

