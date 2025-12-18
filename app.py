import streamlit as st
from agent import create_agent, chat

st.set_page_config(
    page_title="Agentic AI Chat",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Agentic AI Chat")
st.caption("Groq + Tavily powered agent")

@st.cache_resource
def load_agent():
    return create_agent()

try:
    agent = load_agent()
    st.success("Agent initialized")
except Exception as e:
    st.error(f"Agent failed to start: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = chat(user_input, agent)
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

