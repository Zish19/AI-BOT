import streamlit as st
from agent import create_agent, chat

st.set_page_config(page_title="Agentic AI Chat", page_icon="🤖")

st.title("🤖 Agentic AI Chat")
st.caption("Groq + Tavily powered agent")

@st.cache_resource
def load_agent():
    return create_agent()

agent = load_agent()
st.success("Agent initialized")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chat(prompt, agent)
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

