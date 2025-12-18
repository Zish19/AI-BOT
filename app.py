import streamlit as st
from agent import chat

st.set_page_config(page_title="Agentic AI Chat", page_icon="🤖")

st.title("🤖 Agentic AI Chat")
st.caption("Groq + Tavily powered agent")

st.success("Agent initialized")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chat(prompt)
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

