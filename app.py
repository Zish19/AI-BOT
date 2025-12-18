import streamlit as st

st.set_page_config(page_title="Agentic AI Chat", page_icon="🤖")

st.write("✅ Streamlit started")

try:
    from agent import create_agent, chat
    st.write("✅ agent.py imported")
except Exception as e:
    st.error(f"❌ Failed to import agent.py: {e}")
    st.stop()

st.title("🤖 Agentic AI Chat")
st.caption("Groq + Tavily powered agent")

@st.cache_resource
def load_agent():
    st.write("⏳ Initializing agent...")
    return create_agent()

try:
    agent = load_agent()
    st.success("✅ Agent initialized")
except Exception as e:
    st.error(f"❌ Agent init failed: {e}")
    st.stop()

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
        try:
            response = chat(prompt, agent)
        except Exception as e:
            response = f"⚠️ Chat error: {e}"
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

