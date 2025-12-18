# AI Agent with FastAPI and Streamlit

A powerful AI agent built with LangChain, FastAPI, and Streamlit that can answer questions and perform web searches.

## 🚀 Features

- Natural language conversation with AI agent
- Web search capabilities using Tavily API
- Web-based chat interface with Streamlit
- FastAPI backend for scalable deployment
- Session-based chat history

## 🛠️ Prerequisites

- Python 3.8+
- [Git](https://git-scm.com/)
- [UV](https://docs.astral.sh/uv/pip/environments/) (recommended) or pip
- [OpenAI API Key](https://platform.openai.com/api-keys)
- [Tavily API Key](https://app.tavily.com/)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Agent.git
   cd Agent
   ```

2. **Set up environment variables**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

3. **Install dependencies**
   Using Poetry:
   ```bash
   poetry install
   ```
   
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Application

1. **Start the FastAPI backend** (in one terminal):
   ```bash
   uvicorn api:app --reload
   ```
   The API will be available at `http://localhost:8000`

2. **Start the Streamlit frontend** (in another terminal):
   ```bash
   streamlit run app.py
   ```
   The web interface will open in your default browser at `http://localhost:8501`

## 🌐 API Endpoints

- `POST /chat` - Send a message to the agent
  ```json
  {
    "message": "Your message here",
    "session_id": "optional-session-id"
  }
  ```

## 📂 Project Structure

```
Agent/
├── .env.example         # Example environment variables
├── .gitignore           # Git ignore file
├── README.md            # This file
├── api.py              # FastAPI application
├── app.py              # Streamlit frontend
├── agent.py            # LangChain agent implementation
└── requirements.txt    # Python dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://python.langchain.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)
- [OpenAI](https://openai.com/)
- [Tavily](https://tavily.com/)
