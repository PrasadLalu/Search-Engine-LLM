import streamlit as st

from langchain_groq import ChatGroq
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentType, initialize_agent
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun, DuckDuckGoSearchRun

# Setup App - Search Engine
st.title("🔎 LangChain - Chat with search")
"""
    In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of 
    an agent in an interactive Streamlit app.Try more LangChain 🤝 Streamlit Agent examples 
    at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar for settings
st.sidebar.title("Settings")
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
if not groq_api_key:
    st.error("Please enter Groq API Key.")

# Wikipedia
wikipedia_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)

# Arxiv
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Search
search = DuckDuckGoSearchRun(name="Search")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I'm a chatbot who can search the web. How can I help you?"
        }
    ]
    
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])
    
if prompt := st.chat_input("What is machine learning?"):
    st.session_state.messages.append({ "role": "user", "content": prompt })
    st.chat_message("user").write(prompt)
    
    # Initialize model
    llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")
    print("I am here...", groq_api_key)
    
    # Create tool
    tools = [wikipedia, arxiv, search]
    
    # Create search agent
    search_agent = initialize_agent(llm=llm, tools=tools, 
                                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                                    handling_parsing_errors=True)

    # Search query
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({ "role": "assistant", "content": response })
        st.write(response)
