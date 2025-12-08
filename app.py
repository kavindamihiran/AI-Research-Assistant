import streamlit as st
import uuid
import builtins
builtins.uuid = uuid  # Fix for Python 3.14 compatibility with LangGraph

from dotenv import load_dotenv, set_key
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from tools import search_tool, wiki_tool, save_tool
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-card {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .api-key-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem auto;
        max-width: 500px;
    }
</style>
""", unsafe_allow_html=True)

# Available FREE models for OpenRouter that SUPPORT TOOL/FUNCTION CALLING
# Only models with tool use support are listed here to avoid 404 errors
AVAILABLE_MODELS = [
    # Verified to support tool calling
    "google/gemini-2.0-flash-exp:free",          # Best free option - fast & reliable
    "google/gemma-3-27b-it:free",                 # Good tool support
    "arcee-ai/trinity-mini:free",                 # Designed for function calling
    "openai/gpt-oss-120b:free",                   # Native tool use support
    "openai/gpt-oss-20b:free",                    # Agentic capabilities
    "amazon/nova-2-lite:free",                    # Multi-step agentic workflows
    "qwen/qwen3-coder-480b-a35b:free",            # Optimized for tool use
    "zhipu/glm-4.5-air:free",                     # Good for tool use
    "meituan/longcat-flash-chat:free",            # Agentic tasks support
    "meta-llama/llama-3.3-70b-instruct:free",     # Tool capable
    "mistralai/mistral-small-3.1-24b-instruct:free",  # Function calling support
]

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

def save_api_key_to_env(api_key: str):
    """Save API key to .env file"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    # Create .env if it doesn't exist
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write('')
    set_key(env_path, 'OPENROUTER_API_KEY', api_key)

def get_api_key():
    """Get API key from session state or environment"""
    if 'api_key' in st.session_state and st.session_state.api_key:
        return st.session_state.api_key
    return os.getenv('OPENROUTER_API_KEY')

def initialize_agent(api_key: str, model_name: str):
    """Initialize the research agent with OpenRouter"""
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        streaming=False,  # Disable streaming for tool compatibility
    )
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    tools = [search_tool, wiki_tool, save_tool]
    
    system_prompt = f"""
    You are a research assistant that will help generate a research paper.
    Answer the user query and use necessary tools.
    Wrap the output in this format and provide no other text
    {parser.get_format_instructions()}
    """
    
    agent = create_react_agent(llm, tools, prompt=system_prompt)

    return agent, parser

def perform_research(agent, parser, query):
    """Perform research with single attempt - no retries to preserve API quota"""
    import re
    import json
    
    try:
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        # Extract the final AI message content
        final_message = result["messages"][-1].content
        
        # Try to extract JSON from the response (handle extra text before/after)
        json_match = re.search(r'\{[\s\S]*\}', final_message)
        if json_match:
            json_str = json_match.group()
            # Parse the extracted JSON
            data = json.loads(json_str)
            structured_response = ResearchResponse(**data)
            return structured_response
        else:
            # Fall back to standard parser
            structured_response = parser.parse(final_message)
            return structured_response
    except Exception as e:
        # Don't retry - fail immediately to preserve API quota
        raise e

def show_api_key_setup():
    """Show API key setup page"""
    st.markdown('<h1 class="main-header">🔬 AI Research Assistant</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="api-key-box">
        <h2>🔑 API Key Required</h2>
        <p>Enter your OpenRouter API key to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("api_key_form"):
        api_key_input = st.text_input(
            "OpenRouter API Key",
            type="password",
            placeholder="sk-or-v1-...",
            help="Get your API key from https://openrouter.ai/keys"
        )
        
        save_to_env = st.checkbox("Save API key to .env file for future sessions", value=True)
        
        submitted = st.form_submit_button("🚀 Start Using App", type="primary", use_container_width=True)
        
        if submitted and api_key_input:
            st.session_state.api_key = api_key_input
            if save_to_env:
                save_api_key_to_env(api_key_input)
            st.rerun()
        elif submitted:
            st.error("Please enter an API key")
    
    st.markdown("---")
    st.markdown("**Get your API key:** [OpenRouter Keys](https://openrouter.ai/keys)")

def main():
    # Initialize session state
    if 'api_key' not in st.session_state:
        st.session_state.api_key = get_api_key()
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = AVAILABLE_MODELS[0]
    if 'research_results' not in st.session_state:
        st.session_state.research_results = None
    if 'research_error' not in st.session_state:
        st.session_state.research_error = None
    
    # Check if API key is available
    api_key = get_api_key()
    if not api_key:
        show_api_key_setup()
        return
    
    # Sidebar for guide and settings
    with st.sidebar:
        st.title("🔬 AI Research Assistant")
        st.markdown("---")

        # Model Selection
        st.markdown("### 🤖 Model Selection")
        selected_model = st.selectbox(
            "Choose AI Model",
            AVAILABLE_MODELS,
            index=AVAILABLE_MODELS.index(st.session_state.selected_model) if st.session_state.selected_model in AVAILABLE_MODELS else 0,
            help="Select the AI model to use for research"
        )
        st.session_state.selected_model = selected_model

        # API Key Settings
        with st.expander("🔑 API Key Settings", expanded=False):
            st.text_input(
                "Current API Key",
                value=api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***",
                disabled=True
            )
            
            new_key = st.text_input("New API Key", type="password", placeholder="Enter new key...")
            if st.button("Update Key", use_container_width=True):
                if new_key:
                    st.session_state.api_key = new_key
                    save_api_key_to_env(new_key)
                    st.success("API key updated!")
                    st.rerun()
            
            if st.button("Clear Key & Logout", use_container_width=True):
                st.session_state.api_key = None
                st.rerun()

        st.markdown("---")

        # Guide section
        with st.expander("📖 User Guide", expanded=False):
            st.markdown("""
            ### How to Use
            1. **Select a model** from the dropdown
            2. **Enter your research query** in the main area
            3. **Click "Start Research"** to begin
            4. **View results** in the structured format below

            ### Features
            - **Web Search**: Uses DuckDuckGo for comprehensive web searches
            - **Wikipedia Integration**: Fetches reliable information from Wikipedia
            - **Structured Output**: Provides organized research summaries
            - **Multi-Model Support**: Choose from various AI models
            """)

        st.markdown("---")
        st.markdown("Built with LangChain & Streamlit")

    # Main content
    st.markdown('<h1 class="main-header">🔬 AI Research Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Get comprehensive research summaries powered by AI")

    # Input section
    with st.container():
        st.markdown("### Enter Your Research Query")
        query = st.text_area(
            "What would you like to research?",
            placeholder="e.g., What is quantum computing? Latest developments in AI...",
            height=100,
            key="query_input"
        )

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            start_research = st.button("🔍 Start Research", type="primary", use_container_width=True)
        with col2:
            clear_results = st.button("🗑️ Clear", use_container_width=True)

    if clear_results:
        st.session_state.research_results = None
        st.session_state.research_error = None
        st.rerun()

    if start_research and query.strip():
        with st.spinner("🔬 Researching... This may take a few moments"):
            try:
                agent_executor, parser = initialize_agent(api_key, selected_model)
                
                # Single attempt research - no retries to preserve API quota
                structured_response = perform_research(agent_executor, parser, query)
                
                st.session_state.research_results = structured_response
                st.session_state.research_error = None
                
            except Exception as e:
                error_msg = str(e)
                st.session_state.research_error = f"❌ Research failed: {error_msg}"
                st.session_state.research_results = None

    # Display results
    if st.session_state.research_error:
        st.error(f"❌ {st.session_state.research_error}")

    if st.session_state.research_results:
        result = st.session_state.research_results

        st.success("✅ Research Complete!")

        # Topic
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### 📋 Topic: {result.topic}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Summary
        with st.expander("📝 Summary", expanded=True):
            st.write(result.summary)

        # Sources
        if result.sources:
            with st.expander("🔗 Sources", expanded=True):
                for i, source in enumerate(result.sources, 1):
                    st.markdown(f"{i}. {source}")

        # Tools Used
        if result.tools_used:
            with st.expander("🛠️ Tools Used", expanded=False):
                st.write(", ".join(result.tools_used))

        # Download option
        st.markdown("### 💾 Download Results")
        download_text = f"""# Research Results

**Topic:** {result.topic}

**Summary:**
{result.summary}

**Sources:**
{chr(10).join(f"- {source}" for source in result.sources)}

**Tools Used:** {", ".join(result.tools_used)}
"""

        st.download_button(
            label="📄 Download as Text File",
            data=download_text,
            file_name=f"research_{result.topic.replace(' ', '_').lower()}.txt",
            mime="text/plain",
            use_container_width=True
        )

if __name__ == "__main__":
    main()