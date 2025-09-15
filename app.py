import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
import os
import time

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
</style>
""", unsafe_allow_html=True)

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

def initialize_agent():
    """Initialize the research agent"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a research assistant that will help generate a research paper.
                Answer the user query and use necessary tools.
                Wrap the output in this format and provide no other text\n{format_instructions}
                """,
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{query}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    tools = [search_tool, wiki_tool, save_tool]
    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )

    return AgentExecutor(agent=agent, tools=tools, verbose=False), parser

def perform_research(agent_executor, parser, query):
    """Perform research with single attempt - no retries to preserve API quota"""
    try:
        raw_response = agent_executor.invoke({"query": query})
        structured_response = parser.parse(raw_response.get("output"))
        return structured_response
    except Exception as e:
        # Don't retry - fail immediately to preserve API quota
        raise e

def main():
    # Sidebar for guide and settings
    with st.sidebar:
        st.title("🔬 AI Research Assistant")
        st.markdown("---")

        # Guide section
        with st.expander("📖 User Guide", expanded=False):
            st.markdown("""
            ### How to Use
            1. **Enter your research query** in the main area
            2. **Click "Start Research"** to begin
            3. **View results** in the structured format below
            4. **Download** your research if needed

            ### Features
            - **Web Search**: Uses DuckDuckGo for comprehensive web searches
            - **Wikipedia Integration**: Fetches reliable information from Wikipedia
            - **Structured Output**: Provides organized research summaries
            - **Auto-Save**: Saves results to timestamped text files

            ### Tips
            - Be specific in your queries for better results
            - Examples: "What is quantum computing?", "Latest AI developments"
            - **Single attempt only** - no retries to preserve API quota
            - If request fails, check your API key and try again
            """)

        # API Key status
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key:
            st.success("✅ API Key Configured")
        else:
            st.error("❌ API Key Missing")
            st.info("Add GOOGLE_API_KEY to your .env file")

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

    # Results section
    if 'research_results' not in st.session_state:
        st.session_state.research_results = None
    if 'research_error' not in st.session_state:
        st.session_state.research_error = None

    if clear_results:
        st.session_state.research_results = None
        st.session_state.research_error = None
        st.rerun()

    if start_research and query.strip():
        with st.spinner("🔬 Researching... This may take a few moments"):
            try:
                agent_executor, parser = initialize_agent()
                
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
{"\n".join(f"- {source}" for source in result.sources)}

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