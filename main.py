import uuid
import builtins
builtins.uuid = uuid  # Fix for Python 3.14 compatibility with LangGraph

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from tools import search_tool, wiki_tool, save_tool
import os

load_dotenv()

# Available FREE models for OpenRouter
AVAILABLE_MODELS = [
    # OpenAI OSS Models
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",

    # Google Models
    "google/gemma-3-27b-it:free",
    "google/gemma-3n-e4b-it:free",
    "google/gemma-3-4b-it:free",
    
    # Meta Llama Models
    "meta-llama/llama-3.3-70b-instruct:free",
    
    # Qwen Models
    "qwen/qwen3-coder:free",
    "qwen/qwen3-4b:free",
    
    # DeepSeek / TNG Models
    "tngtech/deepseek-r1t2-chimera:free",
    "tngtech/deepseek-r1t-chimera:free",
    "tngtech/tng-r1t-chimera:free",
    
    # Nvidia Models
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "nvidia/nemotron-nano-9b-v2:free",
    
    # Other Models
    "alibaba/tongyi-deepresearch-30b-a3b:free",
]

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

def select_model():
    """Prompt user to select a model"""
    print("\n🤖 Available AI Models:")
    print("-" * 60)
    for num, model_id, description in AVAILABLE_MODELS:
        print(f"  [{num}] {description}")
        print(f"      └─ {model_id}")
    print("-" * 60)
    
    while True:
        choice = input("\nSelect model (1-30) or press Enter for default [1]: ").strip()
        if choice == "":
            choice = "1"
        
        for num, model_id, description in AVAILABLE_MODELS:
            if num == choice:
                print(f"✅ Selected: {description}")
                return model_id
        
        print("❌ Invalid choice. Please enter a number 1-30.")

def perform_research(agent, parser, query):
    """Perform research with single attempt - no retries to preserve API quota"""
    try:
        # Add recursion limit to prevent infinite loops
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            {"recursion_limit": 15}
        )
        # Extract the final AI message content
        final_message = result["messages"][-1].content
        structured_response = parser.parse(final_message)
        return structured_response
    except Exception as e:
        # Don't retry - fail immediately to preserve API quota
        raise e


def main():
    print("=" * 60)
    print("🔬 AI Research Assistant - Command Line Interface")
    print("=" * 60)
    
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not found in environment")
        print("Please add OPENROUTER_API_KEY to your .env file")
        print("\nGet your free API key at: https://openrouter.ai/keys")
        exit(1)
    
    # Select model
    model_name = select_model()
    
    # Initialize LLM with OpenRouter
    llm = ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        streaming=False,  # Disable streaming for tool/function calling
        disable_streaming=True,  # Explicitly disable streaming for tool use
    )
    
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    tools = [search_tool, wiki_tool, save_tool]
    
    system_prompt = f"""
    You are a research assistant that will help generate a research paper.
    Answer the user query and use necessary tools.
    
    IMPORTANT: After gathering sufficient information (usually 2-3 tool calls), 
    you MUST stop and provide your final response in the JSON format below.
    Do NOT keep calling tools indefinitely.
    
    Wrap the output in this format and provide no other text:
    {parser.get_format_instructions()}
    """
    
    agent = create_react_agent(llm, tools, prompt=system_prompt)
    
    # Get research query
    print("\n" + "-" * 60)
    query = input("🔍 What can I help you research? ")
    
    if not query.strip():
        print("❌ No query provided. Exiting.")
        exit(1)
    
    print("\n🔬 Researching... This may take a moment.\n")
    
    try:
        # Single attempt research - no retries to preserve API quota
        structured_response = perform_research(agent, parser, query)
        
        print("=" * 60)
        print("✅ RESEARCH COMPLETE")
        print("=" * 60)
        print(f"\n📋 Topic: {structured_response.topic}")
        print(f"\n📝 Summary:\n{structured_response.summary}")
        print(f"\n🔗 Sources:")
        for i, source in enumerate(structured_response.sources, 1):
            print(f"   {i}. {source}")
        print(f"\n🛠️ Tools Used: {', '.join(structured_response.tools_used)}")
        print("=" * 60)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Research failed: {error_msg}")
        
        # Provide specific guidance based on error type
        if "429" in error_msg:
            print("\n💡 This model is rate-limited. Try a different model.")
        elif "404" in error_msg and "tool" in error_msg.lower():
            print("\n💡 This model may not support tool calling. Try a different model.")
        elif "400" in error_msg and "streaming" in error_msg.lower():
            print("\n💡 Streaming mode issue. Please report this bug.")
        else:
            print("\n💡 Tip: Make sure your API key is valid and try a different model.")


if __name__ == "__main__":
    main()