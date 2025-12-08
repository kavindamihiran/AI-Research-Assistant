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

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

def perform_research(agent, parser, query):
    """Perform research with single attempt - no retries to preserve API quota"""
    try:
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        # Extract the final AI message content
        final_message = result["messages"][-1].content
        structured_response = parser.parse(final_message)
        return structured_response
    except Exception as e:
        # Don't retry - fail immediately to preserve API quota
        raise e


# Get API key from environment
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("❌ Error: OPENROUTER_API_KEY not found in environment")
    print("Please add OPENROUTER_API_KEY to your .env file")
    exit(1)

# Default model
model_name = "google/gemini-flash-1.5"

llm = ChatOpenAI(
    model=model_name,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
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
query = input("What can I help you research? ")

try:
    # Single attempt research - no retries to preserve API quota
    structured_response = perform_research(agent, parser, query)
    print(structured_response)
except Exception as e:
    error_msg = str(e)
    print(f"❌ Research failed: {error_msg}")
    print("💡 Tip: Make sure your API key is valid and you haven't exceeded rate limits.")