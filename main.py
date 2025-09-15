from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

def perform_research(agent_executor, parser, query):
    """Perform research with single attempt - no retries to preserve API quota"""
    try:
        raw_response = agent_executor.invoke({"query": query})
        structured_response = parser.parse(raw_response.get("output"))
        return structured_response
    except Exception as e:
        # Don't retry - fail immediately to preserve API quota
        raise e
    

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
query = input("What can i help you research? ")

try:
    # Single attempt research - no retries to preserve API quota
    structured_response = perform_research(agent_executor, parser, query)
    print(structured_response)
except Exception as e:
    error_msg = str(e)
    print(f"❌ Research failed: {error_msg}")
    print("💡 Tip: Make sure your API key is valid and you haven't exceeded rate limits.")