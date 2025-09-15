from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

def should_retry_exception(exception):
    """Determine if an exception should trigger a retry"""
    error_message = str(exception).lower()
    
    # Retry on quota/rate limit errors
    retry_errors = [
        'quota exceeded',
        'rate limit',
        'resource exhausted',
        'too many requests',
        '429',  # HTTP 429 Too Many Requests
        '503',  # HTTP 503 Service Unavailable
        '502',  # HTTP 502 Bad Gateway
        '500',  # HTTP 500 Internal Server Error
    ]
    
    return any(error in error_message for error in retry_errors)

@retry(
    stop=stop_after_attempt(3),  # Maximum 3 attempts
    wait=wait_exponential(multiplier=1, min=4, max=32),  # Exponential backoff: 4s, 8s, 16s, 32s
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda retry_state: print(f"Retrying in {retry_state.next_action.sleep:.1f} seconds... (Attempt {retry_state.attempt_number}/5)")
)
def perform_research_with_retry(agent_executor, parser, query):
    """Perform research with automatic retry on transient failures"""
    try:
        raw_response = agent_executor.invoke({"query": query})
        structured_response = parser.parse(raw_response.get("output"))
        return structured_response
    except Exception as e:
        if should_retry_exception(e):
            print(f"Transient error detected: {e}. Retrying...")
            raise e  # Re-raise to trigger retry
        else:
            raise e  # Don't retry for non-transient errors
    

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
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
    # Use retry-enabled research function
    structured_response = perform_research_with_retry(agent_executor, parser, query)
    print(structured_response)
except Exception as e:
    error_msg = str(e)
    if should_retry_exception(e):
        print(f"❌ Research failed after multiple attempts due to API limits. Please wait a few minutes and try again.")
        print(f"Error details: {error_msg}")
    else:
        print(f"❌ Research failed: {error_msg}")