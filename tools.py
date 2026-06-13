from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool
from datetime import datetime
from pathlib import Path
import warnings

# Suppress the Wikipedia BeautifulSoup parser warning
warnings.filterwarnings("ignore", message="No parser was explicitly specified", category=UserWarning)

def save_to_txt(data: str, filename: str = "research_output.txt"):
    output_dir = Path(__file__).resolve().parent / "research_outputs"
    output_dir.mkdir(exist_ok=True)
    safe_name = Path(filename).name or "research_output.txt"
    if not safe_name.endswith(".txt"):
        safe_name = f"{safe_name}.txt"
    output_path = output_dir / safe_name

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {output_path}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchResults(output_format="json", num_results=6)


def safe_search(query: str) -> str:
    try:
        return search.run(query)
    except Exception as exc:
        return f"Web search failed for {query!r}: {exc}"


search_tool = Tool(
    name="search",
    func=safe_search,
    description="Search the web for current information. Returns JSON results with titles, snippets, and URLs.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)


def safe_wikipedia(query: str) -> str:
    try:
        return wiki.run(query)
    except Exception as exc:
        return f"Wikipedia lookup failed for {query!r}: {exc}"


wiki_tool = Tool(
    name="wikipedia",
    func=safe_wikipedia,
    description="Look up encyclopedia context from Wikipedia. Returns a failure note instead of raising when Wikipedia is unavailable.",
)
