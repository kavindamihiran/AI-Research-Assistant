# AI Research Assistant

An AI-powered research assistant built with LangChain and Google Gemini API. This agent can perform web searches, query Wikipedia, and save research findings to text files.

## Features

- **Web Search**: Uses DuckDuckGo for web searches
- **Wikipedia Queries**: Fetches information from Wikipedia
- **Structured Output**: Provides research results in a structured format
- **File Saving**: Automatically saves research output to timestamped text files

## Prerequisites

- Python 3.8+
- Google AI API key (free tier available)
- Virtual environment (recommended)

## Installation

1. Clone or download this repository
2. Create a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your API key:
   - Get a Google AI API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a `.env` file in the project root:
     ```
     GOOGLE_API_KEY=your_api_key_here
     ```

## Usage

Run the research assistant:

```bash
python main.py
```

Enter your research query when prompted. The agent will:
- Search for relevant information
- Query Wikipedia if needed
- Generate a structured research summary
- Save results to `research_output.txt`

Example queries:
- "What is quantum computing?"
- "Latest developments in AI"
- "History of the internet"

## Project Structure

```
├── main.py              # Main application script
├── tools.py             # Custom tools for search and saving
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API keys)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## API Limits

The free tier of Google Gemini API has rate limits:
- 15 requests per minute
- If you hit limits, wait a minute or upgrade to a paid plan

## Dependencies

- langchain: Framework for building LLM applications
- langchain-google-genai: Google Gemini integration
- langchain-community: Community tools
- duckduckgo-search: Web search functionality
- wikipedia: Wikipedia API wrapper
- python-dotenv: Environment variable management
- pydantic: Data validation

## Troubleshooting

- **ModuleNotFoundError**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **API Quota Exceeded**: Wait for quota reset or enable billing in Google Cloud Console
- **Model Not Found**: Update the model name in `main.py` if Gemini models change

## License

This project is for educational purposes. Check individual library licenses for commercial use.