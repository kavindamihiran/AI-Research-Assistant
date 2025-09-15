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

### Web Interface (Recommended)

Run the research assistant with a modern web UI:

```bash
streamlit run app.py
```

This will open a web browser with an intuitive interface featuring:

- **Interactive Query Input**: Enter your research questions
- **Real-time Results**: View structured research summaries
- **Built-in Guide**: Access help and feature explanations
- **Download Options**: Save research results as text files
- **Progress Indicators**: See research progress in real-time

### Command Line Interface

Run the research assistant in terminal mode:

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

```bash
├── main.py              # Main application script (CLI)
├── app.py               # Streamlit web application
├── tools.py             # Custom tools for search and saving
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (API keys)
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## API Limits & Retry Mechanism

The free tier of Google Gemini API has rate limits:

- 15 requests per minute
- If you hit limits, the application will automatically retry with exponential backoff
- Retry attempts: up to 5 attempts with increasing delays (4s, 8s, 16s, 32s, 60s)
- Retries are triggered for quota exceeded, rate limit, and transient server errors
- If all retries fail, wait a few minutes before trying again or upgrade to a paid plan

### Retry Behavior

- **Automatic**: No user intervention required
- **Smart**: Only retries on recoverable errors (quota, rate limits, server errors)
- **Progressive**: Waits longer between each retry attempt
- **Informative**: Shows retry progress in the console/web interface

## Dependencies

- langchain: Framework for building LLM applications
- langchain-google-genai: Google Gemini integration
- langchain-community: Community tools
- duckduckgo-search: Web search functionality
- wikipedia: Wikipedia API wrapper
- python-dotenv: Environment variable management
- pydantic: Data validation
- streamlit: Web UI framework
- tenacity: Retry mechanism with exponential backoff

## Troubleshooting

- **ModuleNotFoundError**: Ensure all dependencies are installed with `pip install -r requirements.txt`
- **API Quota Exceeded**: Wait for quota reset or enable billing in Google Cloud Console
- **Model Not Found**: Update the model name in `main.py` and `app.py` if Gemini models change (currently using `gemini-1.5-pro`)

## License

This project is for educational purposes. Check individual library licenses for commercial use.
