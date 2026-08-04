from langchain_exa import ExaSearchResults
from dotenv import load_dotenv
import os
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

# Initialize the ExaSearchResults tool
search_tool = ExaSearchResults(exa_api_key=os.environ["EXA_API_KEY"])

# Perform a search query
search_results = search_tool._run(
    query="When was the last time the New York Knicks won the NBA Championship?",
    num_results=2,
    text_contents_options=True,
    highlights=True,
)

print("Search Results:", search_results)