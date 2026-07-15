from agents.base_agent import BaseAgent
from tools.wikipedia_tools import TOOL_SCHEMAS, TOOL_FUNCTIONS

RESEARCHER_SYSTEM_PROMPT = """You are the Researcher agent in a multi-agent system.
You have access to Wikipedia search and summary tools. Use them to gather
factual information for the subtask you're given. Cite the Wikipedia page
title(s) you used. Keep your final answer focused and factual -- you are
producing input for another agent (a coder) or for a final report, not a
full essay."""


def build_researcher_agent() -> BaseAgent:
    return BaseAgent(
        name="researcher",
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        tool_schemas=TOOL_SCHEMAS,
        tool_functions=TOOL_FUNCTIONS,
    )
