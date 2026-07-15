"""
BaseAgent: a small ReAct-style tool-calling loop used INSIDE each agent
(researcher, coder).

This is a different layer from the orchestrator's Plan->Execute strategy:
the orchestrator decides WHICH agent handles WHICH subtask up front
(Plan->Execute, static), but each agent still needs to decide WHICH of
its OWN tools to call -- possibly more than one, in sequence -- to finish
that subtask. That's naturally a short local ReAct loop (reason, call a
tool, look at the result, decide whether to call another tool or answer).
So: Plan->Execute at the top level, ReAct-lite inside each agent.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import client, MODEL_NAME

MAX_TOOL_ITERATIONS = 5


class BaseAgent:
    def __init__(self, name: str, system_prompt: str, tool_schemas: list, tool_functions: dict):
        self.name = name
        self.system_prompt = system_prompt
        self.tool_schemas = tool_schemas
        self.tool_functions = tool_functions

    def run(self, subtask: str, context: str = "") -> str:
        """
        Execute one subtask. `context` is prior orchestrator step output,
        passed in so later steps can build on earlier research/code.
        Returns the agent's final text answer for this subtask.
        """
        user_content = subtask
        if context:
            user_content = f"Context from previous steps:\n{context}\n\nYour subtask:\n{subtask}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        for _ in range(MAX_TOOL_ITERATIONS):
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=self.tool_schemas,
            )
            msg = response.choices[0].message
            # Normalize the SDK object to a dict at append time -- same
            # pattern as before, avoids type mismatches on the next turn.
            messages.append(msg.model_dump())

            if not msg.tool_calls:
                return msg.content or ""

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                fn = self.tool_functions.get(fn_name)
                if fn is None:
                    result = f"Unknown tool '{fn_name}'"
                else:
                    try:
                        result = fn(**fn_args)
                    except Exception as e:
                        result = f"Tool error: {type(e).__name__}: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                })

        return "(agent hit max tool iterations without a final answer)"
