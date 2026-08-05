# Tools Practice Folder

This folder contains hands-on examples for learning how tools work with LangChain and LangGraph agents.

The main idea of this section is to understand how an LLM can:

- bind tools to a model,
- choose the right tool for a user request,
- validate tool input through schemas,
- handle tool errors gracefully,
- and access conversation state inside a tool.

## Files Covered

### 1. problem1.py - Basic tool binding

**Topic:** tool decorator, bind_tools, manual tool-call loop

**What we are solving:**
Learn the basic flow of connecting a tool to a model, inspecting the model's tool call request, and manually executing the tool step by step.

### 2. problem2.py - Multiple tools and tool selection

**Topic:** multiple tools, tool selection, tool execution

**What we are solving:**
Show how a model chooses among several tools based on the user's question and how the selected tool is executed with the correct arguments.

### 3. problem3.py - Pydantic schema validation

**Topic:** args_schema, Pydantic models, field validation

**What we are solving:**
Demonstrate how tool inputs can be validated before execution using a schema, so invalid or incomplete values are rejected or handled properly.

### 4. problem4.py - Tool error handling

**Topic:** exception handling, graceful tool failure, ToolMessage

**What we are solving:**
Practice handling tool failures without crashing the whole flow. If a tool raises an error, we return an informative message so the model can respond more gracefully.

### 5. problem5.py - create_agent with ToolRuntime

**Topic:** create_agent, ToolRuntime, conversation state access

**What we are solving:**
Move from manual tool loops to a full agent workflow and show how a tool can read the conversation history from runtime state without the LLM passing that history explicitly.

### 6. tool.py - Shared helper / reference file

**Topic:** reusable tool patterns

**What we are solving:**
Serve as a supporting file for experimenting with tool definitions and reusable patterns used across the exercises.

## Key Concepts Learned

- Creating tools with the @tool decorator
- Binding tools to an LLM
- Manual tool-call loops
- Tool selection by the model
- Schema-based validation with Pydantic
- Error handling for failed tools
- Using ToolRuntime to access agent state

## Quick Summary

These examples take you from the simplest tool usage to more advanced patterns such as validation, error handling, and state-aware tools inside an agent.
