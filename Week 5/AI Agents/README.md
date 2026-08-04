# AI Agents Practice Folder

This folder contains small hands-on examples for learning LangChain/LangGraph agents, tools, memory, and middleware.

The main goal of these files is to understand how an agent can:

- use tools to answer questions,
- remember past context across conversations,
- add safety or approval steps before sensitive actions,
- and wrap tool execution with custom logic.

## Files Covered

### 1. problem1.py - Minimal create_agent

**Topic:** create_agent, tools, multi-step reasoning

**What we are solving:**
Build a simple agent that uses multiple tools to answer a question that needs more than one step. This helps us understand how an agent chooses tools and how it can reason through a multi-part query.

### 2. problem2.py - Checkpointer + thread_id with InMemorySaver

**Topic:** InMemorySaver, checkpointer, thread_id, conversation memory

**What we are solving:**
Show how an agent can remember context within a conversation and how memory is scoped using thread_id. We test two different sessions to see whether the agent remembers previous information.

### 3. problem3.py - Tool-call middleware with retry logic

**Topic:** wrap_tool_call middleware, tool logging, retry handling

**What we are solving:**
Wrap tool execution with custom middleware so we can log tool calls and retry failed tool actions automatically. This teaches how to add error-handling logic at the agent level.

### 4. problem4.py - HumanInTheLoopMiddleware

**Topic:** HumanInTheLoopMiddleware, approval flow, interrupt/resume

**What we are solving:**
Demonstrate a human approval step before a sensitive tool action such as sending an email. The agent pauses, waits for a human decision, and then resumes based on whether the action is approved or rejected.

## Key Concepts Learned

- Agent creation with create_agent
- Tool integration in agents
- Multi-step tool calling
- Conversation memory with InMemorySaver
- Thread-based memory scoping using thread_id
- Middleware for monitoring and retrying tool calls
- Human-in-the-loop interruptions for approvals

## Quick Summary

These files together show the journey from a basic agent to a more interactive and controlled agent that can think, remember, retry, and ask for human approval when needed.
