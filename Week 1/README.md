# Multi-Agent GenAI System (Researcher + Coder)

A small multi-agent system: an **orchestrator** plans a user goal into
ordered subtasks and routes each to a **researcher** agent (Wikipedia
tools) or a **coder** agent (file + code-execution tools).

## Architecture

```
User goal
   |
   v
Orchestrator (Plan->Execute)
   |
   |-- 1. PLAN: single LLM call breaks the goal into ordered steps,
   |            each tagged with an agent ("researcher" | "coder")
   |
   `-- 2. EXECUTE: run steps in order, each agent uses its own
                    internal ReAct-lite tool loop, prior step
                    outputs are passed forward as context
                        |
              +---------+---------+
              |                   |
      Researcher agent      Coder agent
      - wikipedia_search    - read_file
      - wikipedia_get_summary - write_file
                             - list_directory
                             - execute_code
```

Two layers of agentic behavior, deliberately different patterns:
- **Orchestrator level = Plan->Execute** (static plan decided once,
  up front).
- **Per-agent level = ReAct-lite** (each agent reasons -> calls a tool
  -> observes -> decides whether to call another tool or answer, in a
  short loop, since a single subtask can require multiple tool calls).

## Design tradeoffs (useful for interview discussion)

- **Plan->Execute vs ReAct at the top level**: Plan->Execute was chosen
  here because the shape of "research X, then build Y using it" is
  predictable up front. If a step's result could plausibly change what
  later steps *should be* (e.g. research reveals the goal needs
  reframing), ReAct's continuous replanning would be the better fit --
  at the cost of more LLM calls and less predictable execution order.
- **exec() for code execution**: chosen for simplicity in this demo.
  It is genuinely unsandboxed -- any code the LLM generates runs with
  full local permissions. File paths ARE scoped to a workspace
  directory, but that doesn't sandbox exec() itself. A production
  version would run generated code in a subprocess with reduced
  permissions or a container.
- **Static plan, no replanning on failure**: if a step errors out, the
  run continues with whatever the agent returned (including error
  text) rather than stopping or replanning. Flagged as a natural
  extension, not built here.

## Setup

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
python main.py
```

## Extending this

- **Summarization for long runs**: if `context_so_far` in
  `orchestrator.py` grows large across many steps, you've already built
  a conversation-summarization pattern before -- it would slot in right
  where that string is assembled.
- **Replanning**: after each step, optionally ask the planner LLM
  "given this result, does the remaining plan still make sense?" and
  let it patch the plan -- a middle ground between static Plan->Execute
  and full ReAct.
- **More tools per agent**: e.g. a `wikipedia_get_full_content` tool for
  the researcher, or a `run_tests` tool for the coder.
