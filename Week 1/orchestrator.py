"""
Orchestrator: Plan->Execute strategy.

  1. PLAN: ask the LLM to break the user's goal into an ordered list of
     subtasks, each tagged with which agent should handle it.
  2. EXECUTE: run each subtask through the assigned agent in order,
     passing forward previous steps' outputs as context (so e.g. the
     coder can use what the researcher found).

This is intentionally a **static** plan -- decided once, up front, before
any execution happens. That's the key difference from ReAct, where the
system would re-decide the next action after every observation. Plan->
Execute is a good fit when subtasks are fairly predictable in advance
(like this one: research then write code). If a step's result could
plausibly change what later steps *should be*, ReAct's replanning is the
better fit -- worth having that tradeoff ready for interviews.

NOT built here (flagged for you to add if you want it):
  - Replanning if a step fails or returns something unusable.
  - Retry logic on individual tool/agent errors.
  - Conversation-level summarization if a run has many steps and context
    balloons -- you've already built this pattern before, would drop in
    cleanly at the point where `context_so_far` is assembled below.
"""
import json
from config import client, MODEL_NAME
from agents.researcher_agent import build_researcher_agent
from agents.coder_agent import build_coder_agent

AGENTS = {
    "researcher": build_researcher_agent(),
    "coder": build_coder_agent(),
}

AGENT_CAPABILITIES = """
- researcher: can search Wikipedia and pull factual summaries. Good for
  gathering background info, definitions, historical facts, explanations.
- coder: can read/write files in a workspace and execute Python code.
  Good for producing scripts, writing files (e.g. markdown reports,
  generated code), or running computations.
"""

PLANNER_SYSTEM_PROMPT = f"""You are the planning module of a multi-agent
system. Given a user's goal, break it into an ORDERED list of subtasks.
Each subtask must be assigned to exactly one of these agents:
{AGENT_CAPABILITIES}

Respond with ONLY valid JSON (no markdown fences, no commentary), in this
exact shape:
{{
  "steps": [
    {{"agent": "researcher", "subtask": "..."}},
    {{"agent": "coder", "subtask": "..."}}
  ]
}}

Rules:
- Only use "researcher" or "coder" as agent values.
- Keep subtasks concrete and self-contained enough for that agent to
  execute using only its own tools.
- If the goal needs research before code can be written, put the
  research step(s) first.
"""


def make_plan(goal: str) -> list:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": goal},
        ],
    )
    raw = response.choices[0].message.content.strip()
    # Defensive: strip accidental markdown fences even though we asked not to
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner returned invalid JSON: {raw!r}") from e

    steps = plan.get("steps", [])
    if not steps:
        raise ValueError(f"Planner returned no steps: {raw!r}")
    for step in steps:
        if step.get("agent") not in AGENTS:
            raise ValueError(f"Planner assigned unknown agent: {step}")
    return steps


def execute_plan(steps: list) -> list:
    """Run each step in order, feeding forward prior outputs as context."""
    results = []
    context_so_far = ""

    for i, step in enumerate(steps, start=1):
        agent = AGENTS[step["agent"]]
        print(f"\n[Step {i}/{len(steps)}] agent={step['agent']} -> {step['subtask']}")

        output = agent.run(subtask=step["subtask"], context=context_so_far)
        print(f"[Step {i} output]\n{output}\n")

        results.append({
            "step": i,
            "agent": step["agent"],
            "subtask": step["subtask"],
            "output": output,
        })
        context_so_far += f"\n[Step {i} - {step['agent']}] {step['subtask']}\nResult: {output}\n"

    return results


def run(goal: str) -> dict:
    print(f"Goal: {goal}\n")
    steps = make_plan(goal)
    print("Plan:")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. [{s['agent']}] {s['subtask']}")

    results = execute_plan(steps)
    return {"goal": goal, "plan": steps, "results": results}
