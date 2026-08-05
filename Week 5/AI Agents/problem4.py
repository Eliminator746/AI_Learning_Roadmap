# Problem 6: HumanInTheLoopMiddleware
# Add a send_email(to: str, body: str) tool (just print instead of actually sending). Wrap it with HumanInTheLoopMiddleware(interrupt_on={"send_email": True}), with a checkpointer + thread_id configured. Invoke the agent with a request that triggers send_email, observe the interrupt, then resume execution with Command(resume={"decisions": [{"type": "approve"}]}). Then try again but resume with a "reject" decision instead, and see how the agent reacts to being told no.


"""
HumanInTheLoopMiddleware — full walkthrough

THE CORE IDEA:
Normally, when an agent's model decides to call a tool, LangGraph just
calls it immediately -- no human ever sees the request first. That's fine
for read-only tools (searching Wikipedia, checking weather), but risky for
tools with real-world side effects (sending an email, deleting a file,
charging a card).

HumanInTheLoopMiddleware sits between "the model decided to call a tool"
and "the tool actually runs." For any tool you mark as needing review, it
PAUSES the graph entirely (this is called an "interrupt"), saves the
current state to the checkpointer, and hands control back to your Python
code. Your code then decides -- based on a real human's input, a UI
button click, whatever -- whether to let the call proceed, edit its
arguments first, or reject it. You express that decision as a Command,
and invoke() a second time to resume exactly where the graph left off.

This is why a checkpointer + thread_id are mandatory here: pausing and
resuming later is only possible because the graph's state was persisted
somewhere in between.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain.agents.middleware import HumanInTheLoopMiddleware
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

MODEL = "gemini-3.6-flash"

model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)


# ---------------------------------------------------------------------------
# Tools. send_email is the "dangerous" one we'll require approval for.
# read_email is harmless -- we'll explicitly mark it as NOT needing review.
# ---------------------------------------------------------------------------

@tool
def read_email(email_id: str) -> str:
    """Read an email by its ID."""
    return f"[MOCK] Email {email_id}: 'Hey, are we still on for Friday?'"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to someone."""
    # In a real tool this would actually send the email. Here we just
    # print, so we can safely test what happens when a human approves
    # vs. rejects the call.
    print(f"\n>>> [SIDE EFFECT] Actually sending email to {to} | subject={subject!r} | body={body!r}\n")
    return f"Email sent to {to} with subject '{subject}'"


# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

agent = create_agent(
    model=model,   # <-- use the Gemini model object you configured, not a hardcoded string
    tools=[read_email, send_email],
    checkpointer=InMemorySaver(),   # REQUIRED for HITL: interrupts need persisted state to resume from
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                # send_email requires human review before it runs.
                # allowed_decisions controls which responses are even legal
                # to send back later -- e.g. you could omit "edit" if you
                # never want to allow tweaking the args, only allow/deny.
                "send_email": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                # read_email is safe -- False means "never interrupt this one,
                # just run it immediately like normal."
                "read_email": False,
            }
        ),
    ],
)


def print_agent_trace(result):
    """Same trace helper as before -- shows each message + any tool calls."""
    if "messages" not in result:
        return
    for msg in result["messages"]:
        role = msg.__class__.__name__
        text = getattr(msg, "text", None)
        content = text if isinstance(text, str) else str(msg.content)
        print(f"[{role}]")
        if content.strip():
            print(f"  content: {content}")
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                print(f"  TOOL CALL -> {tc['name']}({tc['args']})  [id={tc['id']}]")
        print("-" * 60)


# ---------------------------------------------------------------------------
# RUN 1: trigger send_email, then APPROVE it
# ---------------------------------------------------------------------------

print("=" * 70)
print("RUN 1: send_email -> APPROVE")
print("=" * 70)

thread_config_1 = {"configurable": {"thread_id": "approve-thread"}}

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Send an email to john@example.com with subject 'Project Update' and body 'We are on track for Friday.'",
            }
        ]
    },
    thread_config_1,
)

# When HumanInTheLoopMiddleware pauses the graph, the result dict contains
# an "__interrupt__" key instead of (or alongside) a normal finished
# "messages" list. This is how you detect "execution paused, waiting on me."
if "__interrupt__" in result:
    interrupt_payload = result["__interrupt__"][0]
    print("\n>>> INTERRUPTED. Details of what's pending approval:")
    print(interrupt_payload.value)

    # Resume by calling invoke() AGAIN, but this time the "input" isn't a
    # new message -- it's a Command telling the graph how to proceed from
    # where it paused. Must use the SAME thread_config so it resumes the
    # correct paused run.
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        thread_config_1,
    )
    print("\n>>> RESUMED after APPROVE:\n")
    print_agent_trace(result)
else:
    print("\n>>> No interrupt happened -- send_email was not called, or middleware didn't trigger.")
    print_agent_trace(result)


# ---------------------------------------------------------------------------
# RUN 2: trigger send_email again, this time REJECT it
# Uses a DIFFERENT thread_id -- interrupts are tied to one specific paused
# run, so you can't reuse a thread that's already been resumed.
# ---------------------------------------------------------------------------

print("\n" + "=" * 70)
print("RUN 2: send_email -> REJECT")
print("=" * 70)

thread_config_2 = {"configurable": {"thread_id": "reject-thread"}}

result2 = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Send an email to john@example.com with subject 'Project Update' and body 'We are on track for Friday.'",
            }
        ]
    },
    thread_config_2,
)

if "__interrupt__" in result2:
    interrupt_payload = result2["__interrupt__"][0]
    print("\n>>> INTERRUPTED. Details of what's pending approval:")
    print(interrupt_payload.value)

    result2 = agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "Do not send this without my review first."}]}),
        thread_config_2,
    )
    print("\n>>> RESUMED after REJECT:\n")
    print_agent_trace(result2)
else:
    print("\n>>> No interrupt happened.")
    print_agent_trace(result2)