"""
Demo entry point -- run this to see the Plan->Execute multi-agent system
in action. Requires GEMINI_API_KEY set in your environment.
"""
from orchestrator import run

if __name__ == "__main__":
    goal = (
        "Research the history of the Transformer architecture in NLP, "
        "then write a Python script that prints a markdown-formatted "
        "timeline of its key milestones, and save that script to a file "
        "called timeline.py in the workspace."
    )
    result = run(goal)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for r in result["results"]:
        print(f"\nStep {r['step']} ({r['agent']}): {r['subtask']}")
        print(f"-> {r['output']}")
