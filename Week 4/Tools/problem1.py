from langchain_core.messages import HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)

MODEL = "gemini-3.6-flash"  # current model id, no change needed

model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)


# Write a @tool-decorated function get_word_length(word: str) -> int. Bind it to your Gemini model with .bind_tools([get_word_length]). Invoke with a prompt like "How many letters are in 'langchain'?" and inspect response.tool_calls — don't execute the tool yet, just print what the model wants to call and with what arguments + Complete the loop manually

@tool
def get_word_length(word:str) -> int:
    """Returns length of the word"""
    return len(word)

llm_with_tool=model.bind_tools([get_word_length])

# question = "How many letters are there in 'langchain'?"
question = "What's the capital of France?"
messages = [HumanMessage(content=question)]
res = llm_with_tool.invoke(messages)
messages.append(res)  # the AIMessage containing tool_calls

print(res)
print(f'{"--" * 20}\n{"--" * 20}')

if res.tool_calls: # Why if -> crashes if the model answers directly without calling any tool.
    tool_call = res.tool_calls[0]
    word = tool_call["args"]["word"]

    result = get_word_length.invoke(tool_call["args"])
    print("Word:", word)
    print("Length:", result)

    tool_mess = ToolMessage(
        content=str(result),
        tool_call_id=tool_call['id']
    )

    messages.append(tool_mess)

    print(messages)
    print(f'{"--" * 20}\n{"--" * 20}')

    final_response = model.invoke(messages)
    print("final_response: ", final_response.text())


else:
    print(res.text)