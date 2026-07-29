# NakliLLM, NakliPromptTemplate, NakliLLMChain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from pathlib import Path
import random

env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)

MODEL="gemini-3.6-flash"


model = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0,
)


class NakliLLM2:
    def __init__(self):
        pass
    def predict(self, prompt):
        response_list = [
                "The product exceeded my expectations and works flawlessly.",
                "Customer support was responsive and resolved my issue quickly.",
                "The interface is intuitive, making it easy to use from the first day.",
                "The capital of India is New Delhi"
            ]
        
        return {'response': random.choice(response_list)}

class NakliPromptTemplate2:
    def __init__(self, template, input_parameter):
        self.template = template
        self.input_parameter = input_parameter
    
    def run(self, new_dict):
        return self.template.format(**new_dict)

llm = NakliLLM2()
res = llm.predict("What is the capital of New Delhi?")
# print(res)

prompt = NakliPromptTemplate2(
    template="What is the capital of {country}?",
    input_parameter= ['India']
)

res_pr = prompt.run({"country": "India"})
# print(res_pr)
    
    




    
# Runnable implementation      
from abc import ABC, abstractmethod
class Runnable(ABC):

  @abstractmethod
  def invoke(input_data):
    pass

import random

class NakliLLM(Runnable):

  def __init__(self):
    print('LLM created')

  def invoke(self, prompt):
    response_list = [
        'Delhi is the capital of India',
        'IPL is a cricket league',
        'AI stands for Artificial Intelligence'
    ]

    return {'response': random.choice(response_list)}


  def predict(self, prompt):

    response_list = [
        'Delhi is the capital of India',
        'IPL is a cricket league',
        'AI stands for Artificial Intelligence'
    ]

    return {'response': random.choice(response_list)}

class NakliPromptTemplate(Runnable):

  def __init__(self, template, input_variables):
    self.template = template
    self.input_variables = input_variables

  def invoke(self, input_dict):
    return self.template.format(**input_dict)

  def format(self, input_dict):
    return self.template.format(**input_dict)

class NakliStrOutputParser(Runnable):

  def __init__(self):
    pass

  def invoke(self, input_data):
    return input_data['response']

class RunnableConnector(Runnable):

  def __init__(self, runnable_list):
    self.runnable_list = runnable_list

  def invoke(self, input_data):

    for runnable in self.runnable_list:
      input_data = runnable.invoke(input_data)

    return input_data

template = NakliPromptTemplate(
    template='Write a {length} poem about {topic}',
    input_variables=['length', 'topic']
)








llm = NakliLLM()
parser = NakliStrOutputParser()
chain = RunnableConnector([template, llm, parser])
chain.invoke({'length':'long', 'topic':'india'})
template1 = NakliPromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)
template2 = NakliPromptTemplate(
    template='Explain the following joke {response}',
    input_variables=['response']
)
llm = NakliLLM()
parser = NakliStrOutputParser()
chain1 = RunnableConnector([template1, llm])
chain2 = RunnableConnector([template2, llm, parser])
final_chain = RunnableConnector([chain1, chain2])
final_chain.invoke({'topic':'cricket'})
