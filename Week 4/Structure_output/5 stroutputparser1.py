from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatOpenAI()

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser
# Template1 prompt h uss prompt ko model ko do, wo model uska response bhejega usko parse karke content string mae nikalo, fir uss str ko template2 ko bhej do modelko and fir parse karo aur final response str mae le lo

result = chain.invoke({'topic':'black hole'})

print(result)

