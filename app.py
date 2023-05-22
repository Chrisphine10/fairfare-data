import os
from constant import huggingFaceKey, openWeatherApiKey, serpApiKey
import streamlit as st

# langchain imports
from langchain import LLMMathChain, SerpAPIWrapper, PromptTemplate, HuggingFaceHub, LLMChain
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.utilities import WikipediaAPIWrapper, OpenWeatherMapAPIWrapper, TextRequestsWrapper

os.environ['HUGGINGFACEHUB_API_TOKEN'] = huggingFaceKey
os.environ['OPENWEATHERMAP_API_KEY'] = openWeatherApiKey
os.environ['SERPAPI_API_KEY'] = serpApiKey

# initialization
search = SerpAPIWrapper()
# maths = LLMMathChain()
weather = OpenWeatherMapAPIWrapper()
wiki = WikipediaAPIWrapper()

# App Framework
st.title('RolexGPT')
question = st.text_input('Enter the command and see magic')

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="""
        Description: This is a tool for querying the web for google search.
        Useful for when you need to answer questions about current events or general knowledge.
        """
    ),
    # Tool(
    #     name="Math",
    #     func=maths.run,
    #     description="""
    #     Description: This is a tool for solving math problems.
    #     Useful for when you need to solve math problems.
    #     """
    # ),
    Tool(
        name="Weather",
        func=weather.run,
        description="""
        Description: This is a tool for querying the weather.
        Useful for when you need to answer questions about the weather.
        """
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="""
        Description: This is a tool for querying wikipedia.
        Useful for when you need to answer questions about general knowledge.
        """
    )
]
llm = HuggingFaceHub(repo_id="google/flan-t5-xl",
                     model_kwargs={"temperature": 0.9,
                                   "max_length": 64})

if prompt:
    thisValue = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    chainResult = LLMChain(prompt=prompt,
                           llm=HuggingFaceHub(repo_id="google/flan-t5-xl",
                                              model_kwargs={"temperature": 0,
                                                            "max_length": 64}))
    with st.expander('Response'):
        st.info(chainResult.run(template))
