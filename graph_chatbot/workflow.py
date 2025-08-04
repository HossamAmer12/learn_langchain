from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from PIL import Image
from IPython.display import Image, display
from random import random

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]

def isEven(state):
    # Generate a random number
    num = random.randint(0, 100)
    state['decision'] = (num%2 == 0) # turns on if even
   

def happy_node():
    prompt_prefix = "You are a happy agent. Please answer the user in a friendly way. The user question is {question}"
    llm = 
    chain = prompt | llm
    happy_answer = chain.invoke(state['query'])
    return None

def sad_node():
    return None


# Compile the graph
print("Compile the graph")
graph = graph_builder.compile()


