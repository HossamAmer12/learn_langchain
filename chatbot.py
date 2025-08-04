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


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]



def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="cuda:0"  # auto-placement on available devices (GPU/CPU)
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2
    )

    return model, pipe

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            # print("DEBUG: ", type(value["messages"][-1]), value["messages"][-1])
            # print("Assistant:", value["messages"][-1].content)
            print("Assistant:", value["messages"][-1])


# Main script:
model_id = "/home/hossamamer/TTC_checkpoints/TTC-checkpoints/tinyllama-math-code-checkpoint-300"


# Build the graph with some states
print("Building the graph and loading the model")
graph_builder = StateGraph(State)
model, pipe = load_model(model_id=model_id)

# Define the llm
print("Define the llm")
llm = HuggingFacePipeline(pipeline=pipe)

# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)

# Add an entry point to tell the graph where to start its work each time it is run:
graph_builder.add_edge(START, "chatbot")

# Add an exit point to indicate where the graph should finish execution. 
graph_builder.add_edge("chatbot", END)

# Compile the graph
print("Compile the graph")
graph = graph_builder.compile()


# Visualize the graph
# try:
#     # display(Image(graph.get_graph().draw_mermaid_png()))
#     # Generate the image bytes
#     img_data = graph.get_graph().draw_mermaid_png()
#     # Save image to file
#     with open("chatbot.png", "wb") as f:
#         f.write(img_data)
#     print("chatbot graph is saved")
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass


# Run the chatbot:
print("Run the chatbot...")
while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
    

