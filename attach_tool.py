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

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
# from langgraph.schema import AIMessage
from langchain_core.messages.ai import AIMessage, ToolCall
import uuid


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

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

@tool
def get_word_length(word: str) -> int:
    """Returns the length of a sentence."""
    print("Response SIZE: ", len(word))
    return len(word)

# def call_model(state: MessagesState):
#     resp = llm_with_tools.invoke(state["messages"])
#     return {"messages": [resp]}

# def call_model(state: MessagesState):
#     # Just use llm normally
#     resp = llm.invoke(state["messages"])
#     return {"messages": [resp]}


def call_model(state):
    # Assume state["messages"] is a list of dicts with "role" and "content"
    messages = state["messages"]
    # print("Hossam ", messages)
    # Use your LLM to generate a response
    # You can pass the whole conversation or just the last user message
    # prompt = "\n".join([f'{m["role"]}: {m["content"]}' for m in messages])
    prompt = messages[0].content
    
    # Get LLM output
    ai_response = llm.invoke(prompt)  # returns string
    # print("AI response: ", ai_response)
    # Append AIMessage to state
    # state["messages"].append(AIMessage(content=ai_response))

     # Create a unique ID for the tool call
    # tool_call_id = str(uuid.uuid4())
    # Build ToolCall object
    tool_call = ToolCall(
        id=str(uuid.uuid4()),
        name="get_word_length",  # must match @tool name
        args={"word": ai_response}
    )

    state["messages"].append(
        AIMessage(
            content=ai_response,  # no normal text content
            tool_calls=[
                tool_call
            ]
        )
    )

    return state

# Conditional edge function
# def should_continue(state: MessagesState):
#     if state["messages"][-1].tool_calls:
#         return "tools"
#     return END
def should_continue(state: MessagesState):
    last_msg = state["messages"][-1]
    print("Should continue: ", last_msg)
    content = getattr(last_msg, "content", "")
    print("Should continue: ", content)
    print("*********** STOP: ", isinstance(content, str))

    # If model output contains our manual trigger, route to tools
    # if isinstance(content, str) and "multiply(" in content:
    if isinstance(content, str):
        return "tools"

    return END



# Main script:
model_id = "/home/hossamamer/TTC_checkpoints/TTC-checkpoints/tinyllama-math-code-checkpoint-300"


# Build the graph and load the model
print("Building the graph and loading the model")
model, pipe = load_model(model_id=model_id)

# Define the LLM
print("Define the llm")
llm = HuggingFacePipeline(pipeline=pipe)

# Create the tool node
tool_node = ToolNode([get_word_length])
# tool_node = ToolNode([multiply])

# Build the state graph
graph = StateGraph(MessagesState)

# Add nodes
graph.add_node("call_model", call_model)  # LLM node
graph.add_node("tools", tool_node)        # Tool node

# Define edges
graph.add_edge(START, "call_model")  # Start -> LLM
graph.add_edge("call_model", "tools")    # LLM -> End
graph.add_edge("tools", END)

# # Add nodes (loops from tools to call model)
# graph.add_node("call_model", call_model)   # LLM node
# graph.add_node("tools", tool_node)         # Tool node

# # Define edges
# graph.add_edge(START, "call_model")  # Start -> LLM
# graph.add_conditional_edges(
#     "call_model",
#     should_continue,
#     ["tools", END]  # LLM -> Tool or LLM -> End
# )
# graph.add_edge("tools", "call_model")  # Tool -> LLM (loop if needed)

print("Graph built successfully!")

# Compile the graph
print("Compile the graph")
_graph_compiled = graph.compile()

# Visualize the graph (optional)
try:
    img_data = _graph_compiled.get_graph().draw_mermaid_png()
    with open("tool.png", "wb") as f:
        f.write(img_data)
    print("Tool graph is saved as tool.png")
except Exception as e:
    print("Graph visualization failed:", e)

# Initial state: only a user message
initial_state = {
    "messages": [{"role": "user", "content": "Please multiply(3, 4)"}]
}

# Invoke the compiled graph
out = _graph_compiled.invoke(initial_state)

# Print the final AI response
print("Final response:", out["messages"][-1].content)

