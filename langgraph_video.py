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

class VQState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    # messages: Annotated[list, add_messages]
    video_path: str   = "/path/to/video-1.mp4"
    cosine_sim: float = 0.0
    is_accepted: bool = False
    count_self_reflect: int = 0

def generate_video(state: VQState):
    state["video_path"] = "../AskVideos-VideoCLIP/data/a_review_of_a_phone_0X0Jm8QValY.mp4"
    print("Video has been generated!")
    return state


def evaluate_video(state: VQState):
    print("Video has been evaluated!")
    state['cosine_sim'] = 0.1
    return state

def self_reflect(state: VQState):
    state["is_accepted"] = state["cosine_sim"] > 0.5
    if not state["is_accepted"]:
        print("Quality: Bad!")
        state['count_self_reflect'] = state['count_self_reflect'] + 1
    else:
        print("Quality: Good")
    return state

def should_stop(state: VQState):
    if state["is_accepted"] or state['count_self_reflect'] > 5:
        return END
    return "generate_video"


# Create an instance of the state
# vq_state = VQState()

# Build the state graph
vq_graph = StateGraph(VQState)

# Add nodes
vq_graph.add_node("generate_video", generate_video)  # LLM node
vq_graph.add_node("evaluate_video", evaluate_video)        # Tool node
vq_graph.add_node("self_reflect", self_reflect)



# Define edges
vq_graph.add_edge(START, "generate_video")  # Start -> LLM
vq_graph.add_edge("generate_video", "evaluate_video")    # LLM -> End
vq_graph.add_edge("evaluate_video", "self_reflect")
vq_graph.add_conditional_edges(
    "self_reflect",
    should_stop,
    [END, "generate_video"]  # LLM -> Tool or LLM -> End
)

# vq_graph.add_conditional_edges(
#     "self_reflect",
#     should_stop,
#     {
#         True: END,
#         False: "generate_video",
#     },
# )


print("Graph built successfully!")

# Compile the graph
print("Compile the graph")
_graph_compiled = vq_graph.compile()

# Visualize the graph (optional)
try:
    img_data = _graph_compiled.get_graph().draw_mermaid_png()
    with open("video_gen.png", "wb") as f:
        f.write(img_data)
    print("Tool graph is saved as video_gen.png")
except Exception as e:
    print("Graph visualization failed:", e)


# Initial state: only a user message
initial_state = {
    "messages": [{"role": "user", "content": "Please multiply(3, 4)"}]
}

# Create an instance of the state
vq_state_instance = VQState(
    video_path="../AskVideos-VideoCLIP/data/a_review_of_a_phone_0X0Jm8QValY.mp4",
    cosine_sim=0.0,
    is_accepted=False,
    count_self_reflect=0
)

# Invoke the compiled graph
out = _graph_compiled.invoke(vq_state_instance)

print("Final response:", out)
