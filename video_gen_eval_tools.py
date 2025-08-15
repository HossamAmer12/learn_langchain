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

import os
from typing import List
import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from modeling import VideoCLIP_XL
from utils.text_encoder import text_encoder

from video_eval import runEval

# THRESHOLD           = 0.5
THRESHOLD           = 10
STATIC_VIDEO_PATH   = "../AskVideos-VideoCLIP/data/a_review_of_a_phone_0X0Jm8QValY.mp4"
# STATIC_INPUT_PROMPT = "a_review_of_a_phone_0X0Jm8QValY"
STATIC_INPUT_PROMPT = "cooking video"


class VQState(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    # messages: Annotated[list, add_messages]
    video_path: str   = "/path/to/video-1.mp4"
    cosine_sim: float = 0.0
    is_accepted: bool = False
    count_self_reflect: int = 0
    input_prompt: str = ""
    weather: str = ""
    _time: str = ""

def generate_video(state: VQState):
    state["video_path"] = STATIC_VIDEO_PATH
    state["input_prompt"] = STATIC_INPUT_PROMPT
    # state["input_prompt"] = "cooking video"
    print(f"Iteration={state['count_self_reflect']} Video has been generated!")
    return state


def prompt_rewrite(state: VQState):
    state["input_prompt"] = STATIC_INPUT_PROMPT
    print("Prompt Rewrite: Time and weather have been collected. Rewriting prompt..")
    return state

def planning_agent(state: VQState):
    print("Planning agent is in progress..")
    return state


def evaluate_video(state: VQState):
    state['cosine_sim'] = runEval(video = state['video_path'], 
                                  input_prompt = state['input_prompt'])
    print(f"Iteration={state['count_self_reflect']} Video has been evaluated!")
    return state

def self_reflect(state: VQState):
    state["is_accepted"] = state["cosine_sim"] > THRESHOLD
    if not state["is_accepted"]:
        print("Quality: Bad!")
        state['count_self_reflect'] = state['count_self_reflect'] + 1
    else:
        print("Quality: Good")
    return state

def should_stop(state: VQState):
    if state["is_accepted"] or state['count_self_reflect'] > 3:
        return END
    return "prompt_rewrite"


def get_weather(state: VQState) -> int:
    """Returns the length of a sentence."""
    word = state["input_prompt"]
    print("Weather is sunny")
    return {"weather": "sunny"}

def get_time(state: VQState) -> int:
    """Returns the length of a sentence."""
    word = state["input_prompt"]
    print("Time is 12:45")
    return {"_time": "12:45"}

# Create an instance of the state
# vq_state = VQState()

# Build the state graph
vq_graph = StateGraph(VQState)

weather_node = ToolNode([get_weather])
time_node    = ToolNode([get_time])


# Add nodes

vq_graph.add_node("generate_video", generate_video)  # LLM node
vq_graph.add_node("planning_agent", planning_agent)
vq_graph.add_node("get_weather", get_weather)
vq_graph.add_node("get_time", get_time)
vq_graph.add_node("prompt_rewrite", prompt_rewrite)
vq_graph.add_node("evaluate_video", evaluate_video)        # Tool node
vq_graph.add_node("self_reflect", self_reflect)



# Define edges
vq_graph.add_edge(START, "planning_agent")  # Start -> LLM

vq_graph.add_edge("planning_agent", "get_weather")  # Start -> LLM
vq_graph.add_edge("planning_agent", "get_time")  # Start -> LLM


# Parallel branch after generate_video
vq_graph.add_edge("get_weather", "prompt_rewrite")
vq_graph.add_edge("get_time", "prompt_rewrite")


vq_graph.add_edge("prompt_rewrite", "generate_video")

# Merge before evaluate_video
vq_graph.add_edge("generate_video", "evaluate_video")
# vq_graph.add_edge("get_time", "evaluate_video")

# Branch after generate_video
# vq_graph.add_edges("generate_video", ["get_weather", "get_time"])

# Merge into a single node before evaluate_video
# vq_graph.add_edges(["get_weather", "get_time"], "evaluate_video")
vq_graph.add_edge("evaluate_video", "self_reflect")
vq_graph.add_conditional_edges(
    "self_reflect",
    should_stop,
    [END, "prompt_rewrite"]  # LLM -> Tool or LLM -> End
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
    count_self_reflect=0,
    input_prompt="test",
    weather = "Sunny",
    _time = "12:00pm"
)

# Invoke the compiled graph
out = _graph_compiled.invoke(vq_state_instance)

print("Final response:", out)
