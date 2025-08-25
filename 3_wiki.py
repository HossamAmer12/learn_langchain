from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from langchain_community.tools import TavilySearchResults
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import json

# ---- Define state ----
class State(TypedDict):
    user_prompt: str
    search_results: List[Dict[str, Any]]
    enhanced_prompt: str
    city: str

# ---- Tools ----
search_tool = TavilySearchResults(max_results=5, tavily_api_key="YOUR_TAVILY_API_KEY")

# Load Qwen on CPU
qwen_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    device=-1,              # CPU
    torch_dtype="auto",
    max_new_tokens=512
)
llm = HuggingFacePipeline(pipeline=qwen_pipeline)

# ---- Nodes ----
# def search_node(state: State):
#     query = f"Touristic places and cultural info related to: {state['user_prompt']}"
#     results = search_tool.invoke(query)   # use invoke instead of run for structured
#     return {"search_results": results}

# def search_node(state: State):
#     query = f"Touristic places and cultural info related to: {state['user_prompt']}"
#     results = search_tool.invoke(query)  # this will now be a list[dict]
#     # Convert to clean JSON-safe structure
#     structured = [
#         {
#             "title": r.get("title", ""),
#             "url": r.get("url", ""),
#             "content": r.get("content", "")
#         }
#         for r in results
#     ]
#     return {"search_results": structured}


from tavily import TavilyClient

tavily = TavilyClient(api_key="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")

def search_node(state: State):
     
    state['city'] = "Paris"
    # query = f"Touristic places and cultural info related to: {state['user_prompt']}"
    query = f"Touristic places: {state['city']}"
    response = tavily.search(query, max_results=5)  # returns JSON
    # response is already a dict with "results": [{title, url, content, score}, ...]
    print("00000000000000000000000000000000000000000")
    print("Tavily response: ", response)
    print("00000000000000000000000000000000000000000")
    return {"search_results": response["results"]}

# def enhance_node(state: State):
#     return state

def enhance_node(state: State):
    snippets = []

    for r in state["search_results"]:
        if isinstance(r, dict):  # structured result (title/content/etc.)
            # title = r.get("title", "")
            content = r.get("content", "")
            # snippets.append(f"- {title}: {content[:200]}...")
            snippets.append(f"-{content[:1000]}")
        else:  # plain string
            snippets.append(f"- {str(r)[:200]}...")

    context = "\n".join(snippets)

    prompt = f"""
    You are helping to enrich prompts for a video generator.

    Original prompt: {state['user_prompt']}

    Related tourist information:
    {context}

    Rewrite the original prompt into an enriched single-sentence prompt
    that highlights key landmarks, cultural events, and the summer atmosphere.

    Respond ONLY in JSON:
    {{
        "enhanced_prompt": "..."
    }}
    """

    enriched = llm(prompt)

    import json
    try:
        parsed = json.loads(enriched.strip().split("```")[-1])
        return {"enhanced_prompt": parsed["enhanced_prompt"]}
    except Exception:
        return {"enhanced_prompt": enriched}


def output_node(state: State):
    print("\n🎬 Final Enhanced Prompt:\n", state["enhanced_prompt"])
    # return {}
    return state

# ---- Build Graph ----
workflow = StateGraph(State)

workflow.add_node("search", search_node)
workflow.add_node("enhance", enhance_node)
workflow.add_node("output", output_node)

workflow.set_entry_point("search")
workflow.add_edge("search", "enhance")
workflow.add_edge("enhance", "output")
workflow.add_edge("output", END)

app = workflow.compile()

# ---- Run Graph ----
user_input = "Generate a video about Paris in summer"
result = app.invoke({"user_prompt": user_input})
