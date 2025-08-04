import sys, os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# model_id = "TinyLlama/TinyLlama_v1.1"
model_id = "/home/hossamamer/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062/"
# model_id = "/home/hossamamer/TTC_checkpoints/TTC-checkpoints/tinyllama-math-code-checkpoint-300"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda:0"  # auto-placement on available devices (GPU/CPU)
)

print("TinyLlama Model loaded: ")
print(model)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.1,
    top_p=0.9,
    repetition_penalty=1.2
)

llm = HuggingFacePipeline(pipeline=pipe)

template_agent1 = "I want to learn how to {activity}. Can you suggest how I can learn this step-by-step?"
prompt = PromptTemplate.from_template(template=template_agent1)

learning_prompt = PromptTemplate(
    input_variables=["activity"],
    template=template_agent1
)


template_agent2 = "I only have one week. Can you create a concise plan to help me hit this goal: {learning_plan}." 
time_prompt = PromptTemplate(
    input_variables=["learning_plan"],
    template= template_agent2
)

# Complete the sequential chain with LCEL
seq_chain = ({"learning_plan": learning_prompt | llm | StrOutputParser()}
    | time_prompt
    | llm
    | StrOutputParser())

# Call the chain
print("------------------------------")
print("Response:")
print(seq_chain.invoke({"activity": "play the harmonica"}))

