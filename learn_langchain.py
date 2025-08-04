import sys, os


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate



# model_id = "TinyLlama/TinyLlama_v1.1"
# model_id = "/home/hossamamer/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062/"
model_id = "/home/hossamamer/TTC_checkpoints/TTC-checkpoints/tinyllama-math-code-checkpoint-300"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda:0"  # auto-placement on available devices (GPU/CPU)
)

print("Model: ")
print(model)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)

llm = HuggingFacePipeline(pipeline=pipe)

template = "You are an artificial intelligence assistant, answer the question. {question}"
prompt = PromptTemplate.from_template(template=template)

# Set up the chain
llm_chain = prompt | llm

question = "How does LangChain make LLM application development easier?"
print("------------------------------")
print("Response:")
print(llm_chain.invoke({"question": question}))








