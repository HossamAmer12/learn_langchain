import wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline



# def load_model(model_id):
#     pipe = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct")
#     return pipe


def load_model(model_id):
    # pipe = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct")
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Load model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",        # force CPU
        torch_dtype="auto"       # let it pick correct dtype
    )
    return tokenizer, model

# # Initialize wrapper
# wiki = WikipediaAPIWrapper(
#     lang="en",          # Wikipedia language
#     top_k_results=1,    # how many search results
#     doc_content_chars_max=100*1e6  # max content length
# )

# # # Example query
# query = "Tourist attractions in Paris"
# results = wiki.run(query)

# print("************************")
# print("Results:  ")
# print("************************")
# print(results)

# 1. Init Wikipedia wrapper
wiki = WikipediaAPIWrapper(lang="en", doc_content_chars_max=10000)

# 2. Search for relevant page
query = "Tourist attractions in Paris"
search_results = wikipedia.search(query)

best_page = search_results[0]   # top result (usually "List of tourist attractions in Paris")
print("Best page:", best_page)

# 3. Load page content
docs = wiki.load(best_page)
best_page_content = docs[0].page_content
print(best_page_content[:800])   # preview

# 4. Use Qwen to extract attractions list
model_id = "Qwen/Qwen2.5-3B-Instruct"

# pipe = load_model(model_id=model_id)
tokenizer, model = load_model(model_id=model_id)

prompt = """
You are given Wikipedia content about tourist attractions in Paris.

From the text, extract a clean bullet-point list of the most famous tourist attractions, landmarks, and museums.
Do not include explanations, just list names.

Wikipedia content:
{page_content}
""".format(page_content=best_page_content[:2000])  # truncate if too long

# result = pipe(prompt, max_new_tokens=300, do_sample=False)
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
# 5. Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False
)

# 6. Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)

import wikipedia
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline



# def load_model(model_id):
#     pipe = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct")
#     return pipe


def load_model(model_id):
    # pipe = pipeline("text-generation", model="Qwen/Qwen2.5-3B-Instruct")
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # 2. Load model on CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",        # force CPU
        torch_dtype="auto"       # let it pick correct dtype
    )
    return tokenizer, model

# # Initialize wrapper
# wiki = WikipediaAPIWrapper(
#     lang="en",          # Wikipedia language
#     top_k_results=1,    # how many search results
#     doc_content_chars_max=100*1e6  # max content length
# )

# # # Example query
# query = "Tourist attractions in Paris"
# results = wiki.run(query)

# print("************************")
# print("Results:  ")
# print("************************")
# print(results)

# 1. Init Wikipedia wrapper
wiki = WikipediaAPIWrapper(lang="en", doc_content_chars_max=10000)

# 2. Search for relevant page
query = "Tourist attractions in Paris"
search_results = wikipedia.search(query)

best_page = search_results[0]   # top result (usually "List of tourist attractions in Paris")
print("Best page:", best_page)

# 3. Load page content
docs = wiki.load(best_page)
best_page_content = docs[0].page_content
print(best_page_content[:800])   # preview

# 4. Use Qwen to extract attractions list
model_id = "Qwen/Qwen2.5-3B-Instruct"

# pipe = load_model(model_id=model_id)
tokenizer, model = load_model(model_id=model_id)

prompt = """
You are given Wikipedia content about tourist attractions in Paris.

From the text, extract a clean bullet-point list of the most famous tourist attractions, landmarks, and museums.
Do not include explanations, just list names.

Wikipedia content:
{page_content}
""".format(page_content=best_page_content[:2000])  # truncate if too long

# result = pipe(prompt, max_new_tokens=300, do_sample=False)
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
# 5. Generate response
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=False
)

# 6. Decode
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)


