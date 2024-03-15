from flask import Flask, request, jsonify
from dotenv import load_dotenv
# from flask_cors import CORS 
import os
import logging
import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = Flask(__name__)
# CORS(app)
query_engine = ''

load_dotenv()

# @app.route('/start', methods=['GET'])
# def start():
import time
# Start time
start_time = time.time()

api_key = os.getenv("OPENAI_API_KEY")

from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
from llama_index.llms.huggingface import HuggingFaceLLM

documents = SimpleDirectoryReader("data").load_data()

# print(documents)


system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""

from llama_index.core import PromptTemplate
# input query format
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")





# !huggingface-cli login
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
model = "meta-llama/Llama-2-7b-chat-hf"
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=model,
    model_name=model,
    device_map="auto",
    model_kwargs={"torch_dtype": torch.float16 }  # "load_in_8bit":True

)


# !pip show llama_index
# from llama_index import ServiceContext
from langchain.text_splitter import CharacterTextSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

llm_model = OpenAI(model_name="gpt-3.5-turbo",api_key=api_key)  # "gpt-4" Next Use model_name instead of gpt3.5-turbo model



text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

settings = Settings
settings.chunk_size = 512
settings.llm = llm_model

settings.embed_model = embed_model

# Create Service Context using Settings
service_context = settings


index=VectorStoreIndex.from_documents(documents,service_context=service_context)
print(index)

# global query_engine

query_engine=index.as_query_engine()


print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
response=query_engine.query("what is the dec 25 in webkorps is it holiday or not if yes it is holiday what is the day")
print(type(response))
print(dir(response))   # ['data', 'response', 'user_query']

response_str = str(response.response)
print(type(response_str))
print(response_str)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


# End time
end_time = time.time()

# Calculate execution time
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")
    # return jsonify({"Update":"Successful"})



@app.route('/health', methods=['GET'])
def health_check():
    return 'OK', 200
        



@app.route('/query', methods=['POST'])
def query():
    try:
        # Get the query from the request body
        data = request.get_json()
        user_query = data.get('query')
        print(user_query)
        # Check if the query is provided
        if not user_query:
            return jsonify({"error": "Query is required"}), 400
        response = query_engine.query(user_query)
        response_str = str(response.response)

        #   How to get the response here in string format
        return jsonify({"response": response_str})
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred"}), 500


if __name__ == '__main__':
    app.run(debug=True)