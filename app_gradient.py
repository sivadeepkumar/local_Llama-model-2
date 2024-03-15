from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext,Settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Local settings
from llama_index.core.node_parser import SentenceSplitter
import os
from dotenv import load_dotenv
load_dotenv()
# from flask_cors import CORS 
# CORS(app)


app = Flask(__name__)

GRADIENT_ACCESS_TOKEN = os.getenv("GRADIENT_ACCESS_TOKEN")
GRADIENT_WORKSPACE_ID = os.getenv("GRADIENT_WORKSPACE_ID")


# Load documents
documents = SimpleDirectoryReader("data").load_data()


from llama_index.embeddings.gradient import GradientEmbedding

embed_model = GradientEmbedding(
    gradient_access_token=GRADIENT_ACCESS_TOKEN,
    gradient_workspace_id=GRADIENT_WORKSPACE_ID,
    gradient_model_slug="bge-large",
)


# Initialize LLama2 model
llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})


# Instantiate Settings
Settings.chunk_size = 100
Settings.llm = llm
Settings.embed_model = embed_model




print("*********************************************")  # This line is for just to find train is over
# Initialize VectorStoreIndex with LLama2 and HuggingFace Embeddings
# index = VectorStoreIndex.from_documents(documents)
index = VectorStoreIndex.from_documents(documents=documents   # storage_context=storage_context,
)
print("*********************************************")  # This line is for just to find train is over

print(index)

# Initialize Query Engine
query_engine = index.as_query_engine(similarity_top_k=5)

def getLLamaresponse(input_text):
    # Prompt Template
    template = """
        Provide me the response for the topic: {input_text}.
            """
    prompt = PromptTemplate(input_variables=["input_text"], template=template)
    # response = llm.generate(prompt(input_text))
    response = query_engine.query(template)
    print(type(response))
    print(dir(response))   # ['data', 'response', 'user_query']

    response_str = str(response.response)
    print(type(response_str))
    print(response_str)
    return response_str

@app.route('/generate-response', methods=['POST'])
def generate_response():
    data = request.get_json()
    input_text = data.get('input_text')

    if input_text is None:
        return jsonify({'error': 'Missing required parameter: input_text'}), 400

    response = getLLamaresponse(input_text)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=8000)


















