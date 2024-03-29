from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from llama_index.core import VectorStoreIndex,Settings,SimpleDirectoryReader, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Local settings
from llama_index.core.node_parser import SentenceSplitter
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from chromadb.db.base import UniqueConstraintError
from chromadb.documents import Document, VectorType
# from chromadb import Document

# from flask_cors import CORS 
# CORS(app)
app = Flask(__name__)


chroma_client = chromadb.PersistentClient()

try:
    chroma_collection = chroma_client.create_collection("quickstart")
except UniqueConstraintError:
    # Handle case where collection already exists
    print("Collection there ******************************")
    chroma_collection = chroma_client.get_collection("quickstart")
    

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# Load documents
# documents = SimpleDirectoryReader("data").load_data()


# Load documents
documents = Document.bulk_from_directory("data")  # Assuming data directory contains your documents

# Insert documents into the vector database
for doc in documents:
    chroma_collection.insert(doc)


# #   huggingface-cli login
# >
# Initialize HuggingFace Embeddings
# embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# >
from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('./models/stsb-distilbert-base')  #  ./stsb-distilbert-base')



# Initialize LLama2 model
llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})


# Instantiate Settings
Settings.chunk_size = 120
Settings.llm = llm
Settings.embed_model = embed_model




print("*********************************************")  # This line is for just to find train is over

index = VectorStoreIndex(storage_context=storage_context)

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
    app.run(debug=True)
