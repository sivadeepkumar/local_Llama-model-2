from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

app = Flask(__name__)

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# Initialize HuggingFace Embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize LLama2 model
llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                    model_type='llama',
                    config={'max_new_tokens': 256,
                            'temperature': 0.01})

from llama_index.core import Settings

settings = Settings
settings.chunk_size = 512
settings.llm = llm

settings.embed_model = embed_model

# Create Service Context using Settings
service_context = settings

# Initialize VectorStoreIndex with LLama2 and HuggingFace Embeddings
index = VectorStoreIndex.from_documents(documents,service_context=service_context)

# Initialize Query Engine
query_engine = index.as_query_engine()

def getLLamaresponse(input_text):
    # Prompt Template
    template = """
        Provide me the response for the topic: {input_text}.
            """
    prompt = PromptTemplate(input_variables=["input_text"], template=template)
    response = llm(prompt(input_text))
    return response

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


