from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext,Settings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from flask_cors import CORS 
# CORS(app)
app = Flask(__name__)

# Load documents
documents = SimpleDirectoryReader("data").load_data()

# #   huggingface-cli login

# Initialize HuggingFace Embeddings
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
from ctransformers import AutoModelForCausalLM,AutoConfig



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
# Initialize VectorStoreIndex with LLama2 and HuggingFace Embeddings
index = VectorStoreIndex.from_documents(documents)

print("*********************************************")  # This line is for just to find train is over

print(index)

# Initialize Query Engine
query_engine = index.as_query_engine()

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
