# from flask import Flask, request, jsonify
# from langchain.prompts import PromptTemplate
# from langchain.llms import CTransformers
# # from flask_cors import CORS 
# # CORS(app)
# app = Flask(__name__)

# from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,ServiceContext
# from llama_index.llms.huggingface import HuggingFaceLLM
# documents = SimpleDirectoryReader("data").load_data()


# # #   huggingface-cli login
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# embed_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# from llama_index.core import Settings


# settings = Settings
# settings.chunk_size = 512
# settings.llm = llm

# settings.embed_model = embed_model

# # Create Service Context using Settings
# service_context = settings

# index=VectorStoreIndex.from_documents(documents,service_context=service_context)
# print(index)

# print("*********************************************")  # This line is for just to find train is over
# query_engine=index.as_query_engine()




# def getLLamaresponse(input_text):
#     # LLama2 model
#     llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
#                         model_type='llama',
#                         config={'max_new_tokens': 256,
#                                 'temperature': 0.01})
    
#     # Prompt Template
#     template = """
#         Provide me the response for the topic: {input_text}.
#             """
    
#     prompt = PromptTemplate(input_variables=["input_text"],
#                             template=template)
    
    
#     return response

# @app.route('/generate-response', methods=['POST'])
# def generate_response():
#     data = request.get_json()
#     input_text = data.get('input_text')

#     if input_text is None:
#         return jsonify({'error': 'Missing required parameter: input_text'}), 400

#     response = getLLamaresponse(input_text)
#     return jsonify({'response': response})
# if __name__ == '__main__':
#     app.run(debug=True)






# def getLLamaresponse(input_text, no_words, blog_style):
#     # LLama2 model
#     llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
#                         model_type='llama',
#                         config={'max_new_tokens': 256,
#                                 'temperature': 0.01})
    
#     # Prompt Template
#     template = """
#         Provide me the response with {blog_style} job profile for a topic {input_text}
#         within {no_words} words.
#             """
    
#     prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
#                             template=template)
    
#     # Generate the response from the LLama 2 model
#     response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
#     return response

# @app.route('/generate-response', methods=['POST'])
# def generate_response():
#     data = request.get_json()
#     input_text = data.get('input_text')
#     no_words = data.get('no_words')
#     blog_style = data.get('blog_style')

#     if input_text is None or no_words is None or blog_style is None:
#         return jsonify({'error': 'Missing required parameters'}), 400

#     response = getLLamaresponse(input_text, no_words, blog_style)
#     return jsonify({'response': response})

# if __name__ == '__main__':
#     app.run(debug=True)


