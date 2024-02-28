from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

app = Flask(__name__)

def getLLamaresponse(input_text, no_words, blog_style):
    # LLama2 model
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type='llama',
                        config={'max_new_tokens': 256,
                                'temperature': 0.01})
    
    # Prompt Template
    template = """
        Provide me the response with {blog_style} job profile for a topic {input_text}
        within {no_words} words.
            """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)
    
    # Generate the response from the LLama 2 model
    response = llm(prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words))
    return response

@app.route('/generate-response', methods=['POST'])
def generate_response():
    data = request.get_json()
    input_text = data.get('input_text')
    no_words = data.get('no_words')
    blog_style = data.get('blog_style')

    if input_text is None or no_words is None or blog_style is None:
        return jsonify({'error': 'Missing required parameters'}), 400

    response = getLLamaresponse(input_text, no_words, blog_style)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)