from flask import Flask, request, jsonify, render_template
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()


app = Flask(__name__)
CORS(app)

# Set the environment variable for Flask configuration
os.environ["FLASK_DEBUG"] = "production"


# Initialize OpenAI
openai_api_key = os.getenv('OPENAI_API_KEY')

if openai_api_key:
    print("API Key loaded successfully.")
else:
    print("API Key not found.")


# Initialize Chat Model
chat_model = ChatOpenAI(
    openai_api_key=openai_api_key,
    model='gpt-4-turbo-preview',
    temperature=0.2
)


# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat')
def chat_view():
    return render_template('chat.html')

@app.route('/send_chat', methods=['POST'])
def send_chat():
    user_message = request.json.get('question')

    if not isinstance(user_message, str):
        return jsonify({'error': 'Message must be a string'}), 400
    
    print(user_message)

    prompt = ChatPromptTemplate(
        messages=[
            SystemMessagePromptTemplate.from_template(
                "You're  OpenHealth, an AI designed to answer health-related questions. \
Please respond concisely, directly addressing the question like in a normal conversation. \
make your answers very brief, just text no astericks or special characters.\
make your answers very brief, just text no astericks or special characters.\
If unsure about an answer, state clearly that you don't know. Keep your responses specific to the inquiry and maintain user privacy."
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{question}")
        ]
    )

    # Generate the response from the chat model
    conversation = LLMChain(
        llm=chat_model,
        prompt=prompt,
        verbose=True,
        memory=memory
    )


    # Generate the response from the chat model
    res = conversation({"question": user_message})
    actual_response = res['text']

    memory.chat_memory.add_user_message(user_message)
    memory.chat_memory.add_ai_message(actual_response)

    return jsonify({'response': actual_response}), 200

if __name__ == "__main__":
    app.run(debug=False)
