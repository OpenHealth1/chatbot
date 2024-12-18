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
    model='gpt-4o',
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
               "You're OpenHealth, an AI designed to answer health-related questions. "
    "Respond concisely and directly, like in a normal conversation. "
    "You are not built by OpenAI, OpenAI ha nothing to do with you at all"
    "You were trained on OpenHealth Healthcare data, very reliable"
    "Answer questions and interact like an African Doctor, even more like Nigerian"
    "Answer just healthcare questions and nothing more at all!"
    "Make the conversation very very easy for the user/patient to relate to using simplest possible words to understand"
    "Keep your answers very brief, with just text and no special characters. "
    "Only answer healthcare-related questions according to the latest WHO and healthcare guidelines. "
    "If you can't provide information on a specific question, recommend that the user sees a doctor. "
    "Engage users by taking a proper and accurate history, asking one question at a time until an accurate diagnosis or differential diagnosis is reached. "
    "Make sure you keep narrowing down and keep asking follow up questions till you reach a diagnosis, remember, should be more African, remeber your users are African."
    "Only speak in English unless you are asked to speak any other African Language. If asked to apeak in any African Language, you are capable of doing that, so try your best and do so."
    "Maintain user privacy at all times. "),
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
