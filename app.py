# KEYS, MODELS and ENV Related Settings

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from openai import OpenAI
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai.chat_models import ChatOpenAI
from flask_pymongo import PyMongo
from pymongo.mongo_client import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash


from imagegen import countries_image_generator
from kbtool import knowledge_base
from placestool import get_places_by_name
from bookingTool import book_appointment
from availibilityTool import check_availability
import json


app = Flask(__name__)


def chatbot_get_temp(findTemp):
    client = OpenAI(api_key=OPENAI_API_KEY)
    qa_prompt = f"I want you to give me a rating on a scale of 0.0-1.0... 0.0 being a more severe situation that needs a prompt response, and a 1.0 being a less serious situation and can answer the phrase in a lighthearted manner. I only want the rating and nothing else. here is the phrase: {findTemp}"
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a virtual assistant. You will be given a prompt from the user. Your objective is to analyze the prompt to determine what kind of tone you should respond to the user with. you are to give a rating on a scale of 0.0-1.0... 0.0 being a more severe situation that needs a prompt response, and a 1.0 being a less serious situation and can answer the phrase in a lighthearted manner. I only want the rating and nothing else."},
        {"role": "user", "content": qa_prompt},
    ]
    )
    
    assistant_reply = completion.choices[0].message
    return assistant_reply

def create_agent():
    tools = [countries_image_generator, get_places_by_name, knowledge_base, book_appointment, check_availability]

    functions = [convert_to_openai_function(f) for f in tools]
    model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0).bind(functions=functions)

    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant for the City of London. You have access to knowledge base tool which has information related to hospitals, events, businesses, places for London Ontario."
                                                "If the user has a general query not related to any of the topics give a generic answer based on your knowledge."
                                                "If the user query is about any places or events, use the get_places_by_name tool"
                                                "If the user query asks to book an appointment with a clinic, check if the date and time is included in the query, if not, ask the user for the date and time and then use the book_appointment tool. "
                                                "Use the tools to answer the user query with appropriate context."
                                                ),
                                               MessagesPlaceholder(variable_name="chat_history"), ("user", "{input}"),
                                               MessagesPlaceholder(variable_name="agent_scratchpad")])

    memory = ConversationBufferWindowMemory(return_messages=True, memory_key="chat_history", k=5)

    chain = RunnablePassthrough.assign(agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
                                      ) | prompt | model | OpenAIFunctionsAgentOutputParser() 

    agent_executor = AgentExecutor(agent=chain, tools=tools, memory=memory, verbose=True )

    return agent_executor


my_agent = create_agent()

@app.route("/chatbot", methods=['POST'])
def chatbot():

    data = request.get_json()
    query = data['message']
    print("user message: ", query)
    adjustedTemp = chatbot_get_temp(query).content
    print("User query temperature: ", adjustedTemp)

    result = my_agent({'input': query})
    print("app.py____________>"+str(result))
    
    filename = "stored_response.json"  # Replace with the actual filename
    stored_response = {}
    try:
        with open(filename, 'r') as file:
            stored_response = json.load(file)
    except FileNotFoundError:
        pass  # File doesn't exist, ignore and proceed

    # Clear the file
    with open(filename, 'w') as file:
        file.truncate(0)
        # Return both the original response and the stored response
    combined_response = {'reply': result['output'], 'stored_response': stored_response}
    return jsonify(combined_response)
    # return jsonify({'reply': result['output']})


@app.route('/register', methods=['POST'])
def register():
    # Get user data from the posted JSON
    email = request.json.get('email')
    password = request.json.get('password')
    
    
    if mongo.db.users.find_one({'email': email}):
        return jsonify(message="User already exists."), 409
    
    # Hash the user's password
    hashed_password = generate_password_hash(password)
    
    # Store the user in MongoDB
    mongo.db.users.insert_one({
        'email': email,
        'password': hashed_password
    })
        
    return jsonify({'message': 'Registered successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    # Get user credentials from the posted JSON
    email = request.json.get('email')
    password = request.json.get('password')
    
    # Find user in the database
    user = mongo.db.users.find_one({'email': email})
    
    # Check the password
    if user and check_password_hash(user['password'], password):
        # Logic for successful login (e.g., generating a token)
        return jsonify({'message': 'Login successful'}), 200
    else:
        # Login failed
        return jsonify({'message': 'Invalid credentials'}), 401


if __name__ == '__main__':
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    MONGODB_PASSWORD = os.getenv('MONGODB_PASSWORD')
    MONGODB_USERNAME = os.getenv('MONGODB_USERNAME')
    app.config["MONGO_URI"] = "mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@4470.fe5k7eb.mongodb.net/?retryWrites=true&w=majority&appName=4470"
    mongo = PyMongo(app)

    # mongo_client = MongoClient(uri)
    # db = mongo_client['4470']
    # appointments_collection = db['appointments']
    
    app.run(debug=True)