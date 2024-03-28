# KEYS, MODELS and ENV Related Settings

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone as PineconeClient

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai.chat_models import ChatOpenAI

from imagegen import countries_image_generator
from kbtool import knowledge_base
from placestool import get_places_by_name
from bookingTool import book_appointment


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

    # print(completion)
    assistant_reply = completion.choices[0].message
    # print("temp : ", assistant_reply.content)
    return assistant_reply


@app.route("/chatbot", methods=['POST'])
def chatbot():

    data = request.get_json()
    query = data['message']
    print("user message: ", query)
    adjustedTemp = chatbot_get_temp(query).content
    print("User query temperature: ", adjustedTemp)

    # creating an agent for testing

    tools = [countries_image_generator, get_places_by_name, knowledge_base, book_appointment]

    functions = [convert_to_openai_function(f) for f in tools]
    model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=adjustedTemp).bind(functions=functions)

    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant for the City of London. You have access to knowledge base tool which has information related to hospitals, events, businesses, places for London Ontario."
                                                "If the user has a general query not related to any of the topics give a generic answer based on your knowledge."
                                                "if the users query is about any places or events, use the get_places_by_name tool"
                                                "If the users query asks to book an appointment with a clinic, use the book_appointment tool. The prompt must include both a date and a time. If the user does not provide that information, ask them for it."
                                                "Use the tools to answer the user query with appropriate context."),
                                               MessagesPlaceholder(variable_name="chat_history"), ("user", "{input}"),
                                               MessagesPlaceholder(variable_name="agent_scratchpad")])

    memory = ConversationBufferWindowMemory(return_messages=True, memory_key="chat_history", k=5)

    chain = RunnablePassthrough.assign(agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
                                      ) | prompt | model | OpenAIFunctionsAgentOutputParser()

    agent_executor = AgentExecutor(agent=chain, tools=tools, memory=memory, verbose=True )

    result = agent_executor({'input': query})
    print("app.py____________>"+str(result))
    return jsonify({'reply': result['output']})


if __name__ == '__main__':
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    app.run(debug=True)