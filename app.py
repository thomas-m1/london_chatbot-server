# KEYS, MODELS and ENV Related Settings

import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify

from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone as PineconeClient
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory


from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai.chat_models import ChatOpenAI

from imagegen import countries_image_generator
# from countriesname import get_countries_by_name
from kbtool import knowledge_base
from placestool import get_places_by_name



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

    tools = [countries_image_generator, get_places_by_name, knowledge_base]

    functions = [convert_to_openai_function(f) for f in tools]
    model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=adjustedTemp).bind(functions=functions)

    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful assistant for the City of London. You have access to knowlege base tool which has information related to hospitals, events, businesses, places for London Ontario."
                                                "If the user has a general query not related to any of the topics give a generic answer based on your knowledege."
                                                "Use the tools to answer the user query with appropriate context."),
                                               MessagesPlaceholder(variable_name="chat_history"), ("user", "{input}"),
                                               MessagesPlaceholder(variable_name="agent_scratchpad")])

    memory = ConversationBufferWindowMemory(return_messages=True, memory_key="chat_history", k=5)

    chain = RunnablePassthrough.assign(agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
                                      ) | prompt | model | OpenAIFunctionsAgentOutputParser()

    agent_executor = AgentExecutor(agent=chain, tools=tools, memory=memory, verbose=True )

    result = agent_executor({'input': query})
    print(result)
    return jsonify({'reply': result['output']})
    # return result['output']

# while (prompt := input("Enter a query (q to quit): ")) != "q":
#     result = agent({'input': prompt})
#     print(result)
    
#     return jsonify({'reply': result["result"],
#                     'source_docs': source_docs})


if __name__ == '__main__':
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    # PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
    # client = PineconeClient(api_key=PINECONE_API_KEY)
    # embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # vector_db = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    # doc_retriever = vector_db.as_retriever(search_kwargs={'k': 3})
    app.run(debug=True)