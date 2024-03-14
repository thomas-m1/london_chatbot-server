# KEYS, MODELS and ENV Related Settings

import os
from dotenv import load_dotenv
import time
import traceback, logging
from flask import Flask, request, jsonify

from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory


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
    print("temp : ", assistant_reply.content)
    return assistant_reply


@app.route("/chatbot", methods=['POST'])
def chatbot():
    
    data = request.get_json()
    query = data['message']
    print("user message: ", query)
    adjustedTemp = chatbot_get_temp(query).content
    print("User query temperature: ", adjustedTemp)

    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    If the question is not related to the context or chat history, politely respond that you are tuned to only answer questions that are related to the context.
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    Question: {question}
    Answer with context:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "history", "question"]
    )
    llm = ChatOpenAI(temperature=adjustedTemp,
                 openai_api_key=OPENAI_API_KEY)

    chain_kwargs = {"verbose": True, 
                    "prompt": PROMPT,
                    "memory": ConversationBufferMemory(
                        memory_key="history",
                        input_key="question",
                    )}
    # Set up the RetrievalQA chain with the retriever
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=doc_retriever,
                                  return_source_documents=True,
                                  chain_type_kwargs=chain_kwargs)

    result = qa_chain({"query": query})
    source_docs = []
    for doc in result["source_documents"]:
        content = doc.page_content
        page_number = doc.metadata['page']
        source = doc.metadata['source']
        source_docs.append({
            'content': content,
            'doc' : f"Page {page_number}, Source: {source}"
        })
    # print(result["source_documents"])
    print ("\n**\n")
    print ("\n**\n")
    return jsonify({'reply': result["result"],
                    'source_docs': source_docs})


if __name__ == '__main__':
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
    client = PineconeClient(api_key=PINECONE_API_KEY)
    vector_db = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME)
    doc_retriever = vector_db.as_retriever(search_kwargs={'k': 2})
    vector_db
    app.run(debug=True)