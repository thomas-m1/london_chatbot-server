# KEYS, MODELS and ENV Related Settings

import os
import time
import traceback, logging
# import openai, langchain, pinecone



from flask import Flask, request, jsonify
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_openai import OpenAIEmbeddings

from langchain_community.llms import OpenAI

# from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone as PineconeStore
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from config.pinecone import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV, PINECONE_INDEX_NAME, PINECONE_NAME_SPACE


app = Flask(__name__)

# Open the data file and read its content
def chatbot_vector_store(filePath):
    
    try:
        loader = DirectoryLoader(filePath, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        # Set up the RecursiveCharacterTextSplitter, then Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
    
        print("split docs:", texts)
        print("\n")
        print("creating a vector store...")
        pc = PineconeClient(api_key=PINECONE_API_KEY)
    
        # index_name = "testingindex"
        index = pc.Index(PINECONE_INDEX_NAME)

        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print("Index does not exist: ", PINECONE_INDEX_NAME)
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
            # wait for index to be initialized
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
            # # docsearch = Pinecone.from_documents(texts, embeddings, index_name = index_name)
        else:
            print("Index exists: ", PINECONE_INDEX_NAME)
            # docsearch = Pinecone.from_existing_index(index_name, embeddings)
        index_stats = index.describe_index_stats()
        print(index_stats)
        vector_count = index_stats['total_vector_count']
        print(vector_count)
        if vector_count >=0:
            # Prepare the embedding so that we can pass it to the pinecone call in the next step
            print("Hello world!")
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            # Create the vector store (new vector store)
            # pinecone_index = PineconeVectorStore.get_pinecone_index
            # print(pinecone_index)
            vector_db = PineconeVectorStore.from_texts(
                texts, 
                embedding=embeddings, 
                index_name=PINECONE_INDEX_NAME)
            print("Hello!!")
            global doc_retriever 
            doc_retriever = vector_db.as_retriever()
            
            # for existing an vector store
            # docsearch = PineconeStore.from_existing_index(index_name, embeddings)
        
        print("Ingestion Complete.")
    except Exception as e:
        logging.error(traceback.format_exc())
        print("Ingestion Incomplete.")

# # Helper function to process the response from the QA chain
# # and isolate result and source docs and page numbers
def parse_response(response):
    print(response['result'])
    print('\n\nSources:')
    for source_name in response["source_documents"]:
        print(source_name.metadata['source'], "page #:", source_name.metadata['page'])


def chatbot_get_temp(findTemp):
    client = OpenAI(api_key=OPENAI_API_KEY)
    qa_prompt = f"I want you to give me a rating on a scale of 0.0-1.0... 0.0 being a more severe situation that needs a prompt response, and a 1.0 being a less serious situation and can answer the phrase in a lighthearted manner. I only want the rating and nothing else. here is the phrase: {findTemp}"
    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "you are a virtual assistant"},
        {"role": "user", "content": qa_prompt},
    ]
    )
    
    # print(completion)
    assistant_reply = completion.choices[0].message
    print("temp : ", assistant_reply)
    return assistant_reply

# Set up the retriever on the pinecone vectorstore
# retriever = docsearch.as_retriever(include_metadata=True, metadata_key = 'source')

@app.route("/chatbot", methods=['POST'])

def chatbot():
    
    doc_retriever 
    data = request.get_json()
    query = data['message']
    print("user message: ", query)
    # userFindTemp = "hey I just fell and got a small cut on my leg after I tripped. it is not bad. what should I do?"
    adjustedTemp = chatbot_get_temp(query).content
    newTemp = float(adjustedTemp)

    llm = OpenAI(temperature=adjustedTemp, 
                 openai_api_key=OPENAI_API_KEY)

    # Set up the RetrievalQA chain with the retriever
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=doc_retriever,
                                  return_source_documents=True)

    response = qa_chain.run(query)
    # parse_response(response)
    print ("\n*******************************************************\n")
    print (parse_response(response))
    print ("\n*******************************************************\n")
    # print (chatbot_response_gpt(userFindTemp))
    return jsonify({'reply': response["result"],
                    'sourcedocs': response["source_documents"]})


if __name__ == '__main__':
    
    # define file path for pdfs
    myFilePath = './docs'
    chatbot_vector_store(myFilePath)
    app.run(debug=True)