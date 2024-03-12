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

app = Flask(__name__)

load_dotenv()  # take environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
embed_model = "text-embedding-ada-002"
myFilePath = './docs'



# Open the data file and read its content
def chatbot_vector_store(filePath):

    try:
        loader = DirectoryLoader(filePath, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        # Set up the RecursiveCharacterTextSplitter, then Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)

        print("\n")
        print("creating a vector store...")
        client = PineconeClient(api_key=PINECONE_API_KEY)

        my_index = client.Index(PINECONE_INDEX_NAME)

        if PINECONE_INDEX_NAME not in client.list_indexes().names():
            print("Index does not exist: ", PINECONE_INDEX_NAME)
            client.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )
            # wait for index to be initialized
            while not client.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
        else:
            print("Index exists: ", PINECONE_INDEX_NAME)
            print("Before Vector Store")
            index_stats = my_index.describe_index_stats()
            print(index_stats)
            vector_count = index_stats['total_vector_count']
            print(vector_count)
            if vector_count >=0:
                # Prepare the embedding so that we can pass it to the pinecone call in the next step
                embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
                # Create the vector store (new vector store)
                vector_db = PineconeVectorStore.from_documents(texts, embeddings, index_name=PINECONE_INDEX_NAME)

                global doc_retriever
                doc_retriever = vector_db.as_retriever()

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
    print("temp : ", assistant_reply.content)
    return assistant_reply

# Set up the retriever on the pinecone vectorstore
# retriever = docsearch.as_retriever(include_metadata=True, metadata_key = 'source')

@app.route("/chatbot", methods=['POST'])
def chatbot():

    # doc_retriever
    data = request.get_json()
    query = data['message']
    print("user message: ", query)
    adjustedTemp = chatbot_get_temp(query).content
    adjustedTempFloat = float(adjustedTemp)

    llm = ChatOpenAI(temperature=adjustedTempFloat,
                 openai_api_key=OPENAI_API_KEY)

    # Set up the RetrievalQA chain with the retriever
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=doc_retriever,
                                  return_source_documents=True)

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
    print(result["source_documents"])
    print ("\n*******************************************************\n")
    print ("\n*******************************************************\n")
    return jsonify({'reply': result["result"],
                    'source_docs': source_docs})


if __name__ == '__main__':
    chatbot_vector_store(myFilePath) # Uncomment this if you want to add pdfs to pinecone vector db
    # terminal_chatbot()
    app.run(debug=True)