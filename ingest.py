import os
import time
from dotenv import load_dotenv
import traceback, logging

from flask import Flask, request, jsonify
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

from langchain.chat_models import ChatOpenAI
from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone as PineconeClient
from pinecone import ServerlessSpec


load_dotenv() 


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
embed_model = "text-embedding-ada-002"


def chatbot_vector_store(filePath):
    
    try:
        loader = DirectoryLoader(filePath, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        # Set up the RecursiveCharacterTextSplitter, then Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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
                PineconeVectorStore.from_documents(texts, embeddings, index_name=PINECONE_INDEX_NAME)
        
            print("Ingestion Complete.")
    except Exception as e:
        logging.error(traceback.format_exc())
        print("Ingestion Incomplete.")


if __name__ == '__main__':
    myFilePath = './docs'
    chatbot_vector_store(myFilePath)