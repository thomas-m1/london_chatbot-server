# KEYS, MODELS and ENV Related Settings

import os
import time
import openai, langchain, pinecone

from flask import Flask
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Pinecone
# from langchain_community.vectorstores import Pinecone

from langchain_community.llms import OpenAI

from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeStore


os.environ["OPENAI_API_KEY"] = "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"
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

embed_model = "text-embedding-ada-002"

os.environ["PINECONE_API_KEY"] = "c31d352a-23fe-4db1-b66b-4acc33308fcf"
PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = "us-west4-gcp-free"


app = Flask(__name__)

# Open the data file and read its content

loader = DirectoryLoader('./docs', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()


# Set up the RecursiveCharacterTextSplitter, then Split the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
# print (type(texts))
# print (len(texts))
# print (texts[100])
# print (texts[100].metadata['source'])


pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "testingindex"
index = pc.Index(index_name)

if index_name not in pc.list_indexes().names():
    print("Index does not exist: ", index_name)
    pc.create_index(
        name=index_name,
        dimension=8,
        metric="euclidean",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
    # wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    # # docsearch = Pinecone.from_documents(texts, embeddings, index_name = index_name)
else:
    print("Index exists: ", index_name)
    # docsearch = Pinecone.from_existing_index(index_name, embeddings)
index_stats = index.describe_index_stats()
# print (index_stats)

vector_count = index_stats['total_vector_count']
vector_count_str = str(vector_count)

# index_description = pc.describe_index(index_name)


if vector_count >1:
    # Prepare the embedding so that we can pass it to the pinecone call in the next step
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # Create the vector store (new vector store)
    docsearch = PineconeStore.from_documents(texts, embeddings, index_name = index_name)
    # for existing an vector store
    # docsearch = PineconeStore.from_existing_index(index_name, embeddings)






from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA



# # set up the llm model to use with our chain/agent

llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)




# # Helper function to process the response from the QA chain
# # and isolate result and source docs and page numbers

def parse_response(response):
    print(response['result'])
    print('\n\nSources:')
    for source_name in response["source_documents"]:
        print(source_name.metadata['source'], "page #:", source_name.metadata['page'])



        # Set up the retriever on the pinecone vectorstore
# Make sure to set include_metadata = True

retriever = docsearch.as_retriever(include_metadata=True, metadata_key = 'source')



# Set up the RetrievalQA chain with the retriever
# Make sure to set return_source_documents = True

qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
# Let's set up the query

query = "List all the street names beginning with the letter a"
# Call the QA chain to get the response

response = qa_chain(query)
# print (response)

# parse_response(response)
print ("*******************************************************")
print (parse_response(response))
print ("*******************************************************")
# print (response["source_documents"])
















# # Import the dependencies

# from langchain.agents.agent_toolkits import (
#     create_vectorstore_agent,
#     VectorStoreToolkit,
#     VectorStoreInfo,
# )
# # Set up the vectorstore info

# vectorstore_info = VectorStoreInfo(
#     name="Generative AI Reports",
#     description="Reports on the State and Trends in Generative AI",
#     vectorstore= docsearch,
# )
# # Setup the VectorStoreToolkit and VectorStore Agent

# toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
# agent_executor = create_vectorstore_agent(llm=llm, toolkit=toolkit, verbose=False)
# # Add the string to ask for source

# query = query + " List the sources."
# print (query)
# # Run the agent

# response = agent_executor.run(query)
# response





if __name__ == '__main__':
    app.run(debug=True)