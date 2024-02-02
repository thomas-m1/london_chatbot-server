# KEYS, MODELS and ENV Related Settings

import os
import time
import openai, langchain, pinecone
# from openai import OpenAI


from flask import Flask
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_community.llms import OpenAI

from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
from langchain.vectorstores import Pinecone as PineconeStore
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA

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



# # Helper function to process the response from the QA chain
# # and isolate result and source docs and page numbers
def parse_response(response):
    print(response['result'])
    print('\n\nSources:')
    for source_name in response["source_documents"]:
        print(source_name.metadata['source'], "page #:", source_name.metadata['page'])
# from openai import OpenAI


def chatbot_response_gpt(findTemp):
    from openai import OpenAI

    client = OpenAI(
        #   organization='YOUR_ORG_ID',
        api_key=os.environ.get(OPENAI_API_KEY),

    )

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "you are a virtual assistant"},
        {"role": "user", "content": f"I want you to give me a rating on a scale of 0.0-1.0... 0.0 being a more severe situation that needs a prompt response, and a 1.0 being a less serious situation and can answer the phrase in a lighthearted manner. I only want the rating and nothing else. here is the phrase: {findTemp}"}
    ]
    )




    print(completion.choices[0].message.content)

    assistant_reply = completion.choices[0].message

    # Extract the assistant's reply from the response
    return assistant_reply

# Set up the retriever on the pinecone vectorstore
retriever = docsearch.as_retriever(include_metadata=True, metadata_key = 'source')


findTemp = "hey I just fell and got a small cut on my leg after I tripped. it is not bad. what should I do?"

response_message = chatbot_response_gpt(findTemp)
adjustedTemp = response_message.content
newTemp = float(adjustedTemp)

llm = OpenAI(temperature=newTemp, openai_api_key=OPENAI_API_KEY)

# Set up the RetrievalQA chain with the retriever
qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)




query = "can you List all the street names beginning with the letter a"
# query1 = "whats 5 x 5?"


# parse_response(response)
print ("*******************************************************")
print (parse_response(qa_chain(query)))
print ("*******************************************************")
print (chatbot_response_gpt(findTemp))

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