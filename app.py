# KEYS, MODELS and ENV Related Settings

import os
from dotenv import load_dotenv
import time
import traceback, logging
from flask import Flask, request, jsonify
import requests


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

app = Flask(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
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
        {"role": "system", "content": "You are a virtual assistant. You will be given a prompt from the user. Your objective is to analyze the prompt to determine what kind of tone you should respond to the user with. you are to give a rating on a scale of 0.0-1.0... 0.0 being a more severe situation that needs a prompt response, and a 1.0 being a less serious situation and can answer the phrase in a lighthearted manner. I only want the rating and nothing else."},
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

    doc_retriever
    data = request.get_json()
    query = data['message']
    print("user message: ", query)
    adjustedTemp = chatbot_get_temp(query).content
    print("User query temperature: ", adjustedTemp)

    prompt_template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer with context:
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    llm = ChatOpenAI(temperature=adjustedTemp,
                 openai_api_key=OPENAI_API_KEY)

    chain_kwargs = {"prompt": PROMPT}
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


@app.route("/place-details", methods=['GET'])
def get_place_details():
    place_name = request.args.get('place_name')
    print("******place name: ", place_name)
    if not place_name:
        return jsonify({'error': 'Missing place_name parameter'})

    place_name = place_name + " London, Ontario, Canada"
    # googles autocomplete service (will get the most relevent result to the search)
    autocomplete_url = f"https://maps.googleapis.com/maps/api/place/autocomplete/json?input={place_name}&key={GOOGLE_PLACES_API_KEY}"
    autocomplete_response = requests.get(autocomplete_url)
    autocomplete_data = autocomplete_response.json()

    if autocomplete_data['status'] != 'OK' or len(autocomplete_data['predictions']) == 0:
        return jsonify({'error': 'No place found with the given name'})

    place_id = autocomplete_data['predictions'][0]['place_id']

    # get place details using the most relevant retrieved place id
    place_details_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&key={GOOGLE_PLACES_API_KEY}"
    place_details_response = requests.get(place_details_url)
    place_details_data = place_details_response.json()

    if place_details_data['status'] != 'OK':
        return jsonify({'error': 'Failed to fetch place details'})

    result = {
        'name': place_details_data['result']['name'],
        'address': place_details_data['result']['formatted_address'],
        'phone_number': place_details_data['result'].get('formatted_phone_number', 'N/A'),
        'website': place_details_data['result'].get('website', 'N/A'),
        'business_hours': place_details_data['result'].get('opening_hours', {}).get('weekday_text', []),
        'photos': [photo['photo_reference'] for photo in place_details_data['result'].get('photos', [])],
        'viewport': place_details_data['result'].get('geometry', {}).get('viewport', {}),
        'rating': place_details_data['result'].get('rating', 'N/A'),
        'price_level': place_details_data['result'].get('price_level', 'N/A'),
        'business_status': place_details_data['result'].get('business_status', 'N/A'),
        'types': place_details_data['result'].get('types', []),
        'menu': place_details_data['result'].get('menu', {}).get('url', 'N/A')
    }

    # if 'reviews' in place_details_data['result']:
    #     for review in place_details_data['result']['reviews']:
    #         result['reviews'].append({
    #             'author_name': review.get('author_name', 'Anonymous'),
    #             'rating': review.get('rating', 'N/A'),
    #             'text': review.get('text', 'No review text available')
    #         })

    return jsonify(result)










if __name__ == '__main__':
    chatbot_vector_store(myFilePath) # Uncomment this if you want to add pdfs to pinecone vector db
    # terminal_chatbot()
    app.run(debug=True)