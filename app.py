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
from langchain.memory import ConversationBufferMemory


app = Flask(__name__)

# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
# GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")


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
    # print("temp : ", assistant_reply.content)
    return assistant_reply

@app.route("/chatbot", methods=['POST'])
def chatbot():

    data = request.get_json()
    query = data['message']
    print("user message: ", query)
    adjustedTemp = chatbot_get_temp(query).content
    print("User query temperature: ", adjustedTemp)

    prompt_template = """Use the following pieces of context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>)to answer the question at the end.
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
    ------



    """


    # INCLUDE EXAMPLES!!!!!
    ###############This kinda works??????????????
    # If the response includes a point of interest at the end of the answer, please write a list of the points of interest you suggested. the list should be delimited by "<POI></POI>"
    # #######################################

    #     If the response includes a point of interest, at the end of the response include the point of interest format the answer delimited by
    # "<abgfr>
    # POI:point of interest name 1
    # POI:point of interest name 2
    # POI:point of interest name 3
    # POI:point of interest name 4
    # etc...
    # </abgfr>"
    # in the order that they occur in the answer


    # If the response includes a point of interest, at the end of the response include the point of interest format the answer as an HTML table in the order that they occur in the answer


    # If the response includes a point of interest, at the end of the response include the point of interest formatted in HTML in the order that they occur in the answer like the following:
    # <POIUniqueIdentifier>
    # POI:point of interest name 1
    # POI:point of interest name 2
    # POI:point of interest name 3
    # POI:point of interest name 4
    # etc...
    # </POIUniqueIdentifier>



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
    print ("\n***************************************\n")
    print ("\n***************************************\n")


    # location = "London, Ontario, Canada"
    # def get_place_details(suggested_place, location):

    return jsonify({'reply': result["result"],
                    'source_docs': source_docs})


@app.route("/place-details", methods=['GET'])
def get_place_details():
    place_name = request.args.get('place_name')
    print("******place name: ", place_name)
    if not place_name:
        return jsonify({'error': 'Missing place_name parameter'})

    place_name = place_name + " London, Ontario, Canada"
    # googles autocomplete service (will get the most relevant result to the search)
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
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME')
    GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
    client = PineconeClient(api_key=PINECONE_API_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_db = PineconeVectorStore.from_existing_index(PINECONE_INDEX_NAME, embeddings)
    doc_retriever = vector_db.as_retriever(search_kwargs={'k': 3})
    app.run(debug=True)