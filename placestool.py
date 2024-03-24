from typing import Optional
import os

from flask import jsonify
import requests
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field, conlist
from requests import PreparedRequest


GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")


def prepare_and_log_request(base_url: str, params: Optional[dict] = None) -> PreparedRequest:
    """Prepare the request and log the full URL."""
    req = PreparedRequest()
    req.prepare_url(base_url, params)
    print(f'\033[92mCalling API: {req.url}\033[0m')
    return req


class Params(BaseModel):
    fields: Optional[conlist(str, min_items=1, max_items=5)] = Field(
        default=None,
        description='Fields to filter the output of the request.'
    )


class PathParams(BaseModel):
    name: str = Field(..., description='Name of the Point of Interest')


class RequestModel(BaseModel):
    params: Optional[Params] = None
    path_params: PathParams


@tool(args_schema=RequestModel)
def get_places_by_name(path_params: PathParams, params: Optional[Params] = None):
    """Useful for finding info about a certain place. Input should be a fully formed question."""
    place_name = path_params.name + " London, Ontario, Canada"
    autocomplete_base_url = f'https://maps.googleapis.com/maps/api/place/autocomplete/json?input={place_name}&key={GOOGLE_PLACES_API_KEY}'

    effective_params = {"fields": ",".join(params.fields)} if params and params.fields else None

    req = prepare_and_log_request(autocomplete_base_url, effective_params)

    # Make the request
    autocomplete_response = requests.get(req.url)
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


    response = {
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


    # Raise an exception if the request was unsuccessful
    response.raise_for_status()

    return response.json()