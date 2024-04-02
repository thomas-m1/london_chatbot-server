from typing import Optional
import os

from flask import jsonify
import requests
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field, conlist
from requests import PreparedRequest
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
MONGODB_PASSWORD = os.getenv('MONGODB_PASSWORD')
MONGODB_USERNAME = os.getenv('MONGODB_USERNAME')

uri = f"mongodb+srv://{MONGODB_USERNAME}:{MONGODB_PASSWORD}@4470.fe5k7eb.mongodb.net/?retryWrites=true&w=majority&appName=4470"
client = MongoClient(uri)
db = client['4470']
appointments_collection = db['appointments']

def get_booked_slots(date):
    booked_slots = []
    appointments = appointments_collection.find({'date': date})
    for appointment in appointments:
        start_time = datetime.strptime(appointment['start_time'], '%H:%M')
        end_time = datetime.strptime(appointment['end_time'], '%H:%M')
        while start_time < end_time:
            booked_slots.append(start_time.strftime('%H:%M'))
            start_time += timedelta(minutes=30)
    return booked_slots


def get_available_slots(date):
    today = datetime.strptime(date, '%Y-%m-%d').date()
    start_datetime = datetime.combine(today, datetime.min.time()) + timedelta(hours=9)
    end_datetime = datetime.combine(today, datetime.min.time()) + timedelta(hours=17)
    booked_slots = get_booked_slots(date)
    available_slots = []
    while start_datetime < end_datetime:
        if start_datetime.weekday() < 5 and start_datetime.strftime('%H:%M') not in booked_slots:
            available_slots.append(start_datetime.strftime('%H:%M'))
        start_datetime += timedelta(minutes=30)
    return available_slots

# def prepare_and_log_request(base_url: str, params: Optional[dict] = None) -> PreparedRequest:
#     """Prepare the request and log the full URL."""
#     req = PreparedRequest()
#     req.prepare_url(base_url, params)
#     print(f'\033[92mCalling API: {req.url}\033[0m')
#     return req


class Params(BaseModel):
    fields: Optional[conlist(str, min_items=1, max_items=2)] = Field(
        default=None,
        description='Fields to filter the output of the request.'
    )


class PathParams(BaseModel):
    date: str = Field(..., description='Date of the appointment. Should be in format of yyyy-mm-dd. If no date is given, do not book the slot, pass in "00", and ask the user to input day and time')
    start_time: str = Field(..., description='Start time of the appointment. Should be in format of HH:MM. If no time is given, pass in "00", and ask the user to input day and time')

class RequestModel(BaseModel):
    params: Optional[Params] = None
    path_params: PathParams


@tool(args_schema=RequestModel)
def book_appointment(path_params: PathParams, params: Optional[Params] = None):
    """for booking clinic appointments"""
    date = path_params.date
    start_time = path_params.start_time
    user_id = "test"
    print("___________date/start_time___________________________>>>>____________")
    print(date)
    print(start_time)

    if not user_id or not date or not start_time:
        return jsonify({'error': 'User ID, date, and start time are required'}), 400

    if datetime.strptime(date, '%Y-%m-%d').date() < datetime.now().date():
        return jsonify({'error': 'Cannot book appointments in the past'}), 400

    if start_time not in get_available_slots(date):
        return jsonify({'error': 'Selected slot is not available'}), 400

    end_time = (datetime.strptime(start_time, '%H:%M') + timedelta(minutes=30)).strftime('%H:%M')

    appointment_data = {
        'date': date,
        'start_time': start_time,
        'end_time': end_time,
        'user_id': user_id,
        'status': 'booked'
    }
    print(appointment_data)

    inserted_doc = appointments_collection.insert_one(appointment_data)

    print("_____>Appointment booked successfully")

    # Fetch the inserted document from the database
    inserted_document = appointments_collection.find_one({'_id': inserted_doc.inserted_id})
    inserted_document['_id'] = str(inserted_document['_id'])  # Convert ObjectId to string

    response_json = {
        'message': 'Appointment booked successfully',
        'appointment': inserted_document
    }

    return jsonify(response_json)


def availability():
    date = request.args.get('date')
    if not date:
        return jsonify({'error': 'Date parameter is required'}), 400

    available_slots = get_available_slots(date)
    return jsonify({'available_slots': available_slots})