from typing import Optional
import os

from flask import jsonify
from langchain.tools import tool
from pydantic.v1 import BaseModel, Field, conlist
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from flask import jsonify
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

class Params(BaseModel):
    fields: Optional[conlist(str, min_items=0, max_items=2)] = Field(
        default=None,
        description='Respond with all of the available times for a given date.'
    )


class PathParams(BaseModel):
    date: str = Field(..., description='Date in YYYY-MM-DD format')

class RequestModel(BaseModel):
    params: Optional[Params] = None
    path_params: PathParams


@tool(args_schema=RequestModel)
def check_availability(path_params: PathParams, params: Optional[Params] = None):
    """return all the available times for a given day"""
    print ("____________________check_availability")
    date = path_params.date
    if not date:
        return jsonify({'error': 'Date parameter is required'}), 400

    available_slots = get_available_slots(date)
    print(available_slots)
    return jsonify({'available_slots': available_slots})