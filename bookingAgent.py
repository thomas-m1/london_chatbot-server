from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from datetime import datetime, timedelta

app = Flask(__name__)
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

@app.route('/availability', methods=['GET'])
def availability():
    date = request.args.get('date')
    if not date:
        return jsonify({'error': 'Date parameter is required'}), 400

    available_slots = get_available_slots(date)
    return jsonify({'available_slots': available_slots})

# Route to book an appointment
@app.route('/book', methods=['POST'])
def book_appointment():
    data = request.get_json()
    user_id = data.get('user_id')
    date = data.get('date')
    start_time = data.get('start_time')

    if not user_id or not date or not start_time:
        return jsonify({'error': 'User ID, date, and start time are required'}), 400

    if datetime.strptime(date, '%Y-%m-%d').date() < datetime.now().date():
        return jsonify({'error': 'Cannot book appointments in the past'}), 400

    if start_time not in get_available_slots(date):
        return jsonify({'error': 'Selected slot is not available'}), 400

    end_time = (datetime.strptime(start_time, '%H:%M') + timedelta(minutes=30)).strftime('%H:%M')

    # Book the appointment by inserting into the database
    appointments_collection.insert_one({
        'date': date,
        'start_time': start_time,
        'end_time': end_time,
        'user_id': user_id,
        'status': 'booked'
    })

    return jsonify({'message': 'Appointment booked successfully'}), 200

if __name__ == '__main__':
    app.run(debug=True)
