"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

import base64

import cv2
from flask import Flask, request, jsonify, json
from flask_cors import CORS
from keras.models import load_model

from face_classifier.main_predict_tools import predict_image
from face_detector.face_detector_tools import locate_faces, crop_image
from http_server.people_data_db_mock import RECORDS_DB

MODEL_PATH = 'model_long/model.h5'

model = load_model(MODEL_PATH)
app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

@app.route('/')
def index():
    return 'test'


@app.route('/analyse_image', methods=['POST'])
def analyse_image():
    """
    Locate the face and classify it + add info from DB
    """
    input_image_path = json.loads(request.data)['image_path']

    image = cv2.imread(input_image_path)
    face_locations = locate_faces(image)

    results = {}
    for pos, face in enumerate(face_locations):
        cropped = crop_image(image, face)
        prediction = predict_image(model, cropped)
        results[pos] = {'bbox': [int(i) for i in face],
                        'class': prediction,
                        'metadata': RECORDS_DB[prediction]}

    return jsonify(results)


@app.route('/analyse_image_base64', methods=['POST'])
def analyse_image_base64():
    """
    Locate the face and classify it + add info from DB
    """
    data = request.data.replace('data:image/jpeg;base64', '')
    # decode the base64 image
    imgdata = base64.b64decode(data)
    filename = 'image_to_process.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)

    image = cv2.imread(filename)
    face_locations = locate_faces(image)

    results = {}
    for pos, face in enumerate(face_locations):
        cropped = crop_image(image, face)
        prediction = predict_image(model, cropped)
        results[pos] = {'bbox': [int(i) for i in face],
                        'class': prediction,
                        'metadata': RECORDS_DB[prediction]}

    return jsonify(results)


@app.route('/detect_face', methods=['POST'])
def detect_face():
    """
    Detect all the possible faces with their locations.
    """
    input_image_path = json.loads(request.data)['image_path']

    image = cv2.imread(input_image_path)
    face_locations = locate_faces(image)
    # TODO possibly store cropped face and return it's path
    results = {}
    for pos, face in enumerate(face_locations):
        results[pos] = {'bbox': [int(i) for i in face]}
    return jsonify(results)


@app.route('/detect_face_base64', methods=['POST'])
def detect_face_base64():
    """
    Detect all the possible faces with their locations.
    """
    data = request.data.replace('data:image/jpeg;base64', '')
    # decode the base64 image
    imgdata = base64.b64decode(data)
    filename = 'image_to_process.jpg'
    with open(filename, 'wb') as f:
        f.write(imgdata)

    image = cv2.imread(filename)
    face_locations = locate_faces(image)
    # TODO possibly store cropped face and return it's path
    results = {}
    for pos, face in enumerate(face_locations):
        results[pos] = {'bbox': [int(i) for i in face]}
    return jsonify(results)


@app.route('/predict_face', methods=['POST'])
def predict_face():
    """
    Predict the person given the cropped patch.
    """
    input_face_path = json.loads(request.data)['face_image']
    image = cv2.imread(input_face_path)
    prediction = predict_image(image)
    return jsonify(prediction)


@app.route('/get_person_data', methods=['POST'])
def get_person_data():
    """
    Get person data from the DB
    """
    person_id = json.load(request.data)['person_id']
    return jsonify(RECORDS_DB[person_id])


if __name__ == '__main__':
    app.run()
