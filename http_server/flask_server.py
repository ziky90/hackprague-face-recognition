"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

import base64
from io import BytesIO

import cv2
from PIL import Image
from flask import Flask, request, jsonify, json
from keras.models import load_model

from face_classifier.main_predict_tools import predict_image
from face_detector.face_detector_tools import locate_faces, crop_image
from http_server.people_data_db_mock import RECORDS_DB

MODEL_PATH = 'model_long/model.h5'

model = load_model(MODEL_PATH)
app = Flask(__name__)


@app.route('/')
def index():
    return 'test'


@app.route('/analyse_image', methods=['POST'])
def analyse_image():
    """
    Locate the face and classify it + add info from DB
    """
    input_image_path = json.load(request.data)['image_path']

    image = cv2.imread(input_image_path)
    face_locations = locate_faces(image)

    results = {}
    for pos, face in enumerate(face_locations):
        cropped = crop_image(image, face)
        prediction = predict_image(cropped)
        results[pos] = {'bbox': face,
                        'class': prediction,
                        'metadata': RECORDS_DB[prediction]}

    return jsonify(results)


@app.route('/analyse_image_base64', methods=['POST'])
def analyse_image_base64():
    """
    Locate the face and classify it + add info from DB
    """
    # decode the base64 image
    input_base64_data = json.load(request.data)['image']
    image = Image.open(BytesIO(base64.b64decode(input_base64_data)))

    face_locations = locate_faces(image)

    results = {}
    for pos, face in enumerate(face_locations):
        cropped = crop_image(image, face)
        prediction = predict_image(cropped)
        results[pos] = (face, prediction, RECORDS_DB[prediction])

    return jsonify(results)


@app.route('/detect_face', methods=['POST'])
def detect_face():
    """
    Detect all the possible faces with their locations.
    """
    input_image_path = json.load(request.data)['image_path']

    image = cv2.imread(input_image_path)
    face_locations = locate_faces(image)
    # TODO possibly store cropped face and return it's path
    return jsonify(face_locations)


@app.route('/detect_face_base64', methods=['POST'])
def detect_face_base64():
    """
    Detect all the possible faces with their locations.
    """
    # decode the base64 image
    input_base64_data = json.load(request.data)['image']
    image = Image.open(BytesIO(base64.b64decode(input_base64_data)))

    face_locations = locate_faces(image)
    # TODO possibly store cropped face and return it's path
    return jsonify(face_locations)


@app.route('/predict_face', methods=['POST'])
def predict_face():
    """
    Predict the person given the cropped patch.
    """
    input_face_path = json.load(request.data)['face_image']
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
