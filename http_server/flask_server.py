"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

import cv2
from flask import Flask, request, jsonify
from keras.models import load_model

from face_classifier.main_predict_tools import predict_image
from face_detector.face_detector_tools import locate_faces, crop_image

MODEL_PATH = 'model/model.h5'

model = load_model(MODEL_PATH)
app = Flask(__name__)


@app.route("/analyse_image", methods=['POST'])
def analyse_image():
    """
    Locate the face and classify it.
    """
    input_image_path = request.data

    image = cv2.imread(input_image_path)
    face_locations = locate_faces(image)

    results = {}
    for pos, face in enumerate(face_locations):
        cropped = crop_image(image, face)
        prediction = predict_image(cropped)
        results[pos] = (face, prediction)

    return jsonify(results)


@app.route("/detect_face", methods=['POST'])
def detect_face():
    """
    Detect all the possible faces with their locations.
    """
    input_image_path = request.data

    image = cv2.imread(input_image_path)
    face_locations = locate_faces(image)
    return jsonify(face_locations)


@app.route("/predict_face", methods=['POST'])
def predict_face():
    input_face_path = request.data
    image = cv2.imread(input_face_path)
    prediction = predict_image(image)
    return jsonify(prediction)
