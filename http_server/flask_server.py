"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

import cv2
from flask import Flask
from keras.models import load_model

from face_classifier.main_predict_tools import predict_image
from face_detector.face_detector_tools import locate_faces, crop_image

MODEL_PATH = 'model/model.h5'

model = load_model(MODEL_PATH)
app = Flask(__name__)


@app.route("/detect_faces")
def detect_faces():
    # TODO load the image

    image = cv2.imread(input_image_path)
    face_locations = locate_faces(image)

    results = {}
    for pos, face in enumerate(face_locations):
        cropped = crop_image(image, face)
        prediction = predict_image(cropped)
        results[pos] = (face, prediction)

    return results
