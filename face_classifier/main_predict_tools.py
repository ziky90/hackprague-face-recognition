"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

import argparse

import numpy as np
from PIL import Image
from keras.models import load_model

MEAN = np.array([90.39201355, 99.05830383, 126.28666687])
STD = np.array([60.41706848, 61.64616013, 64.53026581])


def parse_args():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser('Path to the input image')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image')
    return parser.parse_args()


def predict_image_path(model, image_path):
    """
    Predict image.

    :param model:
    :type model:
    :param image_path:
    :type image_path:
    :return:
    :rtype:
    """
    input_image = np.expand_dims(np.asarray(Image.open(image_path)), axis=0)
    # input_image = (input_image - MEAN) / STD
    prediction = model.predict(input_image)
    return np.argmax(prediction)


def predict_image(model, image):
    input_image = np.expand_dims(image, axis=0)
    prediction = model.predict(input_image)
    return np.argmax(prediction)


def main():
    args = parse_args()
    model_path = args.model_path
    image_path = args.image_path

    model = load_model(model_path)

    prediction = predict_image_path(model, image_path)
    print prediction


if __name__ == '__main__':
    main()
