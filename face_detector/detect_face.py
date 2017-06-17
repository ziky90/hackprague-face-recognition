"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

import argparse

import cv2
import matplotlib.pylab as plt
import numpy as np
from scipy.misc import imresize


CASCADE_FILE_PATH = "haarcascade_frontalface_default.xml"
DESIRED_IMAGE_SIZE = (224, 224, 3)


def parse_args():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser('Path to the input image')
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input image')
    return parser.parse_args()


def locate_faces(input_image):
    """
    Locate faces in the original image.

    :param input_image: Array representing the input image.
    :type input_image: np.ndarray
    :return: (x, y, w, h)
    :rtype: (int, int, int, int)
    """
    face_cascade = cv2.CascadeClassifier(CASCADE_FILE_PATH)
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    # detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    return faces


def plot_located_face(image, face_locations):
    """

    :param image:
    :type image:
    :param face_locations:
    :type face_locations:
    :return:
    :rtype:
    """
    for (x, y, w, h) in face_locations:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()


def crop_image(image, face_location):
    """

    :param image:
    :type image:
    :param face_location:
    :type face_location:
    :return:
    :rtype:
    """
    x, y, w, h = face_location
    croped_image = image[y:y + h, x:x + w, :]
    croped_image = imresize(croped_image, DESIRED_IMAGE_SIZE)
    return croped_image


def main():
    args = parse_args()
    input_image_path = args.input_image

    image = cv2.imread(input_image_path)
    face_locations = locate_faces(image)

    # TODO crop the image
    cropped_images = []
    for face in face_locations:
        cropped_images.append(crop_image(image, face))

    for crop in cropped_images:
        # Store the cropped image
        plt.imshow(np.asarray(crop))
        plt.show()

    # optional
    plot_located_face(image, face_locations)


if __name__ == '__main__':
    main()
