"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

import argparse

import cv2
import matplotlib.pylab as plt
import numpy as np

from face_detector.face_detector_tools import locate_faces, crop_image


def parse_args():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser('Path to the input image')
    parser.add_argument('--input_image', type=str, required=True,
                        help='Path to input image')
    return parser.parse_args()


def plot_located_face(image, face_locations):
    """
    Plot all the located faces.

    :param image: Input image
    :type image: np.ndarray
    :param face_locations: Locations of all the faces in the format
                           (x_position, y_position, width, height)
    :type face_locations: (int, int, int, int)
    :return:
    :rtype:
    """
    for (x, y, w, h) in face_locations:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(image)
    plt.show()


def main():
    args = parse_args()
    input_image_path = args.input_image

    image = cv2.imread(input_image_path)
    face_locations = locate_faces(image)

    cropped_images = []
    for face in face_locations:
        cropped_images.append(crop_image(image, face))

    for crop in cropped_images:
        plt.imshow(np.asarray(crop))
        plt.show()

    # optional
    plot_located_face(image, face_locations)


if __name__ == '__main__':
    main()
