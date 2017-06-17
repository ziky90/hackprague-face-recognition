"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

import argparse
import glob
import os
import logging

import cv2
from PIL import Image

from face_detector.face_detector_tools import locate_faces, crop_image


def parse_args():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser('Path to the input image')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input images folder')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path where to store images')
    return parser.parse_args()


def process_images(input_path):
    """
    Process all the images and return face patches.

    :param input_path: input path to the image
    :type input_path: str
    :return: List of cropped images with faces.
    :rtype: [np.ndarray]
    """
    logging.info('Processing faces')
    cropped_images = []
    # TODO iterate over all the .jpg files
    for path in glob.glob(os.path.join(input_path, '*.jpg')):
        image = cv2.imread(path)
        face_locations = locate_faces(image)

        for face in face_locations:
            cropped_images.append(crop_image(image, face))

    return cropped_images


def main():
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path

    cropped_images = process_images(input_path)

    for pos, crop in enumerate(cropped_images):
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        image = Image.fromarray(crop)
        image.save(os.path.join(output_path, 'img' + str(pos) + '.png'))


if __name__ == '__main__':
    main()
