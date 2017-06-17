"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

import cv2
from scipy.misc import imresize

CASCADE_FILE_PATH = "haarcascade_frontalface_default.xml"
DESIRED_IMAGE_SIZE = (224, 224, 3)


def crop_image(image, face_location):
    """
    Crop the image given the face location.

    :param image: Array representing the image.
    :type image: np.ndarray
    :param face_location: face bounding box given the (x, y, w, h)
    :type face_location: (int, int, int, int)
    :return: Cropped face from the image
    :rtype: np.ndarray
    """
    x, y, w, h = face_location
    croped_image = image[y:y + h, x:x + w, :]
    croped_image = imresize(croped_image, DESIRED_IMAGE_SIZE)
    return croped_image


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
