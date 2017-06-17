"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

import argparse
import os
import glob

import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

NUM_CLASSES = 17
LEARNING_RATE = 0.0001
EPOCHS = 10
BATCH_SIZE = 10
NET_INPUT_SHAPE = (224, 224, 3)
AUGMENT_DATA = True


def parse_args():
    """
    Argument parser.
    """
    parser = argparse.ArgumentParser('Path to the input image')
    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to input images')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to the output folder')
    return parser.parse_args()


def prepare_dataset(dataset_path):
    """
    Build the Keras data generator from the input folder of the format:
    -dir
    ---image1
    ---image2
    -dir2
    ---image3
    ...
    :param dataset_path: Path to the dataset labeled by folder.
    :type dataset_path: str
    :return: Keras image data generator
    :rtype: :class:`keras.preprocessing.image.ImageDataGenerator`
    """
    persons = list(os.walk(dataset_path))[0][1]
    onehots = np.eye(len(persons))
    x_data = []
    y_data = []
    for pos, person in enumerate(persons):
        person_dir = os.path.join(dataset_path, person)
        for path in glob.glob(os.path.join(person_dir, '*.png')):
            x_data.append(np.asarray(Image.open(path)))
            y_data.append(onehots[pos])
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # since we have really small dataset, we'll augment a lot.
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zca_whitening=False)
    datagen.fit(x_data)
    return datagen, x_data, y_data


def custom_categorical_crossentropy_on_logits(y_true, y_pred):
    """
    Custom categoricakl cross entropy function, because of this issue:
    https://github.com/fchollet/keras/issues/6983

    :param y_true: Real one-hot represented label
    :type y_true: tf.Tensor
    :param y_pred: Predicted probabilities for the given class.
    :type y_pred: tf.Tensor
    :return: computed loss
    :rtype: float
    """
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


def construct_vgg16_model():
    """
    Build the classification network.
    """
    base_net = VGG16(weights='imagenet', include_top=False,
                     input_shape=NET_INPUT_SHAPE)
    # set all the base layers as not trainable in order to train the model much
    # faster
    for layer in base_net.layers:
        layer.trainable = False
    # build custom output dense layer
    net = base_net.output
    net = Flatten()(net)
    net = Dense(NUM_CLASSES)(net)
    return Model(inputs=base_net.input, outputs=net)


def perform_model_training(model, datagen, x_train, y_train):
    """
    Perform model training. (Since this is a hackathon, we'd like to overfit the
    model for particular data, we don't care with valid and train datasets for
    now)

    :param model: Constructed keras model.
    :type model: :class:`keras.models.Model`
    :param datagen: Dataset generator / augmenter
    :type datagen: :class:`keras.preprocessing.image.ImageDataGenerator`
    :param x_train: training data images
    :type x_train: np.ndarray
    :param y_train: trainig data labels
    :type y_train: np.ndarray
    :return: Trained keras model
    :rtype: :class:`keras.models.Model`
    """
    model.compile(optimizer=Adam(lr=LEARNING_RATE),
                  loss=custom_categorical_crossentropy_on_logits)
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
                        steps_per_epoch=len(x_train) / 32, epochs=EPOCHS)
    return model


def main():
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path

    model = construct_vgg16_model()
    datagen, x_data, y_data = prepare_dataset(input_path)
    model = perform_model_training(model, datagen, x_data, y_data)

    # store trained model
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model.save(os.path.join(output_path, 'model.h5'))


if __name__ == '__main__':
    main()
