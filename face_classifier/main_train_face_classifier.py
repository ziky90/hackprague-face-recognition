"""
MIT licence https://opensource.org/licenses/MIT

Copyright (c) 2017 Jan Zikes zikesjan@gmail.com
"""

import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.optimizers import Adam

NUM_CLASSES = 2
LEARNING_RATE = 0.0001
NET_INPUT_SHAPE = (224, 224, 3)


def prepare_dataset(dataset_path):
    pass


def custom_categorical_crossentropy_on_logits(y_true, y_pred):
    return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)


def construct_vgg16_model():
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


def perform_model_training(model, images, labels):
    model.compile(optimizer=Adam(lr=LEARNING_RATE),
                  loss=custom_categorical_crossentropy_on_logits)
    # TODO implement it generatior (in memory)
    model.fit(images, labels, batch_size=20)


def main():
    model = construct_vgg16_model()
    prepare_dataset()
    perform_model_training(model)


if __name__ == '__main__':
    main()
