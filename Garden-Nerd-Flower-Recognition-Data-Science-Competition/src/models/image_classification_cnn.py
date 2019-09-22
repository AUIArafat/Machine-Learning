from keras.layers import Input, Reshape, Flatten, Dropout, Dense, MaxPooling2D, Activation, Cropping2D, Conv2D, Lambda
from keras.models import Model
from keras import regularizers, Sequential
from keras import backend as K
from models.network import Network

IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
IMAGE_CHANNEL = 3

BOTTOM_CROP = 178

class ImageClassificationCNN(Network):
    def __init__(self, num_classes, input_dim=(128, 500, 500), activation='relu', dropout_rate=0.5):
        """
        :param num_classes: number of classes
        :param input_dim: input dimension
        :param dropout_rate: fraction of units to drop
        """
        super().__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.input = Input(shape=input_dim)

    def setup(self):

        # model = Sequential()
        # model.add(Cropping2D(cropping=((0, BOTTOM_CROP), (0, 0)),
        #                      input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)))
        # model.add(Lambda(lambda x: x / 127.5 - 1.))
        # model.add(Conv2D(32, 8, strides=(4, 4), padding="same", activation='relu'))
        # model.add(MaxPooling2D(2, 2))
        # model.add(Conv2D(32, 4, strides=(2, 2), padding="same", activation='relu'))
        # model.add(MaxPooling2D(2, 2))
        # #     model.add(Conv2D(64, 5, strides=(2, 2), padding="same", activation='relu'))
        # model.add(Flatten())
        # model.add(Dropout(.35))
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(.5))
        # model.add(Dense(self.num_classes))

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # the model so far outputs 3D feature maps (height, width, features)

        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        self.network = model
