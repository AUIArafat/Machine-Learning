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
        # input_reshape = Reshape((self.input_dim[0], self.input_dim[1], 1))(self.input)
        # x = Network.conv_block(input_reshape, 32, 5, activation=self.activation)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        #
        # x = Network.conv_block(x, 64, 5, activation=self.activation)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        #
        # x = Network.conv_block(x, 96, 3, activation=self.activation)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        #
        # x = Network.conv_block(x, 128, 3, activation=self.activation)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        #
        # x = Network.conv_block(x, 192, 3, activation=self.activation)
        # x = MaxPooling2D(pool_size=(2, 2))(x)
        #
        # x = Flatten()(x)
        # x = Dropout(0.5)(x)
        # x = Dense(64, kernel_regularizer=regularizers.l2(0.001))(x)
        # x = Activation(self.activation)(x)
        #
        # x = Dropout(self.dropout_rate)(x)
        # x = Dense(self.num_classes, kernel_regularizer=regularizers.l2(0.001))(x)
        # x = Activation('softmax')(x)
        #
        # model = Model(self.input, x, name='okay')
        model = Sequential()
        model.add(Cropping2D(cropping=((0, BOTTOM_CROP), (0, 0)),
                             input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL)))
        model.add(Lambda(lambda x: x / 127.5 - 1.))
        model.add(Conv2D(32, 8, strides=(4, 4), padding="same", activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(32, 4, strides=(2, 2), padding="same", activation='relu'))
        model.add(MaxPooling2D(2, 2))
        #     model.add(Conv2D(64, 5, strides=(2, 2), padding="same", activation='relu'))
        model.add(Flatten())
        model.add(Dropout(.35))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(.5))
        model.add(Dense(self.num_classes))
        # model.add(Lambda(lambda x: (K.exp(x) + 1e-4) / (K.sum(K.exp(x)) + 1e-4)))
        #model.add(Lambda(lambda x: K.tf.nn.softmax(x)))

        self.network = model
