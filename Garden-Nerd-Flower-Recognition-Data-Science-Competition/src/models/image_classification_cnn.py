from keras.layers import Input, Reshape, Flatten, Dropout, Dense, MaxPooling2D, Activation
from keras.models import Model
from keras import regularizers

from models.network import Network


class ImageClassificationCNN(Network):
    def __init__(self, num_classes, input_dim=(128, 128), activation='relu', dropout_rate=0.5):
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
        input_reshape = Reshape((self.input_dim[0], self.input_dim[1], 1))(self.input)
        x = Network.conv_block(input_reshape, 32, 5, activation=self.activation)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Network.conv_block(x, 64, 5, activation=self.activation)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Network.conv_block(x, 96, 3, activation=self.activation)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Network.conv_block(x, 128, 3, activation=self.activation)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Network.conv_block(x, 192, 3, activation=self.activation)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(64, kernel_regularizer=regularizers.l2(0.001))(x)
        x = Activation(self.activation)(x)

        x = Dropout(self.dropout_rate)(x)
        x = Dense(self.num_classes, kernel_regularizer=regularizers.l2(0.001))(x)
        x = Activation('softmax')(x)

        model = Model(self.input, x, name='okay')

        self.network = model
