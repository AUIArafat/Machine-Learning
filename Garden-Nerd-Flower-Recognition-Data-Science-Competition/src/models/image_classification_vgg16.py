from keras.applications import VGG16
from keras.layers import Input, Reshape, Flatten, Dropout, Dense, MaxPooling2D, Activation, Cropping2D, Conv2D, Lambda
from keras.models import Model
from keras import regularizers, Sequential
from keras import backend as K
from models.network import Network

IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
IMAGE_CHANNEL = 3

BOTTOM_CROP = 178

class ImageClassificationVGG16(Network):
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
        # loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
        vgg = VGG16(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL), weights='imagenet',
                    include_top=False)  # input_shape = (64,64,3) as required by VGG

        # this will exclude the initial layers from training phase as there are already been trained.
        for layer in vgg.layers:
            layer.trainable = False

        x = Flatten()(vgg.output)
        # x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
        x = Dense(self.num_classes, activation='softmax')(
            x)  # adding the output layer with softmax function as this is a multi label classification problem.

        model = Model(inputs=vgg.input, outputs=x)

        self.network = model
