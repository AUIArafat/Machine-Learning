from keras.layers import Conv2D, BatchNormalization, Activation


class Network(object):
    def __init__(self):
        self.inputs = []
        self.network = None

    def setup(self):
        """
        setup the required network with the parameters
        """
        raise NotImplementedError("Must be subclassed.")

    def get(self):
        """
        :return: the keras network
        """
        if self.network is None:
            self.setup()
        print(self.network.summary())
        return self.network

    @staticmethod
    def conv_block(x, filters, kernel_size, strides=(1, 1), activation=None, padding='same'):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        return x
