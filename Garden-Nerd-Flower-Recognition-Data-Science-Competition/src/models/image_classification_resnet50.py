from keras.layers import Input, Reshape, Flatten, Dropout, Dense, MaxPooling2D, Activation, Cropping2D, Conv2D, Lambda, \
    BatchNormalization, concatenate, UpSampling2D, SpatialDropout2D
from keras.models import Model
from keras import regularizers, Sequential
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from models.network import Network

IMAGE_HEIGHT = 500
IMAGE_WIDTH = 500
IMAGE_CHANNEL = 3
BOTTOM_CROP = 178
WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


class ImageClassificationResNet50(Network):
    def __init__(self, num_classes, input_dim=(128, 128, 3), activation='relu', dropout_rate=0.5,
            include_top=False, weights='imagenet',
            input_tensor=None, input_shape=None,
            pooling=None):
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
        self.include_top = include_top
        self.weights = weights
        self.input_tensor = input_tensor
        self.input_shape = input_shape
        self.pooling = pooling,

    def ResNet50(self):

        if self.weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `imagenet` '
                             '(pre-training on ImageNet).')

        if self.weights == 'imagenet' and self.include_top and self.num_classes != 102:
            raise ValueError('If using `weights` as imagenet with `include_top`'
                             ' as true, `classes` should be 102')

        if self.input_tensor is None:
            img_input = Input(shape=self.input_dim)
        else:
            if not K.is_keras_tensor(self.input_tensor):
                img_input = Input(tensor=self.input_tensor, shape=self.input_dim)
            else:
                img_input = self.input_tensor
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1

        x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
        x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

        x = Network.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = Network.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = Network.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = Network.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = Network.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = Network.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = Network.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = Network.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = Network.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = Network.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = Network.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = Network.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = Network.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = Network.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = Network.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = Network.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        #     x = AveragePooling2D((7, 7), name='avg_pool')(x)

        #     if include_top:
        #         x = Flatten()(x)
        #         x = Dense(classes, activation='softmax', name='fc1000')(x)
        #     else:
        #         if pooling == 'avg':
        #             x = GlobalAveragePooling2D()(x)
        #         elif pooling == 'max':
        #             x = GlobalMaxPooling2D()(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if self.input_tensor is not None:
            inputs = get_source_inputs(self.input_tensor)
        else:
            inputs = img_input
        # Create model.
        model = Model(inputs, x, name='resnet50')

        # load weights
        if self.weights == 'imagenet':
            if self.include_top:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH,
                                        cache_subdir='models',
                                        md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
            else:
                weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models',
                                        md5_hash='a268eb855778b3df3c7506639542a6af')
            model.load_weights(weights_path, by_name=True)

        return model

    def setup(self):
        resnet_base = self.ResNet50()

        for l in resnet_base.layers:
            l.trainable = True
        conv1 = resnet_base.get_layer("activation_1").output
        conv2 = resnet_base.get_layer("activation_10").output
        conv3 = resnet_base.get_layer("activation_22").output
        conv4 = resnet_base.get_layer("activation_40").output
        conv5 = resnet_base.get_layer("activation_49").output

        up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
        conv6 = Network.conv_block_simple(up6, 256, "conv6_1")
        conv6 = Network.conv_block_simple(conv6, 256, "conv6_2")

        up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
        conv7 = Network.conv_block_simple(up7, 192, "conv7_1")
        conv7 = Network.conv_block_simple(conv7, 192, "conv7_2")

        up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
        conv8 = Network.conv_block_simple(up8, 128, "conv8_1")
        conv8 = Network.conv_block_simple(conv8, 128, "conv8_2")

        up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
        conv9 = Network.conv_block_simple(up9, 64, "conv9_1")
        conv9 = Network.conv_block_simple(conv9, 64, "conv9_2")

        up10 = UpSampling2D()(conv9)
        conv10 = Network.conv_block_simple(up10, 32, "conv10_1")
        conv10 = Network.conv_block_simple(conv10, 32, "conv10_2")
        conv10 = SpatialDropout2D(0.2)(conv10)
        x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
        model = Model(resnet_base.input, x)

        self.network = model
