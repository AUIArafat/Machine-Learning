import os

import numpy as np
import keras
import random
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class ImageDataGenerator(keras.utils.Sequence):
    """
    Data Generator class using Keras
    This takes data location and their labels and some configurations to
    generate data for each step instead of loading the complete dataset at once
    """
    def __init__(self, labels, num_classes, feature_dim, duration, batch_size=64, shuffle=True,image_height = 500, image_width = 500, image_channel = 3, root_path='../data/train/'):
        """
        initialize AudioDataGenerator

        :param labels: dictionary {full_file_path: class_label}
        :param num_classes: number of classes
        :param feature_dim: audio feature dimension (time, frequency dimension)
        :param duration: duration of audio to use for computing feature
        :param batch_size: batch size during training (default 32)
        :param shuffle: whether to shuffle data during training
        """
        self.labels = labels
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.duration = duration
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_height = image_height
        self.image_width = image_width
        self.image_channel = image_channel
        self.root_path = root_path
        self.indexes = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        """
        :return: number of batches(steps) per epoch
        """
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data

        :param index: starting index of the batch
        :return: (data, label) for the batch
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:min(len(self.labels), (index+1) * self.batch_size)]
        # Generate data
        x, y = self.__data_generation(indexes)
        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = list(self.labels.keys())
        if self.shuffle:
            random.shuffle(self.indexes)

    # fuction only to read image from file
    def get_image(self, index, data, should_augment):
        # Read image and appropiate traffic light color
        file_path = data + ".jpg"
        path = os.path.join(self.root_path, file_path)
        image = cv2.imread(path)
        class_name = self.labels[data]
        return [image, class_name]

    def __data_generation(self, data, should_augment=False):
            # initializing the arrays, x_train and y_train
            # x_train = np.empty(
            #     [0, self.image_height, self.image_width, self.image_channel], dtype=np.float32)
            # y_train = np.empty([0], dtype=np.int32)

            x_train = np.empty((len(data), self.image_height, self.image_width, self.image_channel), dtype=np.float32)
            y_train = np.empty(len(data), dtype=np.int32)
            for i, file_path in enumerate(data):
                # get an image and its corresponding color for an traffic light
                #try:
                [image, class_name] = self.get_image(i, file_path, should_augment)

                # Appending them to existing batch
                x_train = np.append(x_train, [image], axis=0)
                y_train = np.append(y_train, [class_name])
                # x_train[i, ] = image
                # y_train[i] = class_name
                # print("X_train : ", x_train)
            y_train = keras.utils.to_categorical(y_train, num_classes=self.num_classes)
            return (x_train, y_train)

    # Change brightness levels
    def random_brightness(image):
        # Convert 2 HSV colorspace from BGR colorspace
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Generate new random brightness
        rand = random.uniform(0.3, 1.0)
        hsv[:, :, 2] = rand * hsv[:, :, 2]
        # Convert back to BGR colorspace
        new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return new_img

    # Zoom-in
    def zoom(self, image):
        zoom_pix = random.randint(0, 10)
        zoom_factor = 1 + (2 * zoom_pix) / self.image_height
        image = cv2.resize(image, None, fx=zoom_factor,
                           fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        top_crop = (image.shape[0] - self.image_height) // 2
        left_crop = (image.shape[1] - self.image_width) // 2
        image = image[top_crop: top_crop + self.image_height,
                left_crop: left_crop + self.image_width]
        return image
