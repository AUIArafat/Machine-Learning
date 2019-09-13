import numpy as np
import keras
import random

from data_prepare.audio_feature import load_audio, get_mel_spectrogram


class AudioDataGenerator(keras.utils.Sequence):
    """
    Data Generator class using Keras
    This takes data location and their labels and some configurations to
    generate data for each step instead of loading the complete dataset at once
    """
    def __init__(self, labels, num_classes, feature_dim, duration, batch_size=64, shuffle=True):
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

    def __data_generation(self, labels_temp):
        """
        Generates data containing batch_size samples

        :param labels_temp: class label for this batch(not one hot encoded)
        :return: (data, class_label) where class label is in categorical(one hot) form
        """
        # Initialization
        x = np.empty((len(labels_temp), self.feature_dim))
        y = np.empty(len(labels_temp), dtype=int)

        # Generate data
        for i, file_path in enumerate(labels_temp):
            # load audio
            audio = load_audio(file_path, duration=self.duration)
            feature = get_mel_spectrogram(audio)
            # ensure feature dimension as required
            if feature.shape[1] < self.feature_dim[1]:
                feature = np.tile(feature, int(np.ceil(self.feature_dim[1]/feature.shape[1])))

            start_idx = np.random.randint(1 + x.shape[1]-128)
            feature = feature[:, start_idx: start_idx+self.feature_dim[0]]

            x[i, ] = feature
            # store class
            y[i] = self.labels[file_path]

        return x, keras.utils.to_categorical(y, num_classes=self.num_classes)
