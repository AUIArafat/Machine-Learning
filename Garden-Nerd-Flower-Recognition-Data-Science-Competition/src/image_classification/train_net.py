import os
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.losses import categorical_crossentropy
import keras.backend as k
import tensorflow as tf

from data_prepare.image_data_generator import ImageDataGenerator
from utils.file_utils import read_csv_label, write_pickle

from utils.cnn_utils import split_data, save_plot, write_model_performance
from models.factory import get_network


class TrainNet:
    def __init__(self, cfg, optimizer, loss=categorical_crossentropy):
        """
        :param cfg: configuration object
        :param optimizer: training optimizer
        :param loss: loss function
        """
        self.cfg = cfg
        self.params = {}
        self.optimizer = optimizer

        self.loss = loss
        self.test_data = None
        self.training_generator, self.validation_generator, self.test_generator = self.prepare_data()

    def set_network_params(self):
        """
        set model parameters from config
        """
        self.params['num_classes'] = self.cfg.MODEL.NUM_CLASSES
        self.params['dropout_rate'] = self.cfg.MODEL.DROPOUT
        self.params['activation'] = self.cfg.MODEL.ACTIVATION

    def prepare_data(self):
        """
        prepare Data training/validation/test Generators
        :return: tuple of (training_generator, validation_generator, test_generator)
        """
        labels = read_csv_label(self.cfg.DATA.LABEL_TRAIN_FILE)
        labels_test = read_csv_label(self.cfg.DATA.LABEL_TEST_FILE)
        labels_train, labels_validation = split_data(labels, self.cfg.MODEL.TRAIN.DATA_SPLIT)
        self.test_data = labels_test

        print("labels train : ", labels_train)
        num_classes = self.cfg.MODEL.NUM_CLASSES
        feature_dim = self.cfg.MODEL.FEATURE_DIM
        duration = self.cfg.MODEL.DURATION

        training_generator = ImageDataGenerator(labels_train, num_classes, feature_dim, duration)
        validation_generator = ImageDataGenerator(labels_validation, num_classes, feature_dim, duration)
        test_generator = ImageDataGenerator(labels_test, num_classes, feature_dim, duration, shuffle=False)

        print("train generator : ", training_generator)
        return training_generator, validation_generator, test_generator

    def train(self, save_log=True):
        """
        train the model, save model history, evaluate model on test dataset using best model obtained from
        validation set, save model performance.

        :return: best validation accuracy
        """
        self.set_network_params()
        model = get_network(self.cfg.MODEL.NETWORK, self.params)

        if self.cfg.MODEL.TRAIN.SAVE_BEST_ONLY:
            checkpoint_file_name = os.path.join(self.cfg.MODEL.TRAIN.OUTPUT_DIR, 'weights.hdf5')
        else:
            checkpoint_file_name = os.path.join(self.cfg.MODEL.TRAIN.OUTPUT_DIR,
                                                'weights.{epoch:03d}-{''val_acc:.4f}.hdf5')

        checkpoint = ModelCheckpoint(checkpoint_file_name, monitor='val_acc', verbose=1, save_best_only=True,
                                     mode='auto')

        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        print("Model : ", model)
        history = model.fit_generator(
            # generator=self.training_generator,
            # validation_data=self.validation_generator,
            # use_multiprocessing=True,
            # verbose=1,
            # epochs=self.cfg.MODEL.EPOCHS,
            # callbacks=[checkpoint]

            self.training_generator,
            #steps_per_epoch=len(d_train) // BATCH_SIZE,
            epochs=self.cfg.MODEL.EPOCHS,
            validation_data=self.validation_generator,
            #validation_steps=len(d_valid) // BATCH_SIZE,
            verbose=1,
            callbacks=[checkpoint]
        )

        # performance using best model
        y_true = list(self.test_data.values())
        model.load_weights(checkpoint_file_name)
        validation_res = model.evaluate_generator(self.validation_generator)

        if save_log:
            write_pickle(history.history, os.path.join(self.cfg.MODEL.TRAIN.OUTPUT_DIR, 'history.pkl'))

            save_plot(history, 'acc', self.cfg.MODEL.TRAIN.OUTPUT_DIR)
            save_plot(history, 'loss', self.cfg.MODEL.TRAIN.OUTPUT_DIR)

            test_prediction = model.predict_generator(self.test_generator)
            predicted_labels = np.argmax(test_prediction, axis=1)

            fname = 'performance_' + str(self.cfg.MODEL.DROPOUT) + '.csv'

            write_model_performance(y_true, predicted_labels,
                                    os.path.join(self.cfg.MODEL.TRAIN.OUTPUT_DIR, fname),
                                    self.cfg.DATA.CATEGORIES)
        del history
        tf.reset_default_graph()
        k.clear_session()
        return validation_res[1]
