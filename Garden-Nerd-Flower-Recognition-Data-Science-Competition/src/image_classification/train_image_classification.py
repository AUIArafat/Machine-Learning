from image_classification.config import cfg

from keras.optimizers import Adam
from image_classification.train_net import TrainNet

cfg.DATA.CREATE_CHUNK = False
cfg.MODEL.TRAIN.OUTPUT_DIR = '../OutputResult'
cfg.MODEL.EPOCHS = 25
cfg.MODEL.NETWORK = 'image_classification_ResNet50'
adam = Adam(lr=0.0001)

train_audio = TrainNet(cfg, optimizer=adam)
train_audio.train()
