from audio_classification.config import cfg

from keras.optimizers import Adam
from audio_classification.train_net import TrainNet

cfg.DATA.CREATE_CHUNK = False
cfg.MODEL.TRAIN.OUTPUT_DIR = '/home/ifilter/Desktop/WFS/Data/demo'
cfg.MODEL.EPOCHS = 25

adam = Adam(lr=0.0001)

train_audio = TrainNet(cfg, optimizer=adam)

train_audio.train()
