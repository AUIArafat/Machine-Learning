from easydict import EasyDict

cfg = EasyDict()

cfg.DATA = EasyDict()
# data directory (after creating chunks)
cfg.DATA.DIR = '/home/user/Desktop/SN/Data/'
# label file used during training
cfg.DATA.LABEL_FILE = '/home/user/Desktop/SN/Data/csv_files/labels.csv'

cfg.DATA.CATEGORIES = ('Doorbell', 'Oven', 'Door', 'Bathtub', 'Laughter', 'Dog', 'Cat', 'Ringtone',
                       'Baby Crying')

cfg.MODEL = EasyDict()
cfg.MODEL.NUM_CLASSES = 9
cfg.MODEL.FEATURE_DIM = (128, 128)
cfg.MODEL.DURATION = 1  # Audio duration in seconds
cfg.MODEL.DROPOUT = 0.5
cfg.MODEL.ACTIVATION = 'relu'

cfg.MODEL.NETWORK = 'audio_classification_cnn'

cfg.MODEL.TRAIN = EasyDict()
cfg.MODEL.TRAIN.BATCH_SIZE = 64
cfg.MODEL.TRAIN.OUTPUT_DIR = '/home/user/Desktop/SN/Data'
cfg.MODEL.TRAIN.DATA_SPLIT = (0.8, 0.1, 0.1)
cfg.MODEL.TRAIN.SAVE_BEST_ONLY = True
cfg.MODEL.TRAIN.RANDOM_SEED = 1
cfg.MODEL.EPOCHS = 50
