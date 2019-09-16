from itertools import product

from keras.optimizers import Adam

from image_classification.config import cfg
from image_classification.train_net import TrainNet

cfg.MODEL.TRAIN.OUTPUT_DIR = '/home/user/Desktop/SN/Data/output'
cfg.MODEL.EPOCHS = 40


def product_dict(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for instance in product(*values):
        yield dict(zip(keys, instance))


hyper_parameters = {
    'dropout_rate': [0.4],
    'lr': [0.0001, 0.00005, 0.00001]
}


hyper_parameters_space = list(product_dict(**hyper_parameters))
result = []

for i in range(len(hyper_parameters_space)):
    params = hyper_parameters_space[i]
    print(i, ":", params)
    cfg.MODEL.DROPOUT = params['dropout_rate']

    # need to define optimizer inside model otherwise encounter error
    adam = Adam(lr=params['lr'], beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    train_word = TrainNet(cfg, adam)

    val_acc = train_word.train(save_log=True)
    result.append(val_acc)

for i in range(len(hyper_parameters_space)):
    print('Parameters:', hyper_parameters_space[i])
    print('Validation Accuracy:', result[i])
