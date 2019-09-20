from models.image_classification_cnn import ImageClassificationCNN
from models.image_classification_vgg16 import ImageClassificationVGG16
from models.image_classification_resnet50 import ImageClassificationResNet50

def get_network(name, params):
    """

    :param name: network name
    :param params: {key, value} parameters
    :return: keras network if it exists else raise KeyError
    """
    if name == 'image_classification_cnn':
        return ImageClassificationCNN(**params).get()
    if name == 'image_classification_VGG16':
        return ImageClassificationVGG16(**params).get()
    if name == 'image_classification_ResNet50':
        return ImageClassificationResNet50(**params).get()
    else:
        raise KeyError("Unknown network " + name)
