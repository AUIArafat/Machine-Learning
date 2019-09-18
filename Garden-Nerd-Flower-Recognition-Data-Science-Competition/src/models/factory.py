from models.image_classification_cnn import ImageClassificationCNN


def get_network(name, params):
    """

    :param name: network name
    :param params: {key, value} parameters
    :return: keras network if it exists else raise KeyError
    """
    if name == 'audio_classification_cnn':
        return ImageClassificationCNN(**params).get()
    else:
        raise KeyError("Unknown network " + name)
