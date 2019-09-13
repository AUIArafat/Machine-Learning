from models.audio_classification_cnn import AudioClassificationCNN


def get_network(name, params):
    """

    :param name: network name
    :param params: {key, value} parameters
    :return: keras network if it exists else raise KeyError
    """
    if name == 'audio_classification_cnn':
        return AudioClassificationCNN(**params).get()
    else:
        raise KeyError("Unknown network " + name)
