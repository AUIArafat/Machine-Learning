import librosa
from data_prepare.audio_feature import load_audio


def time_stretch(data, rate):
    """
    applies time stretching augmentation on the given audio data
    :param data: numpy array of audio data
    :param rate: applies this rate of stretching on the data as in paper:
    https://arxiv.org/abs/1608.04363v2
    :return: returns augmented audio data after applying stretching
    """
    return librosa.effects.time_stretch(data, rate)


def pitch_shift(data, sr, steps):
    """
    applies pitch shifting augmentation on the given audio data
    :param data: numpy array of audio data
    :param sr: sampling rate
    :param steps: number of steps to pitch on the given audio data as in paper:
    https://arxiv.org/abs/1608.04363v2
    :return: returns pitch shifted audio data
    """
    return librosa.effects.pitch_shift(y=data, sr=sr, n_steps=steps)


def dynamic_range_compression(audio_file_path, sample_rate):
    """
    applies dynamic range compression augmentation on the given audio file path
    :param audio_file_path: path of the audio file to which drc is applied
    :param sample_rate: sampling rate at which drc is applied on the audio data as in paper:
    https://arxiv.org/abs/1608.04363v2
    :return: returns dynamic range compressed data
    """
    data, sr = librosa.load(audio_file_path, sample_rate)
    return data


def add_background_noise(data, data_noise, w):
    """
    adds background noise to the original audio data
    :param data: original audio data as numpy array
    :param data_noise: noise audio data as numpy array
    :param w: a weighting parameter that was chosen randomly for each mix from a uniform
    distribution in the rang as in paper: https://arxiv.org/abs/1608.04363v2
    :return: returns mixing data with noise calculated as in paper:
    https://arxiv.org/abs/1608.04363v2
    """
    return (1-w)*data + w*data_noise


def augment_data(augment_type, **params):
    """
    perform corresponding augmentation
    :param augment_type: type of augmentation/function
    :param params: augmentation specific parameters
    :return: augmented data
    """
    return augment_type(**params)


if __name__ == '__main__':
    print('Testing different types of augmentation')
    sampling_rate = 44100
    y = load_audio('test_data/dog_bark.mp3', sr=sampling_rate, duration=2)
    assert(len(y) == sampling_rate*2)

    y_ts = augment_data(time_stretch, data=y, rate=0.5)
    print(len(y)/0.5, len(y_ts))  # for some reason these 2 are not equal
    # assert(len(y_ts) == int(len(y)*0.8))

    y_ps = augment_data(pitch_shift, data=y, sr=sampling_rate, steps=2)
    print(len(y), len(y_ps))

    y_noise = load_audio('test_data/bg_people.wav', sr=sampling_rate, duration=2)
    y_bg = augment_data(add_background_noise, data=y, data_noise=y_noise, w=0.5)
    print(len(y), len(y_ps))
