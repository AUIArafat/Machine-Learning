import os
import time
import numpy as np

from data_prepare.audio_feature import load_audio, get_mel_spectrogram
from data_prepare.augmentation import time_stretch, pitch_shift, add_background_noise


def time_analysis():
    """
    compute time required in reading audio, performing augmentation, feature extraction
    """
    files = os.listdir('test_data/test_time')
    time_read = 0
    time_ts = 0
    time_ps = 0
    time_bg = 0
    time_feature = 0

    sampling_rate = 44100
    y_noise = load_audio('test_data/bg_people.wav', sr=sampling_rate, duration=1)

    time_total_start = time.time()

    for file in files:
        start = time.time()
        y = load_audio(os.path.join('test_data/test_time', file), sr=sampling_rate, duration=1)
        time_read += time.time() - start

        start = time.time()
        _ = time_stretch(y, 1.2)
        time_ts += time.time() - start

        start = time.time()
        _ = pitch_shift(y, sampling_rate, 2)
        time_ps += time.time() - start

        start = time.time()
        if len(y) < len(y_noise):
            y = np.pad(y, (0, len(y_noise) - len(y)), 'constant', constant_values=0)
        _ = add_background_noise(y, y_noise, 0.5)
        time_bg += time.time() - start

        start = time.time()
        _ = get_mel_spectrogram(y)
        time_feature += time.time() - start

    print('Time taken in reading audio:', time_read)
    print('Time taken in time stretching audio:', time_ts)
    print('Time taken in pitch shifting:', time_ps)
    print('Time taken in adding noise:', time_bg)
    print('Time taken in extracting feature:', time_feature)
    print('Total time taken:', time.time()-time_total_start)


if __name__ == '__main__':
    time_analysis()
