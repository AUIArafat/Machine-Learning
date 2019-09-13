import os
import csv
import subprocess
from tqdm import tqdm

# AudioSet ID for selected categories
audioset_class_ids = {
    'doorbell': '/m/03wwcy',
    'oven': '/m/0fx9l',
    'door': '/m/02dgv',
    'bathtub': '/m/03dnzn',
    'laughter': '/m/01j3sz',
    'dog': '/m/0bt9lr',
    'cat': '/m/01yrx',
    'ringtone': '/m/01hnzm',
    'baby_cry': '/t/dd00002'
}


def is_required(label):
    """
    if the audio has a category from our list
    """
    label_list = label.split(',')
    for _, label in audioset_class_ids.items():
        if label in label_list:
            return True
    return False


def download_audio(csv_name):
    """
    AudioSet has 3 different csv files balanced_train_segments.csv, unbalanced_train_segments and eval_segments
    we will download them separately
    each csv file has format YTID, start_seconds, end_seconds, positive_labels
    """

    with open(csv_name, 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        data_list = list(reader)

    data_list_categories = []
    for i in range(3, len(data_list)):  # skip first three comment lines
        row = data_list[i]
        if is_required(row[3]):
            data_list_categories.append(row)

    for i in tqdm(range(len(data_list_categories))):
        try:
            # print('youtube-dl -f 22 --get-url "https://youtube.com/watch?v='+data_list[i][0]+'"')
            return_result = \
                subprocess.run('youtube-dl --get-url https://youtube.com/watch?v=' + data_list_categories[i][0],
                               shell=True, stdout=subprocess.PIPE)
            if return_result.returncode == 0:
                for idx, byte in enumerate(return_result.stdout):
                    if byte == 10:
                        url = return_result.stdout[idx+1:-1].decode('utf-8')
                        duration = int(float(data_list_categories[i][2]) - float(data_list_categories[i][1]))
                        print(duration)
                        cmd = 'ffmpeg -ss ' + str(int(float(data_list_categories[i][1]))) + ' -i "' + url \
                              + '" -t ' + str(duration) + ' -f mp3 ' + \
                              os.path.join('ffmpeg_test', str(data_list_categories[i][0]) + '.mp3')
                        subprocess.run(cmd, shell=True, stdout=subprocess.PIPE)
                        break
        except Exception as e:
            print(e)


if __name__ == '__main__':
    DATA_ROOT = 'labels'
    download_audio(os.path.join(DATA_ROOT, 'balanced_train_segments.csv'))
    download_audio(os.path.join(DATA_ROOT, 'unbalanced_train_segments.csv'))
