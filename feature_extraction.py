import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import csv
import joblib
import librosa

from constant import *


def extract_features(file):
    y, _ = librosa.load(file, sr=SR)
    data = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=N_HOP,
                                          power=1, n_mels=N_MEL, fmin=20, fmax=5000)
    data = np.reshape(data, (data.shape[0], data.shape[1], 1))
    return data


def folder_to_joblib(base_folder, annotation_file, feature_file):
    # 读取标注文件
    with open(annotation_file, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        labeled_keys = [row[0] for row in reader]

    # 提取频谱特征
    feature_set = {}
    for (dirpath, _, filenames) in os.walk(base_folder):
        for file in [f for f in filenames if f.endswith('.mp3') or f.endswith('.wav')]:
            # 以带有相对路径的文件名作为key，保留数据集名和后缀名
            key = (dirpath + '\\' + file).replace(data_path, '')
            
            # 如果被标注，则提取Mel-spectrogram
            if key in labeled_keys:
                features = extract_features(os.path.join(dirpath, file))
                feature_set[key] = features
                print(file, '->', key)

    joblib.dump(feature_set, feature_file)
    print(len(feature_set), 'features saved to', feature_file)


if __name__ == '__main__':
    folders = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    folders = [f for f in folders if not os.path.isfile(f)]
    for folder in folders:
        feature_file = feature_path + folder.split('\\')[-1] + '.joblib'
        annotation_file = 'annotation\\' + folder.split('\\')[-1] + '.tsv'
        if os.path.exists(feature_file):
            continue
        print("Converting", folder)
        folder_to_joblib(folder, annotation_file, feature_file)

