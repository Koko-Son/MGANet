import math
import random
import numpy as np

from keras.preprocessing.image import apply_affine_transform
from keras.utils import to_categorical


def std_normalize(data): 
    # normalize as 64 bit, to avoid numpy warnings
    data = data.astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)
    data = data.copy() - mean
    if std != 0.:
        data = data / std
    return data.astype(np.float32)


def radom_crop(feature, label, label_max, augmentation, n_frame):
    ## 选取合适的缩放尺度，并获得新标签
    if augmentation == True:
        scale_factor = random.choice([x / 100.0 for x in range(80, 124, 4)])
        label_scaled = label * scale_factor
        # 当类别标签超出上限
        while round(label_scaled) >= label_max:
            scale_factor -= 0.04
            label_scaled = label * scale_factor
    else:
        scale_factor = 1
        label_scaled = label

    # 需要保留的最小尺寸
    length = max(n_frame, math.ceil(n_frame * scale_factor))
    # 当所频谱长度不够，先补零
    if feature.shape[1] < length:
        feature = np.pad(feature, ((0, 0), (0, length - feature.shape[1]), (0, 0)))
    # 剪裁频谱图
    offset = random.randint(0, feature.shape[1] - length)
    feature_unscale = feature[:, offset : offset + length, :]

    # 非1时才缩放，提高效率
    if scale_factor != 1:
        # 缩放频谱图（length->scale_factor*length）
        feature_scaled = apply_affine_transform(
            feature_unscale,
            zy=1/scale_factor, fill_mode='nearest'
        ).astype(np.float32)
        # 只保留所需要的长度
        feature_cropped = feature_scaled[:, :n_frame, :]
    else:
        feature_cropped = feature_unscale
    
    return feature_cropped, label_scaled


def create_data_generator(features, labels, augmentation, label_offset, n_class, n_frame, batch_size):
    # 从labels获得key的list
    keys = list(labels.keys())
    length = len(keys)

    idx = 0
    random.shuffle(keys)
    while True:
        # 每个epoch后进行shuffle
        if idx + batch_size > length:
            idx = 0
            random.shuffle(keys)

        # 每次取batch_size个key
        X, y = [], []
        for i in range(idx, idx + batch_size):
            # 对每一个，取feature
            feature = features[keys[i]]
            label = labels[keys[i]]
            # 裁剪数据并进行数据增强
            feature_cropped, label_scaled = \
                radom_crop(feature, label, label_offset + n_class, augmentation, n_frame)
            # 对数据进行标准化
            feature_normed = std_normalize(feature_cropped)
            # 存入集合
            X.append(feature_normed)
            y.append(round(label_scaled - label_offset))

        # 对label进行one-hot编码
        y_coded = to_categorical(y, num_classes=n_class)

        idx += batch_size
        yield np.stack(X, axis=0), np.stack(y_coded, axis=0)


