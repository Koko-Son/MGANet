import numpy as np


def std_normalize(data): 
    # normalize as 64 bit, to avoid numpy warnings
    data = data.astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)
    data = data.copy() - mean
    if std != 0.:
        data = data / std
    return data.astype(np.float32)


def same_tempo(true_value, estimated_value, factor=1., tolerance=0.04):
    """在阈值范围内，判断tempo是否相同"""
    if tolerance is None or tolerance == 0.0:
        return round(estimated_value * factor) == round(true_value)
    else:
        return abs(estimated_value * factor - true_value) < true_value * tolerance


def accuracy_stats(predictions, labels):
    """计算测试结果的准确率，返回三种准确率"""
    acc0_sum = 0
    acc1_sum = 0
    acc2_sum = 0
    count = 0
    for key, label in labels.items():
        if key in predictions:
            predicted_label = predictions[key]

            acc0 = same_tempo(label, predicted_label, tolerance=0.0)
            acc1 = same_tempo(label, predicted_label)
            acc2 = acc1 or same_tempo(label, predicted_label, factor=2.) \
                    or same_tempo(label, predicted_label, factor=1. / 2.) \
                    or same_tempo(label, predicted_label, factor=3.) \
                    or same_tempo(label, predicted_label, factor=1. / 3.)

            if acc0:
                acc0_sum += 1
            if acc1:
                acc1_sum += 1
            if acc2:
                acc2_sum += 1

        else:
            print('No prediction for key {}'.format(key))

        count += 1
    acc0_result = acc0_sum / float(count)
    acc1_result = acc1_sum / float(count)
    acc2_result = acc2_sum / float(count)
    
    return [acc0_result, acc1_result, acc2_result]


def predict(model, input_shape, windowed, features, labels, label_offset):
    """对一个数据集进行预测，返回预测结果"""
    results = {}
    # 对于每一个key
    for key, label in labels.items():
        # make sure we don't modify the original!
        feature = np.copy(features[key])
        if windowed:
            fragments = []
            cnt = feature.shape[1] // input_shape[1]
            for i in range(cnt):
                feature_cropped = feature[:, input_shape[1]*i : input_shape[1]*(i+1), :]
                feature_normed = std_normalize(feature_cropped)
                fragments.append(feature_normed)  
            X = np.array(fragments)
        else:
            # this assumes that we can predict spectrograms of arbitrary lengths (dim=1)
            feature = std_normalize(feature)
            X = np.expand_dims(feature, axis=0)

        predictions = model.predict(X, X.shape[0])
        predictions = np.sum(predictions, axis=0)
        index = np.argmax(predictions)

        results[key] = index + label_offset

    return results
