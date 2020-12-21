import argparse
import importlib
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model

from constant import *
from loader import load_labels, load_features
from evaluator import predict, accuracy_stats


## 测试函数
def test(annotation_file, feature_files, model):
    print('\n=============='+annotation_file.split('/')[-1].replace('.tsv', '')+'==============')

    ##--- 加载标签 ---##
    labels = load_labels(annotation_file)
    print('Loaded {} labels from {}.'.format(len(labels), annotation_file))
    
    ##--- 加载数据 ---##
    features = load_features(feature_files)
    print('Loaded features for {} files from {}.'.format(len(features), feature_files))

    ##--- 测试---##
    print('\nTesting...')
    predictions = predict(model, SHAPE, True, features, labels, L_OFFSET)
    result = accuracy_stats(predictions, labels)
    for i, acc in enumerate(result):
        print('Accuracy{}: {:.4f}'.format(i, acc*100))


# 测试准备
parser = argparse.ArgumentParser()
parser.add_argument('-n', "--network")
parser.add_argument('-m', "--mark")
args = parser.parse_args()
# exit()

# 导入网络结构
network_module = importlib.import_module('network.' + args.network)
create_model = network_module.create_model

# 载入模型
model_file = 'model/' + args.network + '_' + args.mark + '.h5'
model = create_model(input_shape=SHAPE, output_dim=N_CLASS)
model.load_weights(model_file)
model.summary()

for i in range(len(test_annotation_files)):
    test(test_annotation_files[i], [test_feature_files[i]], model)