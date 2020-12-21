import os
import argparse
import importlib
import shutil
import time
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from constant import *
from loader import load_labels, load_features
from generator import create_data_generator
from evaluator import predict, accuracy_stats


##--- 训练准备 ---##
parser = argparse.ArgumentParser()
parser.add_argument('-n', "--network")
parser.add_argument('-m', "--mark")
args = parser.parse_args()

# 导入网络结构
network_module = importlib.import_module('network.' + args.network)
create_model = network_module.create_model

# 模型文件
checkpoint_model_file = 'model/' + args.network + '_' + args.mark + '.h5'

# 日志记录
log_file_name = checkpoint_model_file.replace('model/', 'log/').replace('.h5', '.log')
log_file = open(log_file_name, 'wb')
def log(message):
    message_bytes = '{}\n'.format(message).encode(encoding='utf-8')
    log_file.write(message_bytes)
    print(message)


##--- 加载标签 ---##
train_labels = load_labels(train_annotation_file)
log('Loaded {} training labels from {}.'.format(len(train_labels), train_annotation_file))
valid_labels = load_labels(valid_annotation_file)
log('Loaded {} validation labels from {}.'.format(len(valid_labels), valid_annotation_file))


##--- 加载数据 ---##
train_features = load_features(train_feature_files)
log('Loaded features for {} files from {}.'.format(len(train_features), train_feature_files))
valid_features = load_features(valid_feature_files)
log('Loaded features for {} files from {}.'.format(len(valid_features), valid_feature_files))


##--- Data Generator ---##
log('\nCreating generators...')
train_generator = create_data_generator(
    train_features, train_labels, 
    augmentation=True, label_offset=L_OFFSET, n_class=N_CLASS, 
    n_frame=N_FRAME, batch_size=BATCH_SIZE
)


##--- 网络 ---##
log('\nCreating model...')
model = create_model(input_shape=SHAPE, output_dim=N_CLASS)
model.compile(loss='categorical_crossentropy', optimizer=(Adam(lr=LR)), metrics=['accuracy'])
# model.summary()


##--- 开始训练 ---##
log('\nTaining...')
log('params={}'.format(model.count_params()))

i_epoch, i_iter = 1, 1
best_acc, best_epoch = 0, 0
mean_loss, mean_acc = 0, 0
epoch_batch = 500
# expoch_batch = len(train_labels) // BATCH_SIZE
time_start = time.time()
while i_epoch < EPOCHS:
    # 取1个batch数据
    X, y = next(train_generator)

    # 训练1个iteration
    i_iter = i_iter % epoch_batch + 1
    loss, acc = model.train_on_batch(X, y)
    mean_loss += loss
    mean_acc += acc

    print('Epoch {}/{} - {}/{} - loss {:.4f} - accuracy {:.4f}'.format(
        i_epoch, EPOCHS, i_iter, epoch_batch, loss, acc
    ), end='\r')

    # 输出信息
    if i_iter == 1:
        # train message
        time_train = time.time() - time_start
        mean_loss /= epoch_batch
        mean_acc /= epoch_batch
        print('', end='\r')
        log('Epoch {}/{} - time {:.1f}s - loss {:.4f} - accuracy {:.4f}'.format(
            i_epoch, EPOCHS, time_train, mean_loss, mean_acc
        ))
        
        # valid result
        predictions = predict(model, SHAPE, True, valid_features, valid_labels, L_OFFSET)
        valid_result = accuracy_stats(predictions, valid_labels)

        # check point
        if valid_result[1] >= best_acc:
            best_acc = valid_result[1]
            best_epoch = i_epoch
            model.save_weights(checkpoint_model_file)
        
        log('ACC0 {:.2f}%  ACC1 {:.2f}%  ACC2 {:.2f}%  Best {:.2f}% {}'.format(
            valid_result[0]*100, valid_result[1]*100, valid_result[2]*100, best_acc*100, best_epoch
        ))

        # early stopping
        if i_epoch - best_epoch > PATIENCE:
            log('Early Stopping.')
            break
        
        i_epoch += 1
        mean_loss, mean_acc = 0, 0
        time_start = time.time()

log_file.close()
