import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Lambda, Conv2D, Conv1D, Dense, BatchNormalization, Dropout, 
    GlobalAveragePooling2D, AveragePooling1D, AveragePooling2D, 
    GlobalMaxPooling2D, MaxPooling1D, MaxPooling2D, UpSampling2D,
    Multiply, Add, Reshape, Softmax, Flatten, Concatenate, Maximum, Average, Dot
)


def grouped_attention_branches(x, k, p, reduc):
    height = K.int_shape(x)[-3]
    length = K.int_shape(x)[-2]
    channel = K.int_shape(x)[-1]
    x = BatchNormalization()(x)

    x_list = []
    f_seg = height//k
    for i in range(k):
        # context modeling
        x_i = Lambda(lambda x: x[:, i*f_seg: (i+1)*f_seg, :, :])(x)
        w_k = Conv2D(1, (1, 1), activation='elu')(x_i)
        w_k = Reshape((1, f_seg * length))(w_k)
        w_k = Softmax(axis=-1)(w_k)
        x_i = Reshape((f_seg*length, channel))(x_i)
        x_i = Dot(axes=(1, 2))([x_i, w_k]) # (F*T, C), (1, F*T) -> (1, C)
        
        # transform
        x_i = Flatten()(x_i)
        x_i = BatchNormalization()(x_i)
        x_i = Dense(channel//reduc, activation='elu')(x_i)
        x_i = Dense(channel, activation='sigmoid')(x_i)
        x_i = Reshape((1, channel))(x_i)
        for j in range(height//k//p):
            x_list.append(x_i)

    if len(x_list) > 1:
        x = Concatenate(axis=-2)(x_list) # (height//shrink, channel)
    else:
        x = x_list[0]
    x = Reshape((height//p, 1, channel))(x)
    return x


def grouped_attention_module(x, C, k, p, reduc=4):
    # adjust channel num.
    x = BatchNormalization()(x)
    x = Conv2D(C, (1, 1), padding='same', activation='elu')(x)

    # trunk branch
    trunk = Conv2D(C, (3, 3), padding='same', activation='elu')(x)
    trunk = Conv2D(C, (3, 3), padding='same', activation='elu')(trunk)
    trunk = Conv2D(C, (3, 3), padding='same', activation='elu')(trunk)
    trunk = AveragePooling2D((p, 1))(trunk)

    # attention branches
    att = grouped_attention_branches(x, k, p, reduc)
    trunk = Multiply()([trunk, att])
    return trunk


def mix_module(x0, C, k=2, p=2):
    x0 = grouped_attention_module(x0, C, k, p)
    return x0


def create_model(input_shape=(40, 256, 1), output_dim=[144, 3]):
    inputs = Input(shape=input_shape)
    
    # input
    x0 = inputs # (81, 256)
    x1 = AveragePooling2D((1, 2))(x0) # (81, 128)
    x2 = AveragePooling2D((1, 2))(x1) # (81, 64)
    
    # stage 1 (81, T, 1) -> (27, T, 32)
    x0, x1, x2 = mix_module(x0, x1, x2, C=32, k=3, p=3)
    # stage 2 (27, T, 32) -> (9, T, 64)
    x0, x1, x2 = mix_module(x0, x1, x2, C=64, k=3, p=3)
    # stage 3 (9, T, 64) -> (3, T, 128)
    x0, x1, x2 = mix_module(x0, x1, x2, C=128, k=3, p=3)
    # stage 4 (3, T, 128) -> (1, T, 128)
    x0, x1, x2 = mix_module(x0, x1, x2, C=128, k=1, p=3)

    x = [x0, x1, x2]

    for i in range(3):
        x[i] = BatchNormalization()(x[i])
        x[i] = Conv2D(256, (1, 1), padding='same', activation='elu')(x[i])
        x[i] = GlobalAveragePooling2D()(x[i]) # (output_dim,)
    x = Concatenate()(x)
    x = Dense(output_dim)(x)
    x = Softmax()(x)

    return Model(inputs=inputs, outputs=x)
