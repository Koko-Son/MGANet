import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Lambda, Conv2D, Conv1D, Dense, BatchNormalization, Dropout, 
    GlobalAveragePooling2D, AveragePooling1D, AveragePooling2D, 
    GlobalMaxPooling2D, MaxPooling1D, MaxPooling2D, UpSampling2D,
    Multiply, Add, Reshape, Softmax, Flatten, Concatenate, Maximum, Average, Dot
)


def grouped_attention_module(x, C, p, reduc=4):
    # adjust channel num.
    x = BatchNormalization()(x)
    x = Conv2D(C, (1, 1), padding='same', activation='elu')(x)

    # trunk branch
    trunk = Conv2D(C, (3, 3), padding='same', activation='elu')(x)
    trunk = Conv2D(C, (3, 3), padding='same', activation='elu')(trunk)
    trunk = Conv2D(C, (3, 3), padding='same', activation='elu')(trunk)
    trunk = AveragePooling2D((p, 1))(trunk)

    return trunk


def mix_module(x0, x1, x2, C, p):
    # multi resolution
    x0 = grouped_attention_module(x0, C, p)
    x1 = grouped_attention_module(x1, C, p)
    x2 = grouped_attention_module(x2, C, p)

    # exchange information
    x0_1 = AveragePooling2D((1, 2))(x0)
    x0_2 = AveragePooling2D((1, 2))(x0_1)
    x1_0 = UpSampling2D((1, 2), interpolation='bilinear')(x1)
    x1_2 = AveragePooling2D((1, 2))(x1)
    x2_1 = UpSampling2D((1, 2), interpolation='bilinear')(x2)
    x2_0 = UpSampling2D((1, 2), interpolation='bilinear')(x2_1)
    x0_c = Concatenate()([x0, x1_0, x2_0])
    x1_c = Concatenate()([x0_1, x1, x2_1])
    x2_c = Concatenate()([x0_2, x1_2, x2])
    x0_c = Conv2D(C, (1, 1), padding='same', activation='elu')(x0_c)
    x1_c = Conv2D(C, (1, 1), padding='same', activation='elu')(x1_c)
    x2_c = Conv2D(C, (1, 1), padding='same', activation='elu')(x2_c)

    return x0_c, x1_c, x2_c


def create_model(input_shape=(40, 256, 1), output_dim=[144, 3]):
    inputs = Input(shape=input_shape)
    
    # input
    x0 = inputs # (81, 256)
    x1 = AveragePooling2D((1, 2))(x0) # (81, 128)
    x2 = AveragePooling2D((1, 2))(x1) # (81, 64)
    
    # stage 1 (81, T, 1) -> (27, T, 32)
    x0, x1, x2 = mix_module(x0, x1, x2, C=32, p=3)
    # stage 2 (27, T, 32) -> (9, T, 64)
    x0, x1, x2 = mix_module(x0, x1, x2, C=64, p=3)
    # stage 3 (9, T, 64) -> (3, T, 128)
    x0, x1, x2 = mix_module(x0, x1, x2, C=128, p=3)
    # stage 4 (3, T, 128) -> (1, T, 128)
    x0, x1, x2 = mix_module(x0, x1, x2, C=128, p=3)

    x = [x0, x1, x2]

    for i in range(3):
        x[i] = BatchNormalization()(x[i])
        x[i] = Conv2D(256, (1, 1), padding='same', activation='elu')(x[i])
        x[i] = GlobalAveragePooling2D()(x[i]) # (output_dim,)
    x = Concatenate()(x)
    x = Dense(output_dim)(x)
    x = Softmax()(x)

    return Model(inputs=inputs, outputs=x)

