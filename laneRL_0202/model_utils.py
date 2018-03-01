import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input,concatenate
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
import para
import pdb

#input_shape = para.input_shape
learning_rate = para.learning_rate
appended_inputshape = 1 + para.num_of_actions*para.num_of_history
'''
def get_q_network(weights_path):
    model = Sequential()
    model.add(Dense(2048,input_shape=(input_shape+appended_inputshape,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4))
    model.add(Activation('softmax'))
    #adam = Adam(lr=learning_rate)
    sgd = SGD(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    if weights_path != "0":
        model.load_weights(weights_path)
    return model

def get_concat_q_network(weights_path):
    fea_input = Input(shape=(input_shape,), name='fea_input')
    state_input = Input(shape=(appended_inputshape,), name='state_input')


    x = Dense(1024, activation='relu')(fea_input)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    mergeVec = concatenate([x, state_input])
    x = Dense(4, activation='linear')(mergeVec)

    model = Model(inputs=[fea_input, state_input], outputs=[x])
    sgd = SGD(lr=learning_rate)
    model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
    if weights_path != "0":
        model.load_weights(weights_path)
    return model
'''
def get_concat_conv_q_network(weights_path):
    fea_input = Input(shape=(4, 20, 512), name='fea_input')
    state_input = Input(shape=(appended_inputshape,), name='state_input')


    x = Conv2D(384, (3, 3), padding = 'valid', kernel_initializer='glorot_normal')(fea_input)
    x = Activation('relu')(x)
    x = Conv2D(256, (2, 2), padding = 'valid', kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)

    x = Flatten()(x)

    # x = Dense(1024, kernel_initializer='glorot_normal')(x)
    # x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    x = Dense(512, kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    # x = Dropout(0.5)(x)

    x = concatenate([x, state_input])
    x = Dense(256, kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    # x = Dense(128, kernel_initializer='glorot_normal')(x)
    # x = Activation('relu')(x)
    x = Dense(4, kernel_initializer='glorot_normal')(x)
    x = Activation('linear')(x)
    model = Model(inputs=[fea_input, state_input], outputs=[x])
    #sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)
    if weights_path != "0":
        model.load_weights(weights_path)
    return model


def get_img_conv_q_network(weights_path,conv_layer,dp):
    fea_input = Input(shape=(20, 100, 3), name='fea_input')
    state_input = Input(shape=(appended_inputshape,), name='state_input')

    x = Conv2D(48, (3, 3), padding = 'valid', kernel_initializer='glorot_normal')(fea_input)
    x = Activation('relu')(x)

    x = Conv2D(96, (3, 3), padding = 'valid', kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (2, 2), padding = 'valid', kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    if conv_layer == 4:
        x = Conv2D(192, (2, 2), padding = 'valid', kernel_initializer='glorot_normal')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    if conv_layer == 5:
        x = Conv2D(192, (2, 2), padding = 'valid', kernel_initializer='glorot_normal')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (2, 2), padding = 'valid', kernel_initializer='glorot_normal')(x)
        x = Activation('relu')(x)

    x = Flatten()(x)

    x = Dense(512, kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)

    x = Dropout(dp)(x)

    x = concatenate([x, state_input])

    x = Dense(256, kernel_initializer='glorot_normal')(x)
    x = Activation('relu')(x)
    # x = Dropout(0.2)(x)

    x = Dense(4, kernel_initializer='glorot_normal')(x)
    x = Activation('linear')(x)
    model = Model(inputs=[fea_input, state_input], outputs=[x])
    #sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=learning_rate)
    model.compile(loss='mse', optimizer=adam)
    if weights_path != "0":
        model.load_weights(weights_path)
    return model