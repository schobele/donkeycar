from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union
import donkeycar as dk

from donkeycar.pipeline.types import TubRecord

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, \
    BatchNormalization
from tensorflow.keras.layers import Activation, Dropout, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed as TD
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Conv2DTranspose
from tensorflow.keras.backend import concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from donkeycar.parts.keras import KerasPilot
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing

XY = Union[float, np.ndarray, Tuple[float, ...], Tuple[np.ndarray, ...]]

class RCSTIG_NVIDIA(KerasPilot):
    def __init__(self, num_outputs=2, input_shape=(120, 160, 3), roi_crop=(0, 0), *args, **kwargs):
        super(RCSTIG_NVIDIA, self).__init__(*args, **kwargs)
        self.model = customArchitecture(num_outputs, input_shape, roi_crop)
        self.compile()

    def compile(self):
        self.model.compile(optimizer="adam",
                loss='mse')

    def inference(self, img_arr, other_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        print("steering" + steering[0][0])
        print("steering" + throttle[0][0])
        return steering[0][0], throttle[0][0]


    def y_transform(self, record: TubRecord):
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return angle, throttle

    def y_translate(self, y: XY) -> Dict[str, Union[float, np.ndarray]]:
        if isinstance(y, tuple):
            angle, throttle = y
            return {'angle_out': angle, 'throttle_out': throttle}
        else:
            raise TypeError('Expected tuple')

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape()[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'angle_out': tf.TensorShape([]),
                   'throttle_out': tf.TensorShape([])})
        return shapes


    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        print("steering")
        print(steering[0][0]/100)
        print("throttle")
        print(steering[0][0])
        return steering[0][0]/100, throttle[0][0]/100


class RCSTIG_IMU(KerasPilot):
    def __init__(self, input_shape=(120, 160, 3), num_sensors=2):
        super().__init__()
        self.num_sensors = num_sensors
        self.model = self.create_imu_model(input_shape)
        self.compile()

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')


    def inference(self, img_arr, other_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        print("steering" + steering[0][0])
        print("steering" + throttle[0][0])
        return steering[0][0], throttle[0][0]

    def x_transform(self, record: TubRecord):
        img_arr = record.image(cached=True)
        return img_arr

    def y_transform(self, record: TubRecord):
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return angle, throttle

    def y_translate(self, y: XY) -> Dict[str, Union[float, np.ndarray]]:
        if isinstance(y, tuple):
            angle, throttle = y
            return {'angle_out': angle, 'throttle_out': throttle}
        else:
            raise TypeError('Expected tuple')

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape()[1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'angle_out': tf.TensorShape([]),
                   'throttle_out': tf.TensorShape([])})
        return shapes


    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        print("steering")
        print(steering[0][0]/100)
        print("throttle")
        print(steering[0][0])
        return steering[0][0]/100, throttle[0][0]/100

        return shapes


class RCSTIG_RNN(KerasPilot):
    def __init__(self, input_shape=(120, 160, 3), seq_length=3, num_outputs=2):
        super().__init__()
        self.input_shape = input_shape
        self.model = rnn(seq_length=seq_length,
                              num_outputs=num_outputs,
                              input_shape=input_shape)
        self.seq_length = seq_length
        self.img_seq = []
        self.batch_size = 3
        self.optimizer = "rmsprop"

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def inference(self, img_arr, other_arr):
        if img_arr.shape[2] == 3 and self.input_shape[2] == 1:
            img_arr = dk.utils.rgb2gray(img_arr)

        while len(self.img_seq) < self.seq_length:
            self.img_seq.append(img_arr)

        self.img_seq = self.img_seq[1:]
        self.img_seq.append(img_arr)
        
        img_arr = np.array(self.img_seq).reshape((1, self.seq_length,
                                                  *self.input_shape))
        outputs = self.model.predict([img_arr])
        steering = outputs[0][0]
        throttle = outputs[0][1]
        return steering, throttle

    def get_input_shape(self) -> tf.TensorShape:
        assert self.model, 'Model not set'
        return (3,120,160,3)

    def x_transform(self, record: TubRecord) -> XY:
        img_arr = record.image(cached=True)
        return img_arr

    def y_transform(self, record: TubRecord):
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return angle, throttle

    def x_translate(self, x: XY) -> Dict[str, Union[float, np.ndarray]]:
        print('x_translate')
        print(x.shape)
        return {'img_in': x}

    def y_translate(self, y: XY) -> Dict[str, Union[float, np.ndarray]]:
        if isinstance(y, tuple):
            angle, throttle = y
            return {'model_outputs': angle, 'model_outputs': throttle}
        else:
            raise TypeError('Expected tuple')

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape()
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                {'model_outputs': tf.TensorShape([]),
                'model_outputs': tf.TensorShape([])})

        return shapes

def customArchitecture(num_outputs, input_shape, roi_crop):

    #input_shape = adjust_input_shape(input_shape, roi_crop)
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in


    #x = tf.keras.layers.Cropping2D(cropping=((60, 0), (0, 0)))(x)
    #x = tf.keras.layers.experimental.preprocessing.Resizing(120, 160)
    
    # Dropout rate
    keep_prob = 0.9
    rate = 1 - keep_prob
    
    # Convolutional Layer 1
    x = Convolution2D(filters=24, kernel_size=5, strides=(2, 2), input_shape = input_shape)(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 2
    x = Convolution2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 3
    x = Convolution2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 4
    x = Convolution2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu')(x)
    x = Dropout(rate)(x)

    # Convolutional Layer 5
    x = Convolution2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu')(x)
    x = Dropout(rate)(x)

    # Flatten Layers
    x = Flatten()(x)

    # Fully Connected Layer 1
    x = Dense(100, activation='relu')(x)

    # Fully Connected Layer 2
    x = Dense(50, activation='relu')(x)

    # Fully Connected Layer 3
    x = Dense(25, activation='relu')(x)
    
    # Fully Connected Layer 4
    x = Dense(10, activation='relu')(x)
    
    # Fully Connected Layer 5
    x = Dense(5, activation='relu')(x)
    outputs = []
    
    for i in range(num_outputs):
        # Output layer
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))
        
    model = Model(inputs=[img_in], outputs=(Dense(1, activation='linear', name='angle_out')(x),Dense(1, activation='linear', name='throttle_out')(x)))
    
    return model




def rnn(seq_length=3, num_outputs=2, input_shape=(120, 160, 3)):
    # add sequence length dimensions as keras time-distributed expects shape
    # of (num_samples, seq_length, input_shape)
    img_seq_shape = (seq_length,) + input_shape   
    print('img_seq_shape')
    print(img_seq_shape)
    img_in = Input(shape=input_shape, name='img_in')
    drop_out = 0.3

    x = Sequential()
    x.add(img_in)
    x.add(TD(Convolution2D(24, (5,5), strides=(2,2), activation='relu'),
             input_shape=input_shape))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Convolution2D(32, (5, 5), strides=(2, 2), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Convolution2D(32, (3, 3), strides=(2, 2), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(Convolution2D(32, (3, 3), strides=(1, 1), activation='relu')))
    x.add(TD(Dropout(drop_out)))
    x.add(TD(MaxPooling2D(pool_size=(2, 2))))
    x.add(TD(Flatten(name='flattened')))
    x.add(TD(Dense(100, activation='relu')))
    x.add(TD(Dropout(drop_out)))
      
    x.add(LSTM(128, return_sequences=True, name="LSTM_seq"))
    x.add(Dropout(.1))
    x.add(LSTM(128, return_sequences=False, name="LSTM_fin"))
    x.add(Dropout(.1))
    x.add(Dense(128, activation='relu'))
    x.add(Dropout(.1))
    x.add(Dense(64, activation='relu'))
    x.add(Dense(10, activation='relu'))
    x.add(Dense(num_outputs, activation='linear', name='model_outputs'))

    return x