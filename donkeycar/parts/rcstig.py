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
        return steering[0][0], throttle[0][0]

        return shapes

def customArchitecture(num_outputs, input_shape, roi_crop):

    #input_shape = adjust_input_shape(input_shape, roi_crop)
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in
    
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
