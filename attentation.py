import TensorFlow as tf
from keras.layers import *
import keras

def att (img,skipcommection):
    img=keras.layers.Conv2D()(img)
    img=keras.layers.BatchNormalization()(img)

    skipcommection=keras.layers.Conv2D()(skipcommection)
    skipcommection=keras.layers.BatchNormalization()(skipcommection)    

    concat=keras.layers.concatenate([img,skipcommection],Activation=ReLU)
    concat=keras.layers.Conv2D(concat)
    concat.activation.sigmoid()

    mul=keras.layers.multiply([concat,skipcommection])
    output=mul