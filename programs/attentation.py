
from keras.layers import *
import keras

def att (img,skipcommection,num_filters):
    X=img
    img=keras.layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",strides=1, kernel_initializer='he_normal')(X)
    img=keras.layers.BatchNormalization()(img)

    skip=keras.layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",strides=2, kernel_initializer='he_normal')(skipcommection)
    skip=keras.layers.BatchNormalization()(skip)

    concat=keras.layers.add([img,skip])
    concat=Conv2D(1, kernel_size=1, padding='same', activation='sigmoid')(concat)
    concat=UpSampling2D()(concat)

    mul=keras.layers.multiply([concat,skipcommection])
    output=mul
    return output