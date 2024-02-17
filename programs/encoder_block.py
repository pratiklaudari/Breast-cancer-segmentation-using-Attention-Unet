from keras import layers
import numpy as np


def encoder_block(inputs, num_filters,rate=0.2,pooling=True):
  X = inputs
  print(X.shape)
  # Two convolutional layers with ReLU activation
  x = layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",strides=1, kernel_initializer='he_normal')(X)
  x = layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",strides=1, kernel_initializer='he_normal')(x)
  x = layers.BatchNormalization()(x)
  if pooling:
    y = layers.MaxPool2D()(x)
    return y,x
  else:
    return x

  
