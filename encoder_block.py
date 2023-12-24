from keras import layers
from keras.preprocessing import image
import numpy as np


def encoder_block(inputs, num_filters, use_batch_norm=True, pool_size=(2, 2)):
  x = inputs

  # Two convolutional layers with ReLU activation
  x = layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu")(x)
  x = layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu")(x)

  # Optional batch normalization
  if use_batch_norm:
    x = layers.BatchNormalization()(x)

  # Max pooling
  skip_connection = x
  x = layers.MaxPooling2D(pool_size=pool_size)(x)

  return x, skip_connection


def unet_encoder(input_image, start_filters=64, num_levels=4):
  skip_connections = []

  # Downsampling encoder path
  x = input_image
  for level in range(num_levels):
    num_filters = start_filters * 2**level
    x, skip_connection = encoder_block(x, num_filters)
    skip_connections.append(skip_connection)

  return x, skip_connections

# Example usage
def encoder(img,num_filters) :
  img=image.img_to_array(img)
  unet_encoder(img,num_filters)
  
