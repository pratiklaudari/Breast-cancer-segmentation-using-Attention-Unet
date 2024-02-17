from keras import layers
import tensorflow as tf
import attentation

def decoder_block(inputs, skip_connection, num_filters):

    X = inputs
    # Upsampling (transposed convolution)
    x = layers.UpSampling2D()(X)
    # Concatenate with attentation
    x = layers.concatenate([x, skip_connection])

    # Two convolutional layers with ReLU activation
    x = layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",strides=1, kernel_initializer='he_normal')(x)
    x = layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu",strides=1, kernel_initializer='he_normal')(x)
    print(x.shape)
    return x