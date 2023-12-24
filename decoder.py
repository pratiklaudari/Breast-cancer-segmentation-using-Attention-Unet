from keras import layers
import tensorflow as tf
import attentation

def decoder_block(inputs, skip_connection, num_filters, use_batch_norm=True):

    x = inputs

    # Upsampling (transposed convolution)
    x = layers.Conv2DTranspose(num_filters, kernel_size=2, strides=2, padding="same")(x)

    # Attention gate
    x = attentation.att(x, skip_connection)  # Apply attention mechanism

    # Concatenate with skip connection
    x = layers.concatenate([x, skip_connection])

    # Two convolutional layers with ReLU activation
    x = layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu")(x)

    # Optional batch normalization
    if use_batch_norm:
        x = layers.BatchNormalization()(x)

    return x

def unet_decoder(encoder_output, skip_connections, start_filters=64, num_levels=4):

    x = encoder_output

    for level in reversed(range(num_levels)):
        num_filters = start_filters // 2**level
        skip_connection = skip_connections[level]
        x = decoder_block(x, skip_connection, num_filters)

    # Final output layer (1x1 convolution)
    x = layers.Conv2D(1, kernel_size=1, activation="sigmoid")(x)  # Adjust for desired output channels

    return x