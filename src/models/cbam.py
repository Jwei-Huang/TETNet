import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape,
    Add, Activation, Conv2D, Concatenate, Multiply, Lambda
)

def channel_attention(input_feature, ratio=8):
    """
    CBAM Channel Attention (channels_last recommended).
    Safe for model serialization: no K.* usage.
    """
    data_format = tf.keras.backend.image_data_format()
    channel_axis = 1 if data_format == "channels_first" else -1
    channel = int(input_feature.shape[channel_axis])

    # shared MLP
    shared_layer_one = Dense(
        max(channel // ratio, 1),
        kernel_initializer="he_normal",
        activation="relu",
        use_bias=True,
        bias_initializer="zeros",
    )
    shared_layer_two = Dense(
        channel,
        kernel_initializer="he_normal",
        use_bias=True,
        bias_initializer="zeros",
    )

    # avg pooling
    avg_pool = GlobalAveragePooling2D(data_format=data_format)(input_feature)
    if data_format == "channels_first":
        # (B, C) -> (B, C, 1, 1)
        avg_pool = Reshape((channel, 1, 1))(avg_pool)
    else:
        # (B, C) -> (B, 1, 1, C)
        avg_pool = Reshape((1, 1, channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    # max pooling
    max_pool = GlobalMaxPooling2D(data_format=data_format)(input_feature)
    if data_format == "channels_first":
        max_pool = Reshape((channel, 1, 1))(max_pool)
    else:
        max_pool = Reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation("sigmoid")(cbam_feature)

    return Multiply()([input_feature, cbam_feature])


def spatial_attention(input_feature, kernel_size=7):
    """
    CBAM Spatial Attention.
    Replaces K.mean/K.max with tf.reduce_mean/tf.reduce_max.
    """
    data_format = tf.keras.backend.image_data_format()

    if data_format == "channels_first":
        # input: (B, C, H, W)
        # mean/max over channel axis=1 -> (B, 1, H, W)
        cbam_mean = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(input_feature)
        cbam_max  = Lambda(lambda x: tf.reduce_max(x,  axis=1, keepdims=True))(input_feature)
        cbam_feature = Concatenate(axis=1)([cbam_mean, cbam_max])

        # Conv2D expects channels_last by default; so set data_format explicitly
        cbam_feature = Conv2D(
            filters=1,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
            data_format="channels_first",
        )(cbam_feature)

    else:
        # input: (B, H, W, C)
        # mean/max over channel axis=3 -> (B, H, W, 1)
        cbam_mean = Lambda(lambda x: tf.reduce_mean(x, axis=3, keepdims=True))(input_feature)
        cbam_max  = Lambda(lambda x: tf.reduce_max(x,  axis=3, keepdims=True))(input_feature)
        cbam_feature = Concatenate(axis=3)([cbam_mean, cbam_max])

        cbam_feature = Conv2D(
            filters=1,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
            data_format="channels_last",
        )(cbam_feature)

    return Multiply()([input_feature, cbam_feature])


def cbam_block(cbam_feature, ratio=8, kernel_size=7):
    cbam_feature = channel_attention(cbam_feature, ratio=ratio)
    cbam_feature = spatial_attention(cbam_feature, kernel_size=kernel_size)
    return cbam_feature
