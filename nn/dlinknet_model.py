"""D-LinkNet-style encoder–decoder for binary road segmentation."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)


CUSTOM_OBJECTS = {
    "bce_dice_loss": bce_dice_loss,
    "dice_coefficient": dice_coefficient,
    "dice_loss": dice_loss,
}


def residual_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(num_filters, (1, 1), padding="same")(input_tensor)
    shortcut = BatchNormalization()(shortcut)
    out = Activation("relu")(Add()([shortcut, x]))
    return out


def dilated_center_block(input_tensor, num_filters):
    d1 = Activation("relu")(
        Conv2D(num_filters, 3, dilation_rate=1, padding="same")(input_tensor)
    )
    d2 = Activation("relu")(Conv2D(num_filters, 3, dilation_rate=2, padding="same")(d1))
    d4 = Activation("relu")(Conv2D(num_filters, 3, dilation_rate=4, padding="same")(d2))
    d8 = Activation("relu")(Conv2D(num_filters, 3, dilation_rate=8, padding="same")(d4))
    return Add()([input_tensor, d1, d2, d4, d8])


def decoder_block(input_tensor, num_filters):
    x = Conv2D(num_filters, (1, 1), padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2DTranspose(num_filters, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    return Activation("relu")(x)


def encoder_block(input_tensor, num_filters, num_res_blocks):
    encoded = residual_block(input_tensor, num_filters)
    for _ in range(num_res_blocks - 1):
        encoded = residual_block(encoded, num_filters)
    return encoded, MaxPooling2D((2, 2))(encoded)


def create_dlinknet(input_shape=(256, 256, 3)):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPooling2D((2, 2))(x)

    e1, p1 = encoder_block(x, 64, 3)
    e2, p2 = encoder_block(p1, 128, 4)
    e3, p3 = encoder_block(p2, 256, 6)
    e4, _ = encoder_block(p3, 512, 3)

    center = dilated_center_block(e4, 512)
    d1 = Add()([decoder_block(center, 256), e3])
    d2 = Add()([decoder_block(d1, 128), e2])
    d3 = Add()([decoder_block(d2, 64), e1])
    d4 = decoder_block(d3, 64)

    x = Conv2DTranspose(32, 3, padding="same")(d4)
    outputs = Conv2D(1, 1, activation="sigmoid")(x)
    return Model(inputs=inputs, outputs=outputs, name="dlinknet_road")
