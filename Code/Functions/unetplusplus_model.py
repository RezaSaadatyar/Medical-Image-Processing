# ================================ Presented by: Reza Saadatyar (2024-2025) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf

def unetplusplus_model(
    img_height: int,
    img_width: int,
    img_channels: int,
    base_filters: int = 16,
    kernel_size: tuple = (3, 3),
    transpose_kernel_size: tuple = (2, 2),
    act_dropout: bool = False,
    use_batchnorm: bool = False,
    dropout_rates: dict = {'shallow':0.1, 'mid':0.2, 'deep':0.3},
    n_class: int = 1):
    """
    Args:
        img_height: Height of input images
        img_width: Width of input images
        img_channels: Number of input channels
        base_filters: Base number of filters for the first layer (default: 16)
        kernel_size: Size of the convolution kernels (default: (3, 3))
        transpose_kernel_size: Size of the transpose convolution kernels (default: (2, 2))
        act_dropout: Global toggle for dropout (default: False)
        use_batchnorm: Whether to use BatchNormalization (default: False)
        dropout_rates: Dictionary defining dropout rates for different depths
        n_class: Number of output classes (default: 1 for binary segmentation)

    Returns:
        A compiled Keras Model instance

    Example:
        img_height, img_width, img_channels = list(train_dataset.element_spec[0].shape)
        model = unetplusplus_model(img_height, img_width, img_channels)
    """
    # Clear the TensorFlow session to free up memory and avoid conflicts
    tf.keras.backend.clear_session()

    # Input layer and normalization
    inputs = Input((img_height, img_width, img_channels))

    def normalize_inputs(x):
        max_val = tf.reduce_max(x)
        return tf.cond(tf.greater(max_val, 1.0), lambda: x / 255.0, lambda: x)

    inputs = Lambda(normalize_inputs, output_shape=(img_height, img_width, img_channels))(inputs)

    # Encoder: Level 1
    c1 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(inputs)
    if use_batchnorm: c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    if act_dropout: c1 = Dropout(dropout_rates['shallow'])(c1)
    c1 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(c1)
    if use_batchnorm: c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    if act_dropout: c1 = Dropout(dropout_rates['shallow'])(c1)
    p1 = MaxPooling2D((2, 2), strides=(2, 2))(c1)

    # Encoder: Level 2
    c2 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(p1)
    if use_batchnorm: c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    if act_dropout: c2 = Dropout(dropout_rates['mid'])(c2)
    c2 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(c2)
    if use_batchnorm: c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    if act_dropout: c2 = Dropout(dropout_rates['mid'])(c2)
    p2 = MaxPooling2D((2, 2), strides=(2, 2))(c2)

    # Decoder: Level 1_2
    up1 = Conv2DTranspose(base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(c2)
    d1_2 = concatenate([up1, c1], axis=3)
    d1_2 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d1_2)
    if use_batchnorm: d1_2 = BatchNormalization()(d1_2)
    d1_2 = Activation('relu')(d1_2)
    if act_dropout: d1_2 = Dropout(dropout_rates['shallow'])(d1_2)
    d1_2 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d1_2)
    if use_batchnorm: d1_2 = BatchNormalization()(d1_2)
    d1_2 = Activation('relu')(d1_2)
    if act_dropout: d1_2 = Dropout(dropout_rates['shallow'])(d1_2)

    # Encoder: Level 3
    c3 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(p2)
    if use_batchnorm: c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    if act_dropout: c3 = Dropout(dropout_rates['mid'])(c3)
    c3 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(c3)
    if use_batchnorm: c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    if act_dropout: c3 = Dropout(dropout_rates['mid'])(c3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(c3)

    # Decoder: Level 2_2
    up2 = Conv2DTranspose(2 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(c3)
    d2_2 = concatenate([up2, c2], axis=3)
    d2_2 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d2_2)
    if use_batchnorm: d2_2 = BatchNormalization()(d2_2)
    d2_2 = Activation('relu')(d2_2)
    if act_dropout: d2_2 = Dropout(dropout_rates['mid'])(d2_2)
    d2_2 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d2_2)
    if use_batchnorm: d2_2 = BatchNormalization()(d2_2)
    d2_2 = Activation('relu')(d2_2)
    if act_dropout: d2_2 = Dropout(dropout_rates['mid'])(d2_2)

    # Decoder: Level 1_3
    up1_3 = Conv2DTranspose(base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(d2_2)
    d1_3 = concatenate([up1_3, c1, d1_2], axis=3)
    d1_3 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d1_3)
    if use_batchnorm: d1_3 = BatchNormalization()(d1_3)
    d1_3 = Activation('relu')(d1_3)
    if act_dropout: d1_3 = Dropout(dropout_rates['shallow'])(d1_3)
    d1_3 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d1_3)
    if use_batchnorm: d1_3 = BatchNormalization()(d1_3)
    d1_3 = Activation('relu')(d1_3)
    if act_dropout: d1_3 = Dropout(dropout_rates['shallow'])(d1_3)

    # Encoder: Level 4
    c4 = Conv2D(8 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(pool3)
    if use_batchnorm: c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    if act_dropout: c4 = Dropout(dropout_rates['deep'])(c4)
    c4 = Conv2D(8 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(c4)
    if use_batchnorm: c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    if act_dropout: c4 = Dropout(dropout_rates['deep'])(c4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(c4)

    # Decoder: Level 3_2
    up3_2 = Conv2DTranspose(4 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(c4)
    d3_2 = concatenate([up3_2, c3], axis=3)
    d3_2 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d3_2)
    if use_batchnorm: d3_2 = BatchNormalization()(d3_2)
    d3_2 = Activation('relu')(d3_2)
    if act_dropout: d3_2 = Dropout(dropout_rates['mid'])(d3_2)
    d3_2 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d3_2)
    if use_batchnorm: d3_2 = BatchNormalization()(d3_2)
    d3_2 = Activation('relu')(d3_2)
    if act_dropout: d3_2 = Dropout(dropout_rates['mid'])(d3_2)

    # Decoder: Level 2_3
    up2_3 = Conv2DTranspose(2 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(d3_2)
    d2_3 = concatenate([up2_3, c2, d2_2], axis=3)
    d2_3 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d2_3)
    if use_batchnorm: d2_3 = BatchNormalization()(d2_3)
    d2_3 = Activation('relu')(d2_3)
    if act_dropout: d2_3 = Dropout(dropout_rates['mid'])(d2_3)
    d2_3 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d2_3)
    if use_batchnorm: d2_3 = BatchNormalization()(d2_3)
    d2_3 = Activation('relu')(d2_3)
    if act_dropout: d2_3 = Dropout(dropout_rates['mid'])(d2_3)

    # Decoder: Level 1_4
    up1_4 = Conv2DTranspose(base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(d2_3)
    d1_4 = concatenate([up1_4, c1, d1_2, d1_3], axis=3)
    d1_4 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d1_4)
    if use_batchnorm: d1_4 = BatchNormalization()(d1_4)
    d1_4 = Activation('relu')(d1_4)
    if act_dropout: d1_4 = Dropout(dropout_rates['shallow'])(d1_4)
    d1_4 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d1_4)
    if use_batchnorm: d1_4 = BatchNormalization()(d1_4)
    d1_4 = Activation('relu')(d1_4)
    if act_dropout: d1_4 = Dropout(dropout_rates['shallow'])(d1_4)

    # Encoder: Level 5
    c5 = Conv2D(16 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(pool4)
    if use_batchnorm: c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    if act_dropout: c5 = Dropout(dropout_rates['deep'])(c5)
    c5 = Conv2D(16 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(c5)
    if use_batchnorm: c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    if act_dropout: c5 = Dropout(dropout_rates['deep'])(c5)

    # Decoder: Level 4_2
    up4_2 = Conv2DTranspose(8 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(c5)
    d4_2 = concatenate([up4_2, c4], axis=3)
    d4_2 = Conv2D(8 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d4_2)
    if use_batchnorm: d4_2 = BatchNormalization()(d4_2)
    d4_2 = Activation('relu')(d4_2)
    if act_dropout: d4_2 = Dropout(dropout_rates['deep'])(d4_2)
    d4_2 = Conv2D(8 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d4_2)
    if use_batchnorm: d4_2 = BatchNormalization()(d4_2)
    d4_2 = Activation('relu')(d4_2)
    if act_dropout: d4_2 = Dropout(dropout_rates['deep'])(d4_2)

    # Decoder: Level 3_3
    up3_3 = Conv2DTranspose(4 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(d4_2)
    d3_3 = concatenate([up3_3, c3, d3_2], axis=3)
    d3_3 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d3_3)
    if use_batchnorm: d3_3 = BatchNormalization()(d3_3)
    d3_3 = Activation('relu')(d3_3)
    if act_dropout: d3_3 = Dropout(dropout_rates['mid'])(d3_3)
    d3_3 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d3_3)
    if use_batchnorm: d3_3 = BatchNormalization()(d3_3)
    d3_3 = Activation('relu')(d3_3)
    if act_dropout: d3_3 = Dropout(dropout_rates['mid'])(d3_3)

    # Decoder: Level 2_4
    up2_4 = Conv2DTranspose(2 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(d3_3)
    d2_4 = concatenate([up2_4, c2, d2_2, d2_3], axis=3)
    d2_4 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d2_4)
    if use_batchnorm: d2_4 = BatchNormalization()(d2_4)
    d2_4 = Activation('relu')(d2_4)
    if act_dropout: d2_4 = Dropout(dropout_rates['mid'])(d2_4)
    d2_4 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d2_4)
    if use_batchnorm: d2_4 = BatchNormalization()(d2_4)
    d2_4 = Activation('relu')(d2_4)
    if act_dropout: d2_4 = Dropout(dropout_rates['mid'])(d2_4)

    # Decoder: Level 1_5 (selected as the "best" feature map)
    up1_5 = Conv2DTranspose(base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(d2_4)
    d1_5 = concatenate([up1_5, c1, d1_2, d1_3, d1_4], axis=3)
    d1_5 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d1_5)
    if use_batchnorm: d1_5 = BatchNormalization()(d1_5)
    d1_5 = Activation('relu')(d1_5)
    if act_dropout: d1_5 = Dropout(dropout_rates['shallow'])(d1_5)
    d1_5 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(d1_5)
    if use_batchnorm: d1_5 = BatchNormalization()(d1_5)
    d1_5 = Activation('relu')(d1_5)
    if act_dropout: d1_5 = Dropout(dropout_rates['shallow'])(d1_5)

    # Output layer using d1_5 as the best feature map
    if n_class < 1: raise ValueError("Number of classes must be at least 1")
    act = 'sigmoid' if n_class == 1 else 'softmax'
    output = Conv2D(n_class, (1, 1), activation=act, kernel_initializer='he_normal', padding='same')(d1_5)

    model = Model(inputs=inputs, outputs=output)

    return model