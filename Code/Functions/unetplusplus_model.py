
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
    base_filters: int = 16,  # Added base_filters parameter
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

    # Lambda layer to check max value and normalize only if > 1
    def normalize_inputs(x):
        max_val = tf.reduce_max(x)
        return tf.cond(tf.greater(max_val, 1.0), lambda: x / 255.0, lambda: x)

    inputs = Lambda(normalize_inputs, output_shape=(img_height, img_width, img_channels))(inputs)

    c1 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(inputs)
    if use_batchnorm: c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    if act_dropout: c1 = Dropout(dropout_rates['shallow'])(c1)
    c1 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(c1)
    if use_batchnorm: c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    if act_dropout: c1 = Dropout(dropout_rates['shallow'])(c1)
    p1 = MaxPooling2D((2, 2), strides=(2, 2))(c1)

    c2 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(p1)
    if use_batchnorm: c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    if act_dropout: c2 = Dropout(dropout_rates['mid'])(c2)
    c2 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(c2)
    if use_batchnorm: c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    if act_dropout: c2 = Dropout(dropout_rates['mid'])(c2)
    p2 = MaxPooling2D((2, 2), strides=(2, 2))(c2)

    up1_2 = Conv2DTranspose(base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(c2)
    conv1_2 = concatenate([up1_2, c1], axis=3)
    c3 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv1_2)
    if use_batchnorm: c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    if act_dropout: c3 = Dropout(dropout_rates['shallow'])(c3)  # Shallow layer
    c3 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(c3)
    if use_batchnorm: c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    if act_dropout: c3 = Dropout(dropout_rates['shallow'])(c3)  # Shallow layer

    conv3_1 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(p2)
    if use_batchnorm: conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)
    if act_dropout: conv3_1 = Dropout(dropout_rates['mid'])(conv3_1)  # Mid layer
    conv3_1 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv3_1)
    if use_batchnorm: conv3_1 = BatchNormalization()(conv3_1)
    conv3_1 = Activation('relu')(conv3_1)
    if act_dropout: conv3_1 = Dropout(dropout_rates['mid'])(conv3_1)  # Mid layer
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_1)

    up2_2 = Conv2DTranspose(2 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, c2], axis=3)
    conv2_2 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv2_2)
    if use_batchnorm: conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)
    if act_dropout: conv2_2 = Dropout(dropout_rates['mid'])(conv2_2)  # Mid layer
    conv2_2 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv2_2)
    if use_batchnorm: conv2_2 = BatchNormalization()(conv2_2)
    conv2_2 = Activation('relu')(conv2_2)
    if act_dropout: conv2_2 = Dropout(dropout_rates['mid'])(conv2_2)  # Mid layer

    up1_3 = Conv2DTranspose(base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, c1, c3], axis=3)
    conv1_3 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv1_3)
    if use_batchnorm: conv1_3 = BatchNormalization()(conv1_3)
    conv1_3 = Activation('relu')(conv1_3)
    if act_dropout: conv1_3 = Dropout(dropout_rates['shallow'])(conv1_3)  # Shallow layer
    conv1_3 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv1_3)
    if use_batchnorm: conv1_3 = BatchNormalization()(conv1_3)
    conv1_3 = Activation('relu')(conv1_3)
    if act_dropout: conv1_3 = Dropout(dropout_rates['shallow'])(conv1_3)  # Shallow layer

    conv4_1 = Conv2D(8 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(pool3)
    if use_batchnorm: conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = Activation('relu')(conv4_1)
    if act_dropout: conv4_1 = Dropout(dropout_rates['deep'])(conv4_1)  # Deep layer
    conv4_1 = Conv2D(8 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv4_1)
    if use_batchnorm: conv4_1 = BatchNormalization()(conv4_1)
    conv4_1 = Activation('relu')(conv4_1)
    if act_dropout: conv4_1 = Dropout(dropout_rates['deep'])(conv4_1)  # Deep layer
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_1)

    up3_2 = Conv2DTranspose(4 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], axis=3)
    conv3_2 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv3_2)
    if use_batchnorm: conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)
    if act_dropout: conv3_2 = Dropout(dropout_rates['mid'])(conv3_2)  # Mid layer
    conv3_2 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv3_2)
    if use_batchnorm: conv3_2 = BatchNormalization()(conv3_2)
    conv3_2 = Activation('relu')(conv3_2)
    if act_dropout: conv3_2 = Dropout(dropout_rates['mid'])(conv3_2)  # Mid layer

    up2_3 = Conv2DTranspose(2 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, c2, conv2_2], axis=3)
    conv2_3 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv2_3)
    if use_batchnorm: conv2_3 = BatchNormalization()(conv2_3)
    conv2_3 = Activation('relu')(conv2_3)
    if act_dropout: conv2_3 = Dropout(dropout_rates['mid'])(conv2_3)  # Mid layer
    conv2_3 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv2_3)
    if use_batchnorm: conv2_3 = BatchNormalization()(conv2_3)
    conv2_3 = Activation('relu')(conv2_3)
    if act_dropout: conv2_3 = Dropout(dropout_rates['mid'])(conv2_3)  # Mid layer

    up1_4 = Conv2DTranspose(base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, c1, c3, conv1_3], axis=3)
    conv1_4 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv1_4)
    if use_batchnorm: conv1_4 = BatchNormalization()(conv1_4)
    conv1_4 = Activation('relu')(conv1_4)
    if act_dropout: conv1_4 = Dropout(dropout_rates['shallow'])(conv1_4)  # Shallow layer
    conv1_4 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv1_4)
    if use_batchnorm: conv1_4 = BatchNormalization()(conv1_4)
    conv1_4 = Activation('relu')(conv1_4)
    if act_dropout: conv1_4 = Dropout(dropout_rates['shallow'])(conv1_4)  # Shallow layer

    conv5_1 = Conv2D(16 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(pool4)
    if use_batchnorm: conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = Activation('relu')(conv5_1)
    if act_dropout: conv5_1 = Dropout(dropout_rates['deep'])(conv5_1)  # Deep layer
    conv5_1 = Conv2D(16 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv5_1)
    if use_batchnorm: conv5_1 = BatchNormalization()(conv5_1)
    conv5_1 = Activation('relu')(conv5_1)
    if act_dropout: conv5_1 = Dropout(dropout_rates['deep'])(conv5_1)  # Deep layer

    up4_2 = Conv2DTranspose(8 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], axis=3)
    conv4_2 = Conv2D(8 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv4_2)
    if use_batchnorm: conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)
    if act_dropout: conv4_2 = Dropout(dropout_rates['deep'])(conv4_2)  # Deep layer
    conv4_2 = Conv2D(8 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv4_2)
    if use_batchnorm: conv4_2 = BatchNormalization()(conv4_2)
    conv4_2 = Activation('relu')(conv4_2)
    if act_dropout: conv4_2 = Dropout(dropout_rates['deep'])(conv4_2)  # Deep layer

    up3_3 = Conv2DTranspose(4 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], axis=3)
    conv3_3 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv3_3)
    if use_batchnorm: conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)
    if act_dropout: conv3_3 = Dropout(dropout_rates['mid'])(conv3_3)  # Mid layer
    conv3_3 = Conv2D(4 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv3_3)
    if use_batchnorm: conv3_3 = BatchNormalization()(conv3_3)
    conv3_3 = Activation('relu')(conv3_3)
    if act_dropout: conv3_3 = Dropout(dropout_rates['mid'])(conv3_3)  # Mid layer

    up2_4 = Conv2DTranspose(2 * base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, c2, conv2_2, conv2_3], axis=3)
    conv2_4 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv2_4)
    if use_batchnorm: conv2_4 = BatchNormalization()(conv2_4)
    conv2_4 = Activation('relu')(conv2_4)
    if act_dropout: conv2_4 = Dropout(dropout_rates['mid'])(conv2_4)  # Mid layer
    conv2_4 = Conv2D(2 * base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv2_4)
    if use_batchnorm: conv2_4 = BatchNormalization()(conv2_4)
    conv2_4 = Activation('relu')(conv2_4)
    if act_dropout: conv2_4 = Dropout(dropout_rates['mid'])(conv2_4)  # Mid layer

    up1_5 = Conv2DTranspose(base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, c1, c3, conv1_3, conv1_4], axis=3)
    conv1_5 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv1_5)
    if use_batchnorm: conv1_5 = BatchNormalization()(conv1_5)
    conv1_5 = Activation('relu')(conv1_5)
    if act_dropout: conv1_5 = Dropout(dropout_rates['shallow'])(conv1_5)  # Shallow layer
    conv1_5 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(conv1_5)
    if use_batchnorm: conv1_5 = BatchNormalization()(conv1_5)
    conv1_5 = Activation('relu')(conv1_5)
    if act_dropout: conv1_5 = Dropout(dropout_rates['shallow'])(conv1_5)  # Shallow layer

    # Output layer
    if n_class < 1: raise ValueError("Number of classes must be at least 1")
    act = 'sigmoid' if n_class == 1 else 'softmax'
    c10 = Conv2D(n_class, (1, 1), activation=act, kernel_initializer='he_normal', padding='same')(conv1_5)  # 1x1 conv for output

    model = Model(inputs=inputs, outputs=c10)

    return model