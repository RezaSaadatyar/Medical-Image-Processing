# ================================ Presented by: Reza Saadatyar (2024-2025) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation, concatenate, Conv2DTranspose
from tensorflow.keras.models import Model
import tensorflow as tf

def unet_model(
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
        model = unet_model(img_height, img_width, img_channels)
    """
    # Clear the TensorFlow session to free up memory and avoid conflicts
    # from previous model definitions
    tf.keras.backend.clear_session()
    
    # Input layer and normalization
    inputs = Input((img_height, img_width, img_channels))
    
    # Lambda layer to check max value and normalize only if > 1
    def normalize_inputs(x):
        max_val = tf.reduce_max(x)
        return tf.cond(tf.greater(max_val, 1.0), lambda: x / 255.0, lambda: x)

    inputs = Lambda(normalize_inputs, output_shape=(img_height, img_width, img_channels))(inputs)

    # ======================================= ENCODER (Contracting Path) =======================================
    # Block 1 (Shallow: Lowest Dropout)
    c1 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(inputs)
    if use_batchnorm: c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(c1)
    if use_batchnorm: c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    # Block 2 (Shallow: Low Dropout)
    c2 = Conv2D(base_filters * 2, kernel_size, kernel_initializer='he_normal', padding='same')(p1)
    if use_batchnorm: c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(base_filters * 2, kernel_size, kernel_initializer='he_normal', padding='same')(c2)
    if use_batchnorm: c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    if act_dropout: c2 = Dropout(dropout_rates['shallow'])(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Block 3 (Mid-Depth: Moderate Dropout)
    c3 = Conv2D(base_filters * 4, kernel_size, kernel_initializer='he_normal', padding='same')(p2)
    if use_batchnorm: c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(base_filters * 4, kernel_size, kernel_initializer='he_normal', padding='same')(c3)
    if use_batchnorm: c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    if act_dropout: c3 = Dropout(dropout_rates['mid'])(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    # Block 4 (Mid-Depth: Moderate Dropout)
    c4 = Conv2D(base_filters * 8, kernel_size, kernel_initializer='he_normal', padding='same')(p3)
    if use_batchnorm: c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(base_filters * 8, kernel_size, kernel_initializer='he_normal', padding='same')(c4)
    if use_batchnorm: c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    if act_dropout: c4 = Dropout(dropout_rates['mid'])(c4)
    p4 = MaxPooling2D((2, 2))(c4)

    # ======================================= BOTTLENECK (Deepest Layer) =======================================
    c5 = Conv2D(base_filters * 16, kernel_size, kernel_initializer='he_normal', padding='same')(p4)
    if use_batchnorm: c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(base_filters * 16, kernel_size, kernel_initializer='he_normal', padding='same')(c5)
    if use_batchnorm: c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    if act_dropout: c5 = Dropout(dropout_rates['deep'])(c5)  # Maximum dropout

    # ======================================== DECODER (Expansive Path) ========================================
    # Upsample and concatenate with corresponding encoder features
    up1 = Conv2DTranspose(base_filters * 8, transpose_kernel_size, strides=(2, 2), padding='same')(c5)
    concat1 = concatenate([c4, up1], axis=3)
    c6 = Conv2D(base_filters * 8, kernel_size, kernel_initializer='he_normal', padding='same')(concat1)
    if use_batchnorm: c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2D(base_filters * 8, kernel_size, kernel_initializer='he_normal', padding='same')(c6)
    if use_batchnorm: c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    if act_dropout: c6 = Dropout(dropout_rates['mid'])(c6)  # Matches Block 4

    up2 = Conv2DTranspose(base_filters * 4, transpose_kernel_size, strides=(2, 2), padding='same')(c6)
    concat2 = concatenate([c3, up2], axis=3)
    c7 = Conv2D(base_filters * 4, kernel_size, kernel_initializer='he_normal', padding='same')(concat2)
    if use_batchnorm: c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2D(base_filters * 4, kernel_size, kernel_initializer='he_normal', padding='same')(c7)
    if use_batchnorm: c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    if act_dropout: c7 = Dropout(dropout_rates['mid'])(c7)  # Matches Block 3

    up3 = Conv2DTranspose(base_filters * 2, transpose_kernel_size, strides=(2, 2), padding='same')(c7)
    concat3 = concatenate([c2, up3], axis=3)
    c8 = Conv2D(base_filters * 2, kernel_size, kernel_initializer='he_normal', padding='same')(concat3)
    if use_batchnorm: c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    c8 = Conv2D(base_filters * 2, kernel_size, kernel_initializer='he_normal', padding='same')(c8)
    if use_batchnorm: c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    if act_dropout: c8 = Dropout(dropout_rates['shallow'])(c8)  # Matches Block 2

    up4 = Conv2DTranspose(base_filters, transpose_kernel_size, strides=(2, 2), padding='same')(c8)
    concat4 = concatenate([c1, up4], axis=3)
    c9 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(concat4)
    if use_batchnorm: c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)
    c9 = Conv2D(base_filters, kernel_size, kernel_initializer='he_normal', padding='same')(c9)
    if use_batchnorm: c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)
    # No dropout in final decoder block (shallowest layer)

    # Output layer
    if n_class < 1: raise ValueError("Number of classes must be at least 1")
    act = 'sigmoid' if n_class == 1 else 'softmax'
    c10 = Conv2D(n_class, (1, 1), activation=act)(c9)  # 1x1 conv for output

    model = Model(inputs=inputs, outputs=c10)
    return model


# def unet_model(
#     img_height: int,
#     img_width: int,
#     img_channels: int,
#     base_filters: int = 16,
#     kernel_size: tuple = (3, 3),
#     transpose_kernel_size: tuple = (2, 2),
#     act_dropout: bool = False,
#     use_batchnorm: bool = False,
#     dropout_rates: dict = {'shallow':0.1, 'mid':0.2, 'deep':0.3},
#     n_class: int = 1):
#     """
#     Args:
#         img_height: Height of input images
#         img_width: Width of input images
#         img_channels: Number of input channels
#         base_filters: Base number of filters for the first layer (default: 16)
#         kernel_size: Size of the convolution kernels (default: (3, 3))
#         transpose_kernel_size: Size of the transpose convolution kernels (default: (2, 2))
#         act_dropout: Global toggle for dropout (default: False)
#         use_batchnorm: Whether to use BatchNormalization (default: False)
#         dropout_rates: Dictionary defining dropout rates for different depths
#         n_class: Number of output classes (default: 1 for binary segmentation)

#     Returns:
#         A compiled Keras Model instance
#     """
#     # Clear the TensorFlow session to free up memory and avoid conflicts
#     # from previous model definitions
#     tf.keras.backend.clear_session()
    
#     # Input processing
#     inputs = Input((img_height, img_width, img_channels))
#     inputs = Lambda(lambda x: x / 255.0 if tf.reduce_max(x) > 1.0 else x, 
#                    output_shape=(img_height, img_width, img_channels))(inputs)

#     # Encoder blocks
#     def encoder_block(x, filters, dropout_rate=None):
#         x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
#         if use_batchnorm: x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
#         if use_batchnorm: x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         if act_dropout and dropout_rate: x = Dropout(dropout_rate)(x)
#         return x, MaxPooling2D((2, 2))(x)

#     c1, p1 = encoder_block(inputs, base_filters)
#     c2, p2 = encoder_block(p1, base_filters * 2, dropout_rates['shallow'])
#     c3, p3 = encoder_block(p2, base_filters * 4, dropout_rates['mid'])
#     c4, p4 = encoder_block(p3, base_filters * 8, dropout_rates['mid'])

#     # Bottleneck
#     c5 = Conv2D(base_filters * 16, kernel_size, padding='same', kernel_initializer='he_normal')(p4)
#     if use_batchnorm: c5 = BatchNormalization()(c5)
#     c5 = Activation('relu')(c5)
#     c5 = Conv2D(base_filters * 16, kernel_size, padding='same', kernel_initializer='he_normal')(c5)
#     if use_batchnorm: c5 = BatchNormalization()(c5)
#     c5 = Activation('relu')(c5)
#     if act_dropout: c5 = Dropout(dropout_rates['deep'])(c5)

#     # Decoder blocks
#     def decoder_block(x, skip, filters, dropout_rate=None):
#         x = Conv2DTranspose(filters, transpose_kernel_size, strides=(2, 2), padding='same')(x)
#         x = concatenate([skip, x], axis=3)
#         x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
#         if use_batchnorm: x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         x = Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
#         if use_batchnorm: x = BatchNormalization()(x)
#         x = Activation('relu')(x)
#         if act_dropout and dropout_rate: x = Dropout(dropout_rate)(x)
#         return x

#     c6 = decoder_block(c5, c4, base_filters * 8, dropout_rates['mid'])
#     c7 = decoder_block(c6, c3, base_filters * 4, dropout_rates['mid'])
#     c8 = decoder_block(c7, c2, base_filters * 2, dropout_rates['shallow'])
#     c9 = decoder_block(c8, c1, base_filters)

#     # Output
#     if n_class < 1: raise ValueError("Number of classes must be at least 1")
#     outputs = Conv2D(n_class, (1, 1), activation='sigmoid' if n_class == 1 else 'softmax')(c9)

#     return Model(inputs=inputs, outputs=outputs)