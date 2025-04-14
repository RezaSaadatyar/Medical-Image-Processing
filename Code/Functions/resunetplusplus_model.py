# ================================ Presented by: Reza Saadatyar (2024-2025) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, Activation, concatenate, Add, UpSampling2D
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Multiply, MaxPooling2D
from tensorflow.keras.models import Model
import tensorflow as tf

# Squeeze and Excitation Block
# This block applies channel-wise attention to feature maps
def squeeze_excite_block(inputs, ratio=8):
    # Get the number of filters/channels from the input tensor
    filters = inputs.shape[-1]
    # Define the shape for the sequence (1x1xFilters)
    seq_shape = (1, 1, filters)

    # Apply global average pooling to reduce spatial dimensions
    seq = GlobalAveragePooling2D()(inputs)
    # Reshape the pooled output to match the sequence shape
    seq = Reshape(seq_shape)(seq)
    # First dense layer with ReLU activation, reducing dimensionality by ratio
    seq = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(seq)
    # Second dense layer with sigmoid activation, restoring original dimensionality
    seq = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(seq)

    # Multiply the input with the attention weights (seq) element-wise
    mult = Multiply()([inputs, seq])

    return mult

# ResNet Block with Skip Connection
# This block implements a residual block with squeeze-excite attention
def resnet_block(x, n_filter, kernel_size=(3, 3), shortcut_kernel_size=(1, 1), strides=1):
    x_init = x

    # First convolution block
    x = BatchNormalization()(x)   # Apply batch normalization to the input
    x = Activation("relu")(x)     # Apply ReLU activation
    x = Conv2D(n_filter, kernel_size=kernel_size, padding="same", strides=strides)(x)

    # Second convolution block
    x = BatchNormalization()(x)   # Apply batch normalization again
    x = Activation("relu")(x)     # Apply ReLU activation again
    x = Conv2D(n_filter, kernel_size=kernel_size, padding="same", strides=1)(x)

    # Skip connection (shortcut path)
    s  = Conv2D(n_filter, kernel_size=shortcut_kernel_size, padding="same", strides=strides)(x_init) # Apply 1x1 convolution to match dimensions
    s = BatchNormalization()(s) # Apply batch normalization to the shortcut

    # Combine main path and shortcut
    x = Add()([s, x])  # Add the main path and shortcut
    x = squeeze_excite_block(x) # Apply squeeze-excite attention mechanism

    return x

# Attention Block
# This block implements attention mechanism between encoder and decoder features
def attention_block(g, x, kernel_size=(3, 3)):
    """
        g: Output of Parallel Encoder block
        x: Output of Previous Decoder block
    """

    # Get number of filters from input tensor x
    filters = x.shape[-1]

    # Process g (encoder output) with batch norm, ReLU, and 3x3 conv
    g_conv = BatchNormalization()(g)  # Normalize g
    g_conv = Activation("relu")(g_conv)  # Apply ReLU activation
    g_conv = Conv2D(filters, kernel_size=kernel_size, padding="same")(g_conv)  # 3x3 conv to match filters

    # Downsample g_conv using max pooling
    g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)  # Reduce spatial dimensions

    # Process x (decoder output) with batch norm, ReLU, and 3x3 conv
    x_conv = BatchNormalization()(x)  # Normalize x
    x_conv = Activation("relu")(x_conv)  # Apply ReLU activation
    x_conv = Conv2D(filters, kernel_size=kernel_size, padding="same")(x_conv)  # 3x3 conv to match filters

    # Add pooled encoder features and processed decoder features
    gc_sum = Add()([g_pool, x_conv])  # Combine information from both paths

    # Process combined features with batch norm, ReLU, and 3x3 conv
    gc_conv = BatchNormalization()(gc_sum)  # Normalize combined features
    gc_conv = Activation("relu")(gc_conv)  # Apply ReLU activation
    gc_conv = Conv2D(filters, kernel_size=kernel_size, padding="same")(gc_conv)  # Final 3x3 conv

    # Multiply attention weights with original decoder features
    gc_mul = Multiply()([gc_conv, x])  # Apply attention mechanism

    return gc_mul  # Return attended features

# ResUNet++ Model
# This function creates the complete ResUNet++ architecture
def resunetplusplus_model(
    img_height: int,
    img_width: int,
    img_channels: int,
    base_filters: int = 16,
    strides: tuple = 1,
    kernel_size: tuple = (3, 3),
    shortcut_kernel_size: tuple = (1, 1),
    rate_scale: int = 1,
    n_class: int = 1,
    ):
    """
    Create a ResUNet model with residual connections.

    Args:
        img_height: Height of input images
        img_width: Width of input images
        img_channels: Number of input channels
        base_filters: Number of filters in first conv layer
        stride: Stride for main convolutions
        kernel_size: Size of main convolution kernels
        shortcut_kernel_size: Size of shortcut convolution kernels
        rate_scale: Scaling factor for dilation rates in ASPP module
        n_class: Number of output classes

    Returns:
        A Keras Model instance
    """
     # Input layer and normalization
    inputs = Input((img_height, img_width, img_channels))

    # Lambda layer to check max value and normalize only if > 1
    def normalize_inputs(x):
        max_val = tf.reduce_max(x)
        return tf.cond(tf.greater(max_val, 1.0), lambda: x / 255.0, lambda: x)

    inputs = Lambda(normalize_inputs, output_shape=(img_height, img_width, img_channels))(inputs)

    # Initial Convolution Block
    c1 = Conv2D(base_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(base_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(c1)

    shortcut = Conv2D(base_filters, kernel_size=shortcut_kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(inputs)
    shortcut = BatchNormalization()(shortcut)
    c1 = Add()([c1, shortcut])
    c1 = squeeze_excite_block(c1)

    # ================================================ Encoder =================================================
    # Encoder path with increasing filters and downsampling
    c2 = resnet_block(c1, 2 * base_filters, kernel_size, shortcut_kernel_size, strides=2 * strides)
    c3 = resnet_block(c2, 4 * base_filters, kernel_size, shortcut_kernel_size, strides=2 * strides)
    c4 = resnet_block(c3, 8 * base_filters, kernel_size, shortcut_kernel_size, strides=2 * strides)

    # ================================================= Bridge =================================================
    # Atrous Spatial Pyramid Pooling (ASPP) for multi-scale feature extraction
    x1 = Conv2D(16 * base_filters, kernel_size=kernel_size, dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(c4)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(16 * base_filters, kernel_size=kernel_size, dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(c4)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(16 * base_filters, kernel_size=kernel_size, dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(c4)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(16 * base_filters, kernel_size=kernel_size, padding="same")(c4)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    b1 = Conv2D(16 * base_filters, (1, 1), padding="same")(y)

    # ================================================ Decoder =================================================
    # Decoder path with attention blocks and upsampling
    d1 = attention_block(c3, b1)
    d1 = UpSampling2D((2, 2))(d1)
    d1 = concatenate([d1, c3])
    d1 = resnet_block(d1, 8 * base_filters)

    d2 = attention_block(c2, d1)
    d2 = UpSampling2D((2, 2))(d2)
    d2 = concatenate([d2, c2])
    d2 = resnet_block(d2, 4 * base_filters)

    d3 = attention_block(c1, d2)
    d3 = UpSampling2D((2, 2))(d3)
    d3 = concatenate([d3, c1])
    d3 = resnet_block(d3, 2 * base_filters)

    # =========================================== Output layer =================================================
    # Atrous Spatial Pyramid Pooling (ASPP)
    x1 = Conv2D(base_filters, kernel_size=kernel_size, dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(d3)
    x1 = BatchNormalization()(x1)

    x2 = Conv2D(base_filters, kernel_size=kernel_size, dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(d3)
    x2 = BatchNormalization()(x2)

    x3 = Conv2D(base_filters, kernel_size=kernel_size, dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(d3)
    x3 = BatchNormalization()(x3)

    x4 = Conv2D(base_filters, kernel_size=kernel_size, padding="same")(d3)
    x4 = BatchNormalization()(x4)

    y = Add()([x1, x2, x3, x4])
    outputs = Conv2D(base_filters, (1, 1), padding="same")(y)

    if n_class < 1: raise ValueError("Number of classes must be at least 1")
    act = 'sigmoid' if n_class == 1 else 'softmax'
    output_layer = Conv2D(n_class, (1, 1), padding="same", activation=act)(outputs)

    # Model
    model = Model(inputs, output_layer)

    return model