# ================================ Presented by: Reza Saadatyar (2024-2025) ====================================
# =================================== E-mail: Reza.Saadatyar@outlook.com =======================================

from tensorflow.keras.layers import Input, Lambda, Conv2D, BatchNormalization, Activation, concatenate, Add, UpSampling2D   
from tensorflow.keras.models import Model
import tensorflow as tf

def resunet_model(
    img_height: int,
    img_width: int,
    img_channels: int,
    base_filters: int = 16,
    strides: tuple = 1,
    kernel_size: tuple = (3, 3),
    shortcut_kernel_size: tuple = (1, 1),
    use_batchnorm: bool = True,
    n_class: int = 1):
    """
    Create a ResUNet model with residual connections.

    Args:
        img_height: Height of input images
        img_width: Width of input images
        img_channels: Number of input channels
        base_filters: Number of filters in first conv layer
        stride: Stride for main convolutions
        kernel_size: Size of main convolution kernels
        shortcut_stride: Stride for shortcut convolutions
        shortcut_kernel_size: Size of shortcut convolution kernels
        transpose_kernel_size: Size of transpose convolution kernels
        use_batchnorm: Whether to use batch normalization
        dropout_rates: Dictionary of dropout rates for different depths
        n_class: Number of output classes

    Returns:
        A Keras Model instance
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
    c1 = Conv2D(base_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(inputs)
    if use_batchnorm: c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(base_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(c1)

    shortcut = Conv2D(base_filters, kernel_size=shortcut_kernel_size, strides=strides, padding='same', kernel_initializer='he_normal')(inputs)
    if use_batchnorm: shortcut = BatchNormalization()(shortcut)
    out1 = Add()([c1, shortcut])

    # Block 2
    if use_batchnorm: out1 = BatchNormalization()(out1)
    res1 = Activation('relu')(out1)
    res1 = Conv2D(base_filters * 2, kernel_size=kernel_size, padding='same', strides=2 * strides)(res1)
    if use_batchnorm: res1 = BatchNormalization()(res1)
    res1 = Activation('relu')(res1)
    res1 = Conv2D(base_filters * 2, kernel_size=kernel_size, padding='same', strides=1)(res1)
    shortcut1 = Conv2D(base_filters * 2, kernel_size=shortcut_kernel_size, padding='same', strides=2 * strides)(out1)
    if use_batchnorm: shortcut1 = BatchNormalization()(shortcut1)
    out2 = Add()([shortcut1, res1])

    # Block 3
    if use_batchnorm: out2 = BatchNormalization()(out2)
    res2 = Activation('relu')(out2)
    res2 = Conv2D(base_filters * 4, kernel_size=kernel_size, padding='same', strides=2 * strides)(res2)
    if use_batchnorm: res2 = BatchNormalization()(res2)
    res2 = Activation('relu')(res2)
    res2 = Conv2D(base_filters * 4, kernel_size=kernel_size, padding='same', strides=1)(res2)
    shortcut2 = Conv2D(base_filters * 4, kernel_size=shortcut_kernel_size, padding='same', strides=2 * strides)(out2)
    if use_batchnorm: shortcut2 = BatchNormalization()(shortcut2)
    out3 = Add()([shortcut2, res2])

    # Block 4
    if use_batchnorm: out3 = BatchNormalization()(out3)
    res3 = Activation('relu')(out3)
    res3 = Conv2D(base_filters * 8, kernel_size=kernel_size, padding='same', strides=2 * strides)(res3)
    if use_batchnorm: res3 = BatchNormalization()(res3)
    res3 = Activation('relu')(res3)
    res3 = Conv2D(base_filters * 8, kernel_size=kernel_size, padding='same', strides=1)(res3)
    shortcut3 = Conv2D(base_filters * 8, kernel_size=shortcut_kernel_size, padding='same', strides=2 * strides)(out3)
    if use_batchnorm: shortcut3 = BatchNormalization()(shortcut3)
    out4 = Add()([shortcut3, res3])

    # Block 5
    if use_batchnorm: out4 = BatchNormalization()(out4)
    res4 = Activation('relu')(out4)
    res4 = Conv2D(base_filters * 16, kernel_size=kernel_size, padding='same', strides=2 * strides)(res4)
    if use_batchnorm: res4 = BatchNormalization()(res4)
    res4 = Activation('relu')(res4)
    res4 = Conv2D(base_filters * 16, kernel_size=kernel_size, padding='same', strides=1)(res4)
    shortcut4 = Conv2D(base_filters*16, kernel_size=shortcut_kernel_size, padding='same', strides=2 * strides)(out4)
    if use_batchnorm: shortcut4 = BatchNormalization()(shortcut4)
    out5 = Add()([shortcut4, res4])

    # ========================================== BOTTLENECK (Bridge) ===========================================
    if use_batchnorm: out5 = BatchNormalization()(out5)
    conv = Activation('relu')(out5)
    conv = Conv2D(base_filters * 16, kernel_size=kernel_size, padding='same', strides=strides)(conv)
    if use_batchnorm: conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(base_filters * 16, kernel_size=kernel_size, padding='same', strides=strides)(conv)

    # ======================================== DECODER (Expansive Path) ========================================
    # Upsample and concatenate with corresponding encoder features
    uconv1 = UpSampling2D((2, 2))(conv)
    uconv1 = concatenate([uconv1, out4])

    if use_batchnorm: uconv1 = BatchNormalization()(uconv1)
    uconv11 = Activation('relu')(uconv1)
    uconv11 = Conv2D(base_filters * 16, kernel_size=kernel_size, padding='same', strides=strides)(uconv11)
    if use_batchnorm: uconv11 = BatchNormalization()(uconv11)
    uconv11 = Activation('relu')(uconv11)
    uconv11 = Conv2D(base_filters * 16, kernel_size=kernel_size, padding='same', strides=1)(uconv11)
    shortcut5 = Conv2D(base_filters * 16, kernel_size=shortcut_kernel_size, padding='same', strides=strides)(uconv1)
    if use_batchnorm: shortcut5 = BatchNormalization()(shortcut5)
    out6 = Add()([uconv11, shortcut5])

    uconv2 = UpSampling2D((2, 2))(out6)
    uconv2 = concatenate([uconv2, out3])

    if use_batchnorm: uconv2 = BatchNormalization()(uconv2)
    uconv22 = Activation('relu')(uconv2)
    uconv22 = Conv2D(base_filters*8, kernel_size=kernel_size, padding='same', strides=strides)(uconv22)
    if use_batchnorm: uconv22 = BatchNormalization()(uconv22)
    uconv22 = Activation('relu')(uconv22)
    uconv22 = Conv2D(base_filters*8, kernel_size=kernel_size, padding='same', strides=1)(uconv22)
    shortcut6 = Conv2D(base_filters*8, kernel_size=shortcut_kernel_size, padding='same', strides=strides)(uconv2)
    if use_batchnorm: shortcut6 = BatchNormalization()(shortcut6)
    out7 = Add()([uconv22, shortcut6])

    uconv3 = UpSampling2D((2, 2))(out7)
    uconv3 = concatenate([uconv3, out2])

    if use_batchnorm: uconv3 = BatchNormalization()(uconv3)
    uconv33 = Activation('relu')(uconv3)
    uconv33 = Conv2D(base_filters*4, kernel_size=kernel_size, padding='same', strides=strides)(uconv33)
    if use_batchnorm: uconv33 = BatchNormalization()(uconv33)
    uconv33 = Activation('relu')(uconv33)
    uconv33 = Conv2D(base_filters*4, kernel_size=kernel_size, padding='same', strides=1)(uconv33)
    shortcut7 = Conv2D(base_filters*4, kernel_size=shortcut_kernel_size, padding='same', strides=strides)(uconv3)
    if use_batchnorm: shortcut7 = BatchNormalization()(shortcut7)
    out8 = Add()([uconv33, shortcut7])

    uconv4 = UpSampling2D((2, 2))(out8)
    uconv4 = concatenate([uconv4, out1])

    if use_batchnorm: uconv4 = BatchNormalization()(uconv4)
    uconv44 = Activation('relu')(uconv4)
    uconv44 = Conv2D(base_filters*2, kernel_size=kernel_size, padding='same', strides=strides)(uconv44)
    if use_batchnorm: uconv44 = BatchNormalization()(uconv44)
    uconv44 = Activation('relu')(uconv44)
    uconv44 = Conv2D(base_filters*2, kernel_size=kernel_size, padding='same', strides=1)(uconv44)
    shortcut8 = Conv2D(base_filters*2, kernel_size=shortcut_kernel_size, padding='same', strides=strides)(uconv4)
    if use_batchnorm: shortcut8 = BatchNormalization()(shortcut8)
    out9 = Add()([uconv44, shortcut8])

    # Output layer
    if n_class < 1: raise ValueError("Number of classes must be at least 1")
    act = 'sigmoid' if n_class == 1 else 'softmax'
    output_layer = Conv2D(n_class, (1, 1), padding="same", activation=act)(out9)

    model = Model(inputs, output_layer)

    return model