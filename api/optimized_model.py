# Optimized model for colorectal tumour detection - real time
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPool2D, UpSampling2D, Concatenate, Activation, Input, BatchNormalization

def conv_block(x, num_filters):
    x = DepthwiseConv2D((3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def build_model():
    size = 128
    num_filters = [32, 64, 128, 256]
    inputs = Input((size, size, 3))

    skip_x = []
    x = inputs

    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()

    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    return Model(inputs, x)
