from keras.layers import Input, Activation, Subtract
from keras.models import Model
from complexnn.conv import ComplexConv2D
from complexnn.bn_v2 import ComplexBatchNormalization
from act_ComplexNN import Complex2Channel, Channel2Complex, modReLU_Layer, zReLU_Layer


def CV_FCN(depth, filters, kernel_size, use_bn=False, mod=None, sub_connect=False):
    # complex to channel
    layer_count = 0
    inputs_1 = Input(shape=(256, 256, 1), dtype="complex64", name='Input_' + str(layer_count))
    layer_count += 1
    outputs_1 = Complex2Channel(name='Complex2Channel_' + str(layer_count))(inputs_1)

    # Input
    layer_count = 0
    inputs_2 = Input(shape=(256, 256, 2), name='Input_' + str(layer_count))

    # the first two parameters of ComplexConv2D are the number of filters and kernel size.
    # They are set based on a specific design
    layer_count += 1
    outs = (ComplexConv2D(filters, kernel_size, strides=(1, 1), padding='same',
                          activation='linear', kernel_initializer='complex_independent',
                          name='ComplexConv_' + str(layer_count)))(inputs_2)
    outs = (Activation('relu', name='relu_' + str(layer_count)))(outs)  # CReLU activation function
    # outs = Channel2Complex()(outs)
    # outs = zReLU_Layer()(outs)  # zReLU or modReLU
    # outs = Complex2Channel()(outs)

    # depth-2 layers of ComplexConv2D + ComplexBN + RELU
    for i in range(depth - 2):
        layer_count += 1
        if mod == 'add':
            filters = filters * 2
        elif mod == 'sub':
            filters = filters // 2
        outs = (ComplexConv2D(filters, kernel_size, strides=(1, 1), padding='same',
                              activation='linear', kernel_initializer='complex_independent',
                              name='ComplexConv_' + str(layer_count)))(outs)
        if use_bn:
            outs = (ComplexBatchNormalization(axis=-1, name='ComplexBN_' + str(layer_count)))(outs)
        outs = (Activation('relu', name='relu_' + str(layer_count)))(outs)
        # outs = Channel2Complex()(outs)
        # outs = zReLU_Layer()(outs)  # zReLU or modReLU
        # outs = Complex2Channel()(outs)

    # last conv layer
    layer_count += 1
    outs = (ComplexConv2D(1, kernel_size, strides=(1, 1), padding='same',
                          activation='linear', kernel_initializer='complex_independent',
                          name='ComplexConv_' + str(layer_count)))(outs)

    # subtract
    if sub_connect == True:
        layer_count += 1
        outputs_2 = Subtract(name='subtract_' + str(layer_count))([inputs_2, outs])
    else:
        outputs_2 = outs

    # channel to complex
    layer_count = 0
    inputs_3 = Input(shape=(256, 256, 2), name='Input_' + str(layer_count))
    layer_count += 1
    outputs_3 = Channel2Complex(name='Channel2Complex_' + str(layer_count))(inputs_3)

    # Model
    sig = Input((256,256,1))
    model_1 = Model(inputs=inputs_1, outputs=outputs_1, name='model_1')
    model_2 = Model(inputs=inputs_2, outputs=outputs_2, name='model_2')
    model_3 = Model(inputs=inputs_3, outputs=outputs_3, name='model_3')

    sig_channels = model_1(sig)
    sig_pred = model_2(sig_channels)
    sig_result = model_3(sig_pred)

    model_big = Model(inputs=sig, outputs=sig_result, name='model_big')
    return model_2, model_big


if __name__ == '__main__':
    for i in range(4,7):
        [model_small,model_big] = CV_FCN(depth=i, filters=16, kernel_size=3, use_bn=False,
                                         mod='fix', sub_connect=True)
        model_small.summary()
