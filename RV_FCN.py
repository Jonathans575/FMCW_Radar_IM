from keras.layers import Input, Conv2D, Activation, BatchNormalization, Subtract
from keras.models import Model


def RV_FCN(depth, filters, kernel_size, use_bn=False, mod=None, sub_connect=False):
    # Input
    layer_count = 0
    inputs = Input(shape=(256, 256, 2), name='Input_' + str(layer_count))

    # the first two parameters of Conv2D are the number of filters and kernel size.
    # They are set based on a specific design
    layer_count += 1
    outs = (Conv2D(filters, kernel_size, strides=(1, 1), padding='same',
                   name='Conv_' + str(layer_count)))(inputs)
    outs = (Activation('relu', name='relu_' + str(layer_count)))(outs)

    # depth-2 layers of Conv2D + BN + RELU
    for i in range(depth - 2):
        layer_count += 1
        if mod == 'add':
            filters = filters * 2
        elif mod == 'sub':
            filters = filters // 2
        outs = (Conv2D(filters, kernel_size, strides=(1, 1), padding='same',
                       name='Conv_' + str(layer_count)))(outs)
        if use_bn:
            outs = (BatchNormalization(axis=-1, name='BN_' + str(layer_count)))(outs)
        outs = (Activation('relu', name='relu_' + str(layer_count)))(outs)

    # last conv layer
    layer_count += 1
    outs = (Conv2D(2, kernel_size, strides=(1, 1), padding='same',
                   name='Conv_' + str(layer_count)))(outs)

    # subtract
    if sub_connect == True:
        layer_count += 1
        outputs = Subtract(name='subtract_' + str(layer_count))([inputs, outs])
    else:
        outputs= outs

    # Model
    model = Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    for i in range(4,11):
        model = RV_FCN(depth=i, filters=32, kernel_size=3, use_bn=False,
                       mod='fix', sub_connect=False)
        model.summary()
