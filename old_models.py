from keras.models import Model
from keras.layers.core import Dense, Activation, Permute, Reshape
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.layers.convolutional import SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import SpatialDropout2D, MaxPooling2D
from keras.regularizers import l1_l2
from keras.layers import Input, Flatten, Dropout
from keras.layers import DepthwiseConv2D
from keras import backend as K
from keras.constraints import max_norm
import numpy as np
import utils

def EEGNet(nb_classes, Chans=64, Samples=128, D=2,
           dropoutRate=0.5, kernLength=64, numFilters=8, norm_rate=0.25):
    """ Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024v3)

    Requires Tensorflow >= 1.5 and Keras >= 2.1.3

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version.

    Note that we use 'image_data_format' = 'channels_first' in there keras.json
    configuration file.

    Inputs:

        nb_classes: int, number of classes to classify
        Chans, Samples: number of channels and time points in the EEG data
        regRate: regularization parameter for L1 and L2 penalties
        dropoutRate: dropout fraction
        kernLength: length of temporal convolution in first layer
        numFilters: number of temporal-spatial filter pairs to learn
        D: number of spatial filters to learn within each temporal convolution. Default: D = 2

    Depending on the task, using numFilters = 4 or 8 seemed to do pretty well
    across tasks.
    """
    input1 = Input(shape=(1, Chans, Samples))
    # input1 = Permute((2, 1))(input_original)
    # input1 = Reshape(target_shape=(1, Chans, Samples))(input_original)
    layer1 = Conv2D(numFilters, (1, kernLength), padding='same',
                    input_shape=(1, Chans, Samples),
                    use_bias=False,
                    data_format='channels_first')(input1)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = DepthwiseConv2D((Chans, 1),
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.),
                             use_bias=False,
                             data_format='channels_first')(layer1)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = Activation('elu')(layer1)
    layer1 = AveragePooling2D((1, 4), data_format='channels_first')(layer1)
    layer1 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer1)

    layer2 = SeparableConv2D(numFilters * D, (1, 16),
                             use_bias=False, padding='same',
                             data_format='channels_first')(layer1)
    layer2 = BatchNormalization(axis=1)(layer2)
    layer2 = Activation('elu')(layer2)
    layer2 = AveragePooling2D((1, 8), data_format='channels_first')(layer2)
    layer2 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer2)

    flatten = Flatten(name='flatten')(layer2)

    dense = Dense(nb_classes, name='dense',kernel_constraint = max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)

def EEGNet_output(nb_classes, Chans=64, Samples=128, D=2, dropoutRate=0.5,
                  kernLength=64, numFilters=8, norm_rate=0.25, x_input=None):
    """ Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024v3)

    Requires Tensorflow >= 1.5 and Keras >= 2.1.3

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version.

    Note that we use 'image_data_format' = 'channels_first' in there keras.json
    configuration file.

    Inputs:

        nb_classes: int, number of classes to classify
        Chans, Samples: number of channels and time points in the EEG data
        regRate: regularization parameter for L1 and L2 penalties
        dropoutRate: dropout fraction
        kernLength: length of temporal convolution in first layer
        numFilters: number of temporal-spatial filter pairs to learn
        D: number of spatial filters to learn within each temporal convolution. Default: D = 2

    Depending on the task, using numFilters = 4 or 8 seemed to do pretty well
    across tasks.
    """
    input1 = x_input
    # input1 = Input(shape=(1, Chans, Samples))
    # input1 = Permute((2, 1))(input_original)
    # input1 = Reshape(target_shape=(1, Chans, Samples))(input_original)
    layer1 = Conv2D(numFilters, (1, kernLength), padding='same',
                    input_shape=(1, Chans, Samples),
                    use_bias=False,
                    data_format='channels_first')(input1)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = DepthwiseConv2D((Chans, 1),
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.),
                             use_bias=False,
                             data_format='channels_first')(layer1)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = Activation('elu')(layer1)
    layer1 = AveragePooling2D((1, 4), data_format='channels_first')(layer1)
    layer1 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer1)

    layer2 = SeparableConv2D(numFilters * D, (1, 16),
                             use_bias=False, padding='same',
                             data_format='channels_first')(layer1)
    layer2 = BatchNormalization(axis=1)(layer2)
    layer2 = Activation('elu')(layer2)
    layer2 = AveragePooling2D((1, 8), data_format='channels_first')(layer2)
    layer2 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer2)

    flatten = Flatten(name='flatten')(layer2)

    dense = Dense(nb_classes, name='dense',kernel_constraint = max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return softmax


def EEGNet_old(nb_classes, Chans=64, Samples=128, regRate=0.0001,
           dropoutRate=0.25, kernLength=64, numFilters=8):
    """ Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024v3)

    Requires Tensorflow >= 1.5 and Keras >= 2.1.3

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version.

    Note that we use 'image_data_format' = 'channels_first' in there keras.json
    configuration file.

    Inputs:

        nb_classes: int, number of classes to classify
        Chans, Samples: number of channels and time points in the EEG data
        regRate: regularization parameter for L1 and L2 penalties
        dropoutRate: dropout fraction
        kernLength: length of temporal convolution in first layer
        numFilters: number of temporal-spatial filter pairs to learn

    Depending on the task, using numFilters = 4 or 8 seemed to do pretty well
    across tasks.
    """
    input1 = Input(shape=(1, Chans, Samples))
    # input1 = Permute((2, 1))(input_original)
    # input1 = Reshape(target_shape=(1, Chans, Samples))(input_original)
    layer1 = Conv2D(numFilters, (1, kernLength), padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=0.0),
                    input_shape=(1, Chans, Samples),
                    use_bias=False,
                    data_format='channels_first')(input1)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = DepthwiseConv2D((Chans, 1),
                             depthwise_regularizer=l1_l2(l1=regRate, l2=regRate),
                             use_bias=False,
                             data_format='channels_first')(layer1)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = Activation('elu')(layer1)
    layer1 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer1)

    layer2 = SeparableConv2D(numFilters, (1, 8),
                             depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
                             use_bias=False, padding='same',
                             data_format='channels_first')(layer1)
    layer2 = BatchNormalization(axis=1)(layer2)
    layer2 = Activation('elu')(layer2)
    layer2 = AveragePooling2D((1, 4), data_format='channels_first')(layer2)
    layer2 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer2)

    layer3 = SeparableConv2D(numFilters * 2, (1, 8), depth_multiplier=2,
                             depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
                             use_bias=False, padding='same',
                             data_format='channels_first')(layer2)
    layer3 = BatchNormalization(axis=1)(layer3)
    layer3 = Activation('elu')(layer3)
    layer3 = AveragePooling2D((1, 4), data_format='channels_first')(layer3)
    layer3 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer3)

    flatten = Flatten(name='flatten')(layer3)

    dense = Dense(nb_classes, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)

def EEGNet_wide(nb_classes, Chans=64, Samples=128, regRate=0.0001,
           dropoutRate=0.25, kernLength=64, numFilters=32):
    """ Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024v3)

    Requires Tensorflow >= 1.5 and Keras >= 2.1.3

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version.

    Note that we use 'image_data_format' = 'channels_first' in there keras.json
    configuration file.

    Inputs:

        nb_classes: int, number of classes to classify
        Chans, Samples: number of channels and time points in the EEG data
        regRate: regularization parameter for L1 and L2 penalties
        dropoutRate: dropout fraction
        kernLength: length of temporal convolution in first layer
        numFilters: number of temporal-spatial filter pairs to learn

    Depending on the task, using numFilters = 4 or 8 seemed to do pretty well
    across tasks.
    """
    input1 = Input(shape=(1, Chans, Samples))
    # input1 = Permute((2, 1))(input_original)
    # input1 = Reshape(target_shape=(1, Chans, Samples))(input_original)
    layer1 = Conv2D(numFilters, (1, kernLength), padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=0.0),
                    input_shape=(1, Chans, Samples),
                    use_bias=False,
                    data_format='channels_first')(input1)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = DepthwiseConv2D((Chans, 1),
                             depthwise_regularizer=l1_l2(l1=regRate, l2=regRate),
                             use_bias=False,
                             data_format='channels_first')(layer1)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = Activation('elu')(layer1)
    layer1 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer1)

    layer2 = SeparableConv2D(numFilters, (1, 8),
                             depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
                             use_bias=False, padding='same',
                             data_format='channels_first')(layer1)
    layer2 = BatchNormalization(axis=1)(layer2)
    layer2 = Activation('elu')(layer2)
    layer2 = AveragePooling2D((1, 4), data_format='channels_first')(layer2)
    layer2 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer2)

    layer3 = SeparableConv2D(numFilters * 2, (1, 8), depth_multiplier=2,
                             depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
                             use_bias=False, padding='same',
                             data_format='channels_first')(layer2)
    layer3 = BatchNormalization(axis=1)(layer3)
    layer3 = Activation('elu')(layer3)
    layer3 = AveragePooling2D((1, 4), data_format='channels_first')(layer3)
    layer3 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer3)

    flatten = Flatten(name='flatten')(layer3)

    dense = Dense(nb_classes, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)

# def EEGNet_output(nb_classes, Chans=64, Samples=128, regRate=0.0001,
#            dropoutRate=0.25, kernLength=64, numFilters=8, x_input=None):
#     """ Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024v3)
#
#     Requires Tensorflow >= 1.5 and Keras >= 2.1.3
#
#     Note that this implements the newest version of EEGNet and NOT the earlier
#     version (version v1 and v2 on arxiv). We strongly recommend using this
#     architecture as it performs much better and has nicer properties than
#     our earlier version.
#
#     Note that we use 'image_data_format' = 'channels_first' in there keras.json
#     configuration file.
#
#     Inputs:
#
#         nb_classes: int, number of classes to classify
#         Chans, Samples: number of channels and time points in the EEG data
#         regRate: regularization parameter for L1 and L2 penalties
#         dropoutRate: dropout fraction
#         kernLength: length of temporal convolution in first layer
#         numFilters: number of temporal-spatial filter pairs to learn
#
#     Depending on the task, using numFilters = 4 or 8 seemed to do pretty well
#     across tasks.
#     """
#     input1 = x_input
#     # input1 = Input(shape=(1, Chans, Samples))
#     # input1 = Permute((2, 1))(input_original)
#     # input1 = Reshape(target_shape=(1, Chans, Samples))(input_original)
#     layer1 = Conv2D(numFilters, (1, kernLength), padding='same',
#                     kernel_regularizer=l1_l2(l1=0.0, l2=0.0),
#                     input_shape=(1, Chans, Samples),
#                     use_bias=False,
#                     data_format='channels_first')(input1)
#     layer1 = BatchNormalization(axis=1)(layer1)
#     layer1 = DepthwiseConv2D((Chans, 1),
#                              depthwise_regularizer=l1_l2(l1=regRate, l2=regRate),
#                              use_bias=False,
#                              data_format='channels_first')(layer1)
#     layer1 = BatchNormalization(axis=1)(layer1)
#     layer1 = Activation('elu')(layer1)
#     layer1 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer1)
#
#     layer2 = SeparableConv2D(numFilters, (1, 8),
#                              depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
#                              use_bias=False, padding='same',
#                              data_format='channels_first')(layer1)
#     layer2 = BatchNormalization(axis=1)(layer2)
#     layer2 = Activation('elu')(layer2)
#     layer2 = AveragePooling2D((1, 4), data_format='channels_first')(layer2)
#     layer2 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer2)
#
#     layer3 = SeparableConv2D(numFilters * 2, (1, 8), depth_multiplier=2,
#                              depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
#                              use_bias=False, padding='same',
#                              data_format='channels_first')(layer2)
#     layer3 = BatchNormalization(axis=1)(layer3)
#     layer3 = Activation('elu')(layer3)
#     layer3 = AveragePooling2D((1, 4), data_format='channels_first')(layer3)
#     layer3 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer3)
#
#     flatten = Flatten(name='flatten')(layer3)
#
#     dense = Dense(nb_classes, name='dense')(flatten)
#     softmax = Activation('softmax', name='softmax')(dense)
#
#     return softmax


def EEGNet_binary(nb_classes, Chans=64, Samples=128, regRate=0.0001,
           dropoutRate=0.25, kernLength=64, numFilters=8):
    """ Keras Implementation of EEGNet (https://arxiv.org/abs/1611.08024v3)

    Requires Tensorflow >= 1.5 and Keras >= 2.1.3

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version.

    Note that we use 'image_data_format' = 'channels_first' in there keras.json
    configuration file.

    Inputs:

        nb_classes: int, number of classes to classify
        Chans, Samples: number of channels and time points in the EEG data
        regRate: regularization parameter for L1 and L2 penalties
        dropoutRate: dropout fraction
        kernLength: length of temporal convolution in first layer
        numFilters: number of temporal-spatial filter pairs to learn

    Depending on the task, using numFilters = 4 or 8 seemed to do pretty well
    across tasks.
    """
    input1 = Input(shape=(1, Chans, Samples))
    # input1 = Permute((2, 1))(input_original)
    # input1 = Reshape(target_shape=(1, Chans, Samples))(input_original)
    layer1 = Conv2D(numFilters, (1, kernLength), padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=0.0),
                    input_shape=(1, Chans, Samples),
                    use_bias=False,
                    data_format='channels_first')(input1)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = DepthwiseConv2D((Chans, 1),
                             depthwise_regularizer=l1_l2(l1=regRate, l2=regRate),
                             use_bias=False,
                             data_format='channels_first')(layer1)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = Activation('elu')(layer1)
    layer1 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer1)

    layer2 = SeparableConv2D(numFilters, (1, 8),
                             depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
                             use_bias=False, padding='same',
                             data_format='channels_first')(layer1)
    layer2 = BatchNormalization(axis=1)(layer2)
    layer2 = Activation('elu')(layer2)
    layer2 = AveragePooling2D((1, 4), data_format='channels_first')(layer2)
    layer2 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer2)

    layer3 = SeparableConv2D(numFilters * 2, (1, 8), depth_multiplier=2,
                             depthwise_regularizer=l1_l2(l1=0.0, l2=regRate),
                             use_bias=False, padding='same',
                             data_format='channels_first')(layer2)
    layer3 = BatchNormalization(axis=1)(layer3)
    layer3 = Activation('elu')(layer3)
    layer3 = AveragePooling2D((1, 4), data_format='channels_first')(layer3)
    layer3 = SpatialDropout2D(dropoutRate, data_format='channels_first')(layer3)

    flatten = Flatten(name='flatten')(layer3)

    dense = Dense(1, name='dense')(flatten)
    tanh = Activation('tanh', name='tanh')(dense)

    return Model(inputs=input1, outputs=tanh)


# def EEGNet_old(nb_classes, Chans=64, Samples=128, regRate=0.0001,
#                dropoutRate=0.25, kernels=[(2, 32), (8, 4)], strides=(2, 4)):
#     """ Keras Implementation of EEGNet_v1 (https://arxiv.org/abs/1611.08024v2)
#
#     This model is the original EEGNet model proposed on arxiv
#             https://arxiv.org/abs/1611.08024v2
#
#     with a few modifications: we use striding instead of max-pooling as this
#     helped slightly in classification performance while also providing a
#     computational speed-up.
#
#     Note that we no longer recommend the use of this architecture, as the new
#     version of EEGNet performs much better overall and has nicer properties.
#
#     Inputs:
#
#         nb_classes     : total number of final categories
#         Chans, Samples : number of EEG channels and samples, respectively
#         regRate        : regularization rate for L1 and L2 regularizations
#         dropoutRate    : dropout fraction
#         kernels        : the 2nd and 3rd layer kernel dimensions (default is
#                          the [2, 32] x [8, 4] configuration)
#         strides        : the stride size (note that this replaces the max-pool
#                          used in the original paper)
#
#     """
#
#     # start the model
#     input_main = Input((1, Chans, Samples))
#     layer1 = Conv2D(16, (Chans, 1), input_shape=(1, Chans, Samples),
#                     kernel_regularizer=l1_l2(l1=regRate, l2=regRate))(input_main)
#     layer1 = BatchNormalization(axis=1)(layer1)
#     layer1 = Activation('elu')(layer1)
#     layer1 = Dropout(dropoutRate)(layer1)
#
#     permute_dims = 2, 1, 3
#     permute1 = Permute(permute_dims)(layer1)
#
#     layer2 = Conv2D(4, kernels[0], padding='same',
#                     kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
#                     strides=strides)(permute1)
#     layer2 = BatchNormalization(axis=1)(layer2)
#     layer2 = Activation('elu')(layer2)
#     layer2 = Dropout(dropoutRate)(layer2)
#
#     layer3 = Conv2D(4, kernels[1], padding='same',
#                     kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
#                     strides=strides)(layer2)
#     layer3 = BatchNormalization(axis=1)(layer3)
#     layer3 = Activation('elu')(layer3)
#     layer3 = Dropout(dropoutRate)(layer3)
#
#     flatten = Flatten(name='flatten')(layer3)
#
#     dense = Dense(nb_classes, name='dense')(flatten)
#     softmax = Activation('softmax', name='softmax')(dense)
#
#     return Model(inputs=input_main, outputs=softmax)

def EEGNet_old(nb_classes, Chans=64, Samples=128, regRate=0.0001,
               dropoutRate=0.25, kernels=[(2, 32), (8, 4)], strides=(2, 4)):
    """ Keras Implementation of EEGNet_v1 (https://arxiv.org/abs/1611.08024v2)

    This model is the original EEGNet model proposed on arxiv
            https://arxiv.org/abs/1611.08024v2

    with a few modifications: we use striding instead of max-pooling as this
    helped slightly in classification performance while also providing a
    computational speed-up.

    Note that we no longer recommend the use of this architecture, as the new
    version of EEGNet performs much better overall and has nicer properties.

    Inputs:

        nb_classes     : total number of final categories
        Chans, Samples : number of EEG channels and samples, respectively
        regRate        : regularization rate for L1 and L2 regularizations
        dropoutRate    : dropout fraction
        kernels        : the 2nd and 3rd layer kernel dimensions (default is
                         the [2, 32] x [8, 4] configuration)
        strides        : the stride size (note that this replaces the max-pool
                         used in the original paper)

    """

    # start the model
    input_main = Input((1, Chans, Samples))
    layer1 = Conv2D(16, (Chans, 1), input_shape=(1, Chans, Samples),
                    kernel_regularizer=l1_l2(l1=regRate, l2=regRate))(input_main)
    layer1 = BatchNormalization(axis=1)(layer1)
    layer1 = Activation('elu')(layer1)
    layer1 = Dropout(dropoutRate)(layer1)

    permute_dims = 2, 1, 3
    permute1 = Permute(permute_dims)(layer1)

    layer2 = Conv2D(4, kernels[0], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                    strides=strides)(permute1)
    layer2 = BatchNormalization(axis=1)(layer2)
    layer2 = Activation('elu')(layer2)
    layer2 = Dropout(dropoutRate)(layer2)

    layer3 = Conv2D(4, kernels[1], padding='same',
                    kernel_regularizer=l1_l2(l1=0.0, l2=regRate),
                    strides=strides)(layer2)
    layer3 = BatchNormalization(axis=1)(layer3)
    layer3 = Activation('elu')(layer3)
    layer3 = Dropout(dropoutRate)(layer3)

    flatten = Flatten(name='flatten')(layer3)

    dense = Dense(nb_classes, name='dense')(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)



# def DeepConvNet(nb_classes, Chans=64, Samples=128, filter_num1=25,
#              filter_num2=50, filter_num3=100, filter_num4=200, sample_kernel=10):
#     input_original = Input(shape=(Samples, Chans))
#     input1 = Reshape(target_shape=(Samples, Chans, 1))(input_original)
#
#     # Conv Pool Block1
#     conv1 = Conv2D(filters=filter_num1, kernel_size=(sample_kernel, 1),
#                    strides=(1, 1), padding='valid',
#                    activation='linear', use_bias=True)(input1)  # (Samples, Chans, filters)
#     conv1_spatial = Conv2D(filters=filter_num1, kernel_size=(1, Chans),
#                            strides=(1, 1), padding='valid', activation='elu',
#                            use_bias=True)(conv1)   # (Samples, 1, filters)
#     layer1_perm = Permute((3, 1, 2))(conv1_spatial)   # (filters, Samples, 1)
#     pool1 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same')(layer1_perm) # (filters, Samples, 1)
#
#     # Conv Pool Block2
#     conv2 = Conv2D(filters=filter_num2, kernel_size=(filter_num1, sample_kernel), strides=(1, 1),
#                    padding='valid', activation='elu',
#                    use_bias=True)(pool1)    # (1, Samples, filters)
#     layer2_perm = Permute((3, 2, 1))(conv2) # (filters, Samples, 1)
#     pool2 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same')(layer2_perm)
#
#     # Conv Pool Block3
#     conv3 = Conv2D(filters=filter_num3, kernel_size=(filter_num2, sample_kernel), strides=(1, 1),
#                    padding='valid', activation='elu',
#                    use_bias=True)(pool2)    # (1, Samples, filters)
#     layer3_perm = Permute((3, 2, 1))(conv3) # (filters, Samples, 1)
#     pool3 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same')(layer3_perm)
#
#     # # Conv Pool Block4
#     # conv4 = Conv2D(filters=filter_num4, kernel_size=(filter_num3, sample_kernel), strides=(1, 1),
#     #                padding='valid', activation='elu',
#     #                use_bias=True)(pool3)    # (1, Samples, filters)
#     # layer4_perm = Permute((3, 2, 1))(conv4) # (filters, Samples, 1)
#     # pool4 = MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same')(layer4_perm)
#
#     # Dense
#     flatten = Flatten()(pool3)
#
#     dense = Dense(nb_classes, activation='softmax', name='dense')(flatten)
#     softmax = Activation('softmax', name='softmax')(dense)
#
#     return Model(inputs=input_original, outputs=softmax)

def DeepConvNet(nb_classes, Chans=64, Samples=256,
                dropoutRate=0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.

    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10

    Note that this implementation has not been verified by the original
    authors.

    """

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(25, (1, 5),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def DeepConvNet_wide(nb_classes, Chans=64, Samples=256,
                dropoutRate=0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.

    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10

    Note that this implementation has not been verified by the original
    authors.

    """

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(50, (1, 5),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(50, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(400, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)

def DeepConvNet_output(nb_classes, Chans=64, Samples=256, dropoutRate=0.5, x_input=None):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.

    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10

    Note that this implementation has not been verified by the original
    authors.

    """

    # start the model
    input_main = x_input
    # input_main = Input((1, Chans, Samples))
    block1 = Conv2D(25, (1, 5),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return softmax


def DeepConvNet_output_wide(nb_classes, Chans=64, Samples=256, dropoutRate=0.5, x_input=None):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.

    This implementation assumes the input is a 2-second EEG signal sampled at
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10

    Note that this implementation has not been verified by the original
    authors.

    """

    # start the model
    input_main = x_input
    # input_main = Input((1, Chans, Samples))
    block1 = Conv2D(50, (1, 5),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(50, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(400, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return softmax

# def ShallowConvNet(nb_classes, Chans=64, Samples=128, filter_num1=25):
#     input_original = Input(shape=(Samples, Chans))
#     input1 = Reshape(target_shape=(Samples, Chans, 1))(input_original)
#
#     # Conv Pool Block
#     conv = Conv2D(filters=filter_num1, kernel_size=(25, 1),
#                    strides=(1, 1), padding='valid',
#                    activation='linear', use_bias=True)(input1)  # (Samples, Chans, filters)
#     conv_spatial = Conv2D(filters=filter_num1, kernel_size=(1, Chans),
#                            strides=(1, 1), padding='valid', activation='linear',
#                            use_bias=True)(conv)  # (Samples, 1, filters)
#     # TODO: Square
#
#     layer_perm = Permute((3, 1, 2))(conv_spatial)  # (filters, Samples, 1)
#     pool = AveragePooling2D(pool_size=(1, 75), strides=(1, 75))(layer_perm)  # (filters, Samples, 1)
#
#     # Dense
#     flatten = Flatten()(pool)
#     dense = Dense(nb_classes, name='dense')(flatten)
#     softmax = Activation('softmax', name='softmax')(dense)
#
#     return Model(inputs=input_original, outputs=softmax)

# need these for ShallowConvNet
def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(40, (1, 13),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def ShallowConvNet_wide(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, numFilters=80):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """

    # start the model
    input_main = Input((1, Chans, Samples))
    block1 = Conv2D(numFilters, (1, 13),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(numFilters, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax)


def ShallowConvNet_output(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, x_input=None):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """

    # start the model
    # input_main = Input((1, Chans, Samples))
    input_main = x_input
    block1 = Conv2D(40, (1, 13),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(40, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return softmax

def ShallowConvNet_output_wide(nb_classes, Chans=64, Samples=128, dropoutRate=0.5, x_input=None):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.

    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.

    Note that we use the max_norm constraint on all convolutional layers, as
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication
    with the original authors.

                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25

    Note that this implementation has not been verified by the original
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations.
    """

    # start the model
    # input_main = Input((1, Chans, Samples))
    input_main = x_input
    block1 = Conv2D(80, (1, 13),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(80, (Chans, 1), use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return softmax


def eval(model, x, y):
    y_pred = np.argmax(model.predict(x), axis=1)
    y_test = np.squeeze(y)
    bca = utils.bca(y_test, y_pred)
    acc = np.sum(y_pred == y_test).astype(np.float32) / len(y_pred)

    return acc, bca