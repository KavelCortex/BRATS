import numpy as np
import keras
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D, Add
from keras.optimizers import Adam
from keras.models import load_model
from unet.utils.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,
                                weighted_dice_coefficient_loss, weighted_dice_coefficient, get_label_dice_coefficient_function)
K.set_image_data_format("channels_first")

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def unet_model_3d(input_shape,
                  pool_size=(2, 2, 2),
                  n_labels=4,
                  initial_learning_rate=0.00001,
                  deconvolution=True,
                  n_segmentation_levels=3,
                  depth=4,
                  n_base_filters=32,
                  include_label_wise_coefficients=False,
                  metrics=dice_coefficient,
                  batch_normalization=True,
                  activation_name='sigmoid'):

    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    ################################### Network ########################################
    #### 1. Down leveling. Default with 2 conv layer and goes down 4 times (depth). ####
    for layer_depth in range(depth):
        layer1 = create_convolution_block(
            input_layer=current_layer,
            n_filters=n_base_filters * (2**layer_depth),
            batch_normalization=batch_normalization)
        layer2 = create_convolution_block(
            input_layer=layer1,
            n_filters=n_base_filters * (2**layer_depth)*2,
            batch_normalization=batch_normalization)

        # when down leveling, add a Max Pooling layer to shrink the size (as shown in red arrows)
        if layer_depth < depth - 1:  # 0 -> depth-1
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])

    #### 2. Reaches bottom layer. Continue without max pooling ####
        else:  # depth
            current_layer = layer2
            levels.append([layer1, layer2])

    #### 3. Up leveling. same configuration as step 1, but goes up. ####
    ## NEW: Segmentation Layer
    segmentation_layers = list()
    for layer_depth in range(depth - 2, -1, -1):  # depth-1 -> 0
        up_convolution = create_up_sampling_module(
            input_layer=current_layer,
            n_filters=current_layer._keras_shape[1])
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(
            n_filters=levels[layer_depth][1]._keras_shape[1],
            input_layer=concat,
            batch_normalization=batch_normalization)
        current_layer = create_convolution_block(
            n_filters=levels[layer_depth][1]._keras_shape[1],
            input_layer=current_layer,
            batch_normalization=batch_normalization,
            kernel=(1,1,1))
        if layer_depth < n_segmentation_levels:
            segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1))(current_layer))

    #### 4. final layers ####
    # final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)
    model = Model(inputs=inputs, outputs=activation_block)
    ####################################################################################

    # metrics configuration
    if not isinstance(metrics, list):
        metrics = [metrics]
    if include_label_wise_coefficients and n_labels > 1:
        label_wise_dice_metrics = [
            get_label_dice_coefficient_function(index)
            for index in range(n_labels)
        ]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    # Compile Model
    model.compile(
        optimizer=Adam(lr=initial_learning_rate),
        loss=dice_coefficient_loss,
        metrics=metrics)
    return model


def create_convolution_block(input_layer,
                             n_filters,
                             batch_normalization=False,
                             kernel=(3, 3, 3),
                             activation=None,
                             padding='same',
                             strides=(1, 1, 1),
                             instance_normalization=False):
    layer = Conv3D(
        n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
        except ImportError:
            raise ImportError(
                "need to install keras_contrib to use instance normalization.")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution

def get_up_convolution(n_filters,
                       pool_size,
                       kernel_size=(2, 2, 2),
                       strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(
            filters=n_filters, kernel_size=kernel_size, strides=strides)
    else:
        return UpSampling3D(size=pool_size)


def load_old_model(model_file):
    print("Loading pre-trained model")
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,
                      'weighted_dice_coefficient': weighted_dice_coefficient,
                      'weighted_dice_coefficient_loss': weighted_dice_coefficient_loss}
    try:
        from keras_contrib.layers import InstanceNormalization
        custom_objects["InstanceNormalization"] = InstanceNormalization
    except ImportError:
        pass
    try:
        return load_model(model_file, custom_objects=custom_objects)
    except ValueError as error:
        if 'InstanceNormalization' in str(error):
            raise ValueError(str(error) + "\n\nPlease install keras-contrib to use InstanceNormalization:\n"
                                          "'pip install git+https://www.github.com/keras-team/keras-contrib.git'")
        else:
            raise error
