"""
Authors: Xiaoqing Liu
Data: 11/18/2021
Title: Final - Building THE Neural network
Comments:
1. Implement Ideal, Butterworth, Gaussian Low Pass Filters
2. According to the paper "fast MRI: An Open Dataset and Benchmarks for Accelerated MRI"
   implement U-net for image reconstruction
3. Loading and Split dataset singlecoil_val fold from https://fastmri.org/dataset/
"""


import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization
from keras.layers import Activation, MaxPool2D, Concatenate
from keras.models import load_model

import numpy as np
import random
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

from scipy.fft import ifft2, fftshift
import os
import re
import h5py
import cv2
# import keras_unet
#
# from keras_unet.models import satellite_unet
# def u2():
#     model = satellite_unet(input_shape=(320, 320, 1))
#     model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='mae')
#     return model


IMAGE_WIDTH = 320
IMAGE_HEIGHT =320

def conv_block(input, num_filters):
    """ The purpose of this function is to implement the 3x3 convolution+normalize+ReLU
     in the architecture of  U-net model, according to the paper
    Inputs:
        -- input: the output from last layer
        -- num_filters: the number of filters, like 32,64,128, 256
    Output:
        -- x: the output after this block
    """
    x = Conv2D(num_filters, (3, 3), padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def encoder_block(input, num_filters):
    """ The purpose of this function is to implement 3x3 convolution+normalize+ReLU
    and 2x2 Max pooling in the architecture of  U-net model, according to the paper
    Inputs:
        -- input: the output from last layer
        -- num_filters: the number of filters, like 32,64,128, 256
    Output:
        -- x: the output after conv_block
        -- p: the output after 2x2 MaxPooling
    """
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p


def decoder_block(input, skip_features, num_filters):
    """ The purpose of this function is to implement upsampling, concatennate
    and 3x3 convolution+normalize+ReLU
    Inputs:
        -- input: the output from last layer
        -- skip_features: the output from related conv_block
        -- num_filters: the number of filters, like 32,64,128, 256
    Output:
        -- x: the output after decoder_block
    """
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    # x = UpSampling2D(interpolation='bilinear')(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet():
    """ The purpose of this function is to build the U-net model, according to the paper
    Output:
        -- model: the U-net model
    """
    input_shape = (320, 320, 1)
    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)
    s4, p4 = encoder_block(p3, 256)

    # x = Conv2D(256, (3, 3), padding="same", activation='relu')(p4)
    x = Conv2D(256, (3, 3), padding="same")(p4)
    x = BatchNormalization()(x)
    b1 = Activation("relu")(x)

    d1 = decoder_block(b1, s4, 128)
    d2 = decoder_block(d1, s3, 64)
    d3 = decoder_block(d2, s2, 32)
    d4 = decoder_block(d3, s1, 32)

    c1 = Conv2D(16, (1, 1), padding="same", activation='relu')(d4)

    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(c1)

    model = Model(inputs, outputs, name="U-Net")
    return model


# def train_predict_model(inputs, outputs, test_inputs):
#     model = build_unet()
#     # model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsoluteError(), metrics=['mae'])
#     model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=tf.keras.losses.MeanAbsoluteError(), metrics=['mae'])
#
#     """
#     Model checkpoint
#     """
#     checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_mri.h5', verbose=1, save_best_only=True)
#     callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
#                  tf.keras.callbacks.TensorBoard(log_dir='mri_logs')]
#     history = model.fit(x=np.array(inputs, np.float32), y=np.array(outputs, np.float32), batch_size=2, epochs=5, verbose=1, callbacks=callbacks,
#                         validation_split=0.1, shuffle=True)
#     ###########################################################
#     """
#     To check the performance of the U net
#     """
#     preds_train = model.predict(np.array(inputs, np.float32), verbose=1)
#     idx = random.randint(0, len(preds_train))
#     imshow(inputs[idx])
#     plt.show()
#     reconstruction_train_img = np.squeeze(preds_train[idx])
#     # reconstruction_img = (reconstruction_img - np.min(reconstruction_img)) / (np.max(reconstruction_img) - np.min(reconstruction_img))
#     imshow(np.log(reconstruction_train_img + 1e-10), cmap="gray")
#     plt.show()
#
#     preds_test = model.predict(np.array(test_inputs, np.float32), verbose=1)
#
#     idx = random.randint(0, len(preds_test))
#     imshow(test_inputs[idx])
#     plt.show()
#     reconstruction_img = np.squeeze(preds_test[idx])
#     # reconstruction_img = (reconstruction_img - np.min(reconstruction_img)) / (np.max(reconstruction_img) - np.min(reconstruction_img))
#     imshow(np.log(reconstruction_img + 1e-10), cmap="gray")
#     plt.show()
#     print('ok')

def train_model(inputs, outputs):
    """ The purpose of this function is to train the U-net model
    Inputs:
        -- inputs: modified images(nx320x320)
        -- outputs: ground truth images(nx320x320)
    Output:
        -- single_coil_Unet_model: save the model, this can be loaded to do the predict
    """
    model = build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=['mae', nmse_metrix, psnr_metrix, ssim_metrix])

    """
    Model checkpoint
    """
    checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_mri.h5', verbose=1, save_best_only=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
                 tf.keras.callbacks.TensorBoard(log_dir='mri_logs')]
    history = model.fit(x=np.array(inputs, np.float32), y=np.array(outputs, np.float32), batch_size=2, epochs=15, verbose=1, callbacks=callbacks,
                        validation_split=0.1, shuffle=True)

    plot_history(history)

    # save model to file
    # model.save("single_coil_Unet.h5")
    model.save("single_coil_Unet_model")
    ###########################################################
    """
    To check the performance of the U net
    """
    # preds_train = model.predict(np.array(inputs, np.float32), verbose=1)
    # idx = random.randint(0, len(preds_train))
    # imshow(inputs[idx])
    # plt.show()
    # reconstruction_train_img = np.squeeze(preds_train[idx])
    # # reconstruction_img = (reconstruction_img - np.min(reconstruction_img)) / (np.max(reconstruction_img) - np.min(reconstruction_img))
    # imshow(np.log(reconstruction_train_img + 1e-10), cmap="gray")
    # plt.show()

def plot_history(h):
    """ The purpose of this function is to display the history of loss, ssim,psnr, nmse of
    each epoch during the training
    Inputs:
        -- h: history of model
    Output:
        None
    """
    loss = h.history['loss']
    val_loss = h.history['val_loss']

    ssim = h.history['ssim_metrix']
    val_ssim = h.history['val_ssim_metrix']

    psnr = h.history['psnr_metrix']
    val_psnr = h.history['val_psnr_metrix']

    nmse = h.history['nmse_metrix']
    val_nmse = h.history['val_nmse_metrix']

    epochs = range(len(loss))

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.figure()

    plt.plot(epochs, ssim, 'r', label='Training ssim')
    plt.plot(epochs, val_ssim, 'b', label='Validation ssim')
    plt.title('Training and validation')
    plt.legend(loc=0)
    plt.figure()

    plt.plot(epochs, psnr, 'r', label='Training psnr')
    plt.plot(epochs, val_psnr, 'b', label='Validation psnr')
    plt.title('Training and validation')
    plt.legend(loc=0)
    plt.figure()

    plt.plot(epochs, nmse, 'r', label='Training nmse')
    plt.plot(epochs, val_nmse, 'b', label='Validation nmse')
    plt.title('Training and validation')
    plt.legend(loc=0)
    plt.figure()

    plt.show()

def predict_reconstruct_image(test_inputs):
    """ The purpose of this function is predict the reconstruction image
    and display one example of the reconstruction image
    Inputs:
        -- test_inputs: modified images
    Output:
        None
    """
    # load model
    # model = load_model('single_coil_Unet.h5')
    model = load_model('single_coil_Unet_model', custom_objects={'nmse_metrix':nmse_metrix, 'psnr_metrix':psnr_metrix, 'ssim_metrix': ssim_metrix})

    preds_test = model.predict(np.array(test_inputs, np.float32), verbose=1)

    idx = random.randint(0, len(preds_test))
    imshow(test_inputs[idx])
    plt.show()
    reconstruction_img = np.squeeze(preds_test[idx])
    # reconstruction_img = (reconstruction_img - np.min(reconstruction_img)) / (np.max(reconstruction_img) - np.min(reconstruction_img))
    imshow(np.log(100 * reconstruction_img + 1e-10), cmap="gray")
    plt.show()
    print('ok')

"""
Reconstruction evaluation metrics
1. Normalized Mean Square Error
2. Peak Signal-to-Noise Ratio
3. Structural Similarity
"""
def nmse_metrix(y_truth, y_predict):
    return tf.divide(tf.reduce_sum(tf.math.squared_difference(y_truth, y_predict)), tf.reduce_sum(y_truth ** 2))

def psnr_metrix(y_truth, y_predict):
    return tf.reduce_mean(tf.image.psnr(y_truth, y_predict, max_val=1.0))

def ssim_metrix(y_truth, y_predict):
    y_truth = tf.expand_dims(y_truth, axis=3)
    y_predict = tf.expand_dims(y_predict, axis=3)

    return tf.reduce_mean(tf.image.ssim(y_truth, y_predict, max_val=1.0, filter_size=7))



def prepare_datasets():
    """ The purpose of this function is to load dataset from h5 file in the filepath,
    call LPF function to modify K-space
    call ifft to get the modified image
    split the dataset into train and test
    Inputs:
        -- shape: the same as K-space [row, col]
    Output:
        -- train_inputs:  modified images
        -- train_outputs: ground truth image
        -- test_inputs: modified images
        -- test_outputs: ground truth image
    """

    file_path = '/Users/xiaoqingliu/Downloads/singlecoil_val'
    # output_path = '/Users/xiaoqingliu/Downloads/singlecoil_val_distorted_image'
    display = False
    h5_re = re.compile('.+(?=\.h5$)')

    h5_list = []
    # output_h5_list = []
    for root, dirs, files in os.walk(file_path):
        for f in files:
            f_base = os.path.basename(f)
            match = h5_re.findall(f_base)
            if match != []:
                h5_list.append(f_base)
                # output_h5_list.append(match[0] + '_img.h5')

    total = len(h5_list)

    images_y = []
    images_x = []
    for i, file_name in enumerate(h5_list):
        print(i, '/', total)
        hf = h5py.File(os.path.join(file_path, file_name), 'r')
        volume_kspace = hf['kspace'][()]
        # # print(volume_kspace.dtype)
        # # print(volume_kspace.shape)
        # data_out = np.zeros(volume_kspace.shape)

        """
        Only picking slice 20 of each volume
        """
        # for i_slice in range(volume_kspace.shape[0]):
        for i_slice in range(10,21):
        # for i_slice in range(20):
            slice_kspace = volume_kspace[i_slice]
            slice_real = fftshift(ifft2(slice_kspace))
            ground_truth_img = np.abs(slice_real)
            # resize image into 320x320
            resize_truth_img = resize(ground_truth_img, (320,320), mode='constant')
            images_y.append(resize_truth_img)
            if display:
                show_coils(np.log(np.abs(slice_kspace) + 1e-9), 0)
                fig = plt.figure()
                plt.imshow(ground_truth_img, cmap='gray')
                fig = plt.figure()
                plt.imshow(resize_truth_img, cmap='gray')
            x, y = slice_kspace.shape
            center_x = int(x / 2)
            center_y = int(y / 2)
            ilp_filter = create_lp_filter(slice_kspace.shape, (center_y, center_x), 50, 2)
            slice_filted = ilp_filter * slice_kspace

            if display:
                fig = plt.figure()
                plt.imshow(ilp_filter, cmap='gray', vmin=0, vmax=1.0)

            slice_filtered_real = np.abs(fftshift(ifft2(slice_filted)))
            out_img = slice_filtered_real / np.max(slice_filtered_real)

            #
            resize_filt_img = resize(out_img, (320, 320), mode='constant')
            images_x.append(resize_filt_img)

            if display:
                fig = plt.figure()
                plt.imshow(out_img, cmap='gray')
                fig = plt.figure()
                plt.imshow(resize_filt_img, cmap='gray')
                plt.close('all')
                # print('OK')
            # data_out[i, :, :] = out_img

        # output_file_name = os.path.join(output_path, output_h5_list[i])
        # hf_out = h5py.File(output_file_name, mode='w')
        # hf_out.create_dataset('img', data=data_out)
        # hf_out.close()
    train_counts = round(len(images_x) * 0.8)
    train_inputs = images_x[:train_counts]
    train_outputs = images_y[:train_counts]
    test_inputs = images_x[train_counts:]
    test_outputs = images_y[train_counts:]

    return train_inputs, train_outputs,test_inputs,test_outputs



def show_coils(data):
    """ This function to display the coil image """
    fig = plt.figure()
    plt.imshow(data, cmap='gray')


def create_lp_filter(shape, center, radius, lp_type=2, n=2):
    """ The Low Pass Filter funtion that implements three LPF
    Inputs:
        -- shape: the same as K-space [row, col]
        -- center: the center of K-space [row/2, col/2]
        -- radius: the distance from the center, like 30
        -- lp_type: 0 is ideal LPF; 1 is Butterworth LPF; 2 is Gaussian LPF;
        -- n: the rank of Butterworth LPF
    Output:
        -- lp_filter: low pass filter matrix
    """
    rows, cols = shape[:2]
    r, c = np.mgrid[0:rows:1, 0:cols:1]
    c -= center[0]
    r -= center[1]
    d = np.power(c, 2.0) + np.power(r, 2.0)

    if lp_type == 0:
        lp_filter = np.copy(d)
        lp_filter[lp_filter < pow(radius, 2.0)] = 1
        lp_filter[lp_filter >= pow(radius, 2.0)] = 0
    elif lp_type == 1:
        lp_filter = 1.0 / (1 + np.power(np.sqrt(d) / radius, 2 * n))
    elif lp_type == 2:
        lp_filter = np.exp(-d / (2 * pow(radius, 2.0)))

    return lp_filter


def main():
    """ The main funtion that calls the function to prepare dataset,
    training U-net, and predict output reconstruct image
    If the training's done, we have single_coil_Unet_model, we can just load the model
    to reconstruct images
    """
    train_inputs, train_outputs,test_inputs,test_outputs = prepare_datasets()
    # train_predict_model(train_inputs, train_outputs, test_inputs)
    Training = False
    if Training == True:
        train_model(train_inputs, train_outputs)
    predict_reconstruct_image(test_inputs)


if __name__ == "__main__":
    main()