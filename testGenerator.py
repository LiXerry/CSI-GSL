import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
import numpy as np
from keras import regularizers
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten, Lambda, Conv2D, Concatenate, Add, Multiply, Reshape, Dropout, Conv3D, LeakyReLU
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
# import keras
# import pickle as p
from data_generator import DataGenerator

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

transfered = "ULA"
scenarios = ["distributed", "URA"]
num_antennas = [64]
# scenario = "URA"
data_path = "/home/lxhit/code/CSI-position/data/new-data/mamimo_measurements/"

tf.logging.set_verbosity(tf.logging.ERROR)

num_samples = 252004

# Training size
trainings_size = 0.25                     # 85% training set
validation_size = 0.1                     # 10% validation set
test_size = 0.05                          # 5% test set

# Number of Antennas
# num_antennas = 64
num_sub = 100

labels = np.load(data_path + 'labels.npy')


for scenario in scenarios:
    # check for bad channels (channels with corrupt data)
    bad_samples = np.load(data_path + "bad_channels_" + scenario + ".npy")
    # buils array with all valid channel indices
    IDs = []
    for x in range(num_samples):
        if x not in bad_samples:
            IDs.append(x)
    IDs = np.array(IDs)
    # shuffle the indices with fixed seed
    np.random.seed(64)
    np.random.shuffle(IDs)
    # get the number of channels
    actual_num_samples = IDs.shape[0]
    # distributed the samples over the train, validation and test set
    train_IDs = IDs[:int(trainings_size*actual_num_samples)] # first 85% of the data
    val_IDs = IDs[int(trainings_size*actual_num_samples):int((trainings_size + validation_size) * actual_num_samples)]
    test_IDs = IDs[-int(test_size * actual_num_samples):] # last 5% of the data
    for num_antenna in num_antennas:
        print("scenario:", scenario, "number of antennas:", num_antenna)
        # nn = build_nn(num_antenna)
        #
        val_generator = DataGenerator(scenario, val_IDs, labels,
                                      num_antennas=num_antenna,
                                      data_path=data_path)
        print(val_generator)
        print(val_generator.shape)
        test_generator = DataGenerator(scenario, test_IDs, labels,
                                       shuffle=False, num_antennas=num_antenna,
                                       data_path=data_path)

