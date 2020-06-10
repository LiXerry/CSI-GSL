import keras
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import tensorflow as tf

from keras import backend as K
from keras.layers import Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils import to_categorical


class GANNumericalLabel():

    def __init__(self):
        self.data_shape = (6400,)
        self.noise_shape = (6400,)

        # Manually tune down learning rate to avoid oscillation
        optimizer = Adam(lr=0.005, beta_1=0.5)

        # -------------
        # Discriminator
        # -------------
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mean_squared_error',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        # ---------
        # Generator
        # ---------
        self.generator = self.build_generator()
        self.generator.compile(loss='mean_squared_error',
                               optimizer=optimizer)
        # --------
        # Combined
        # --------
        # The combined model is created by stacking generator and discriminator.
        # Noise ---Generator--> Generated Data ---Discriminator--> Validity

        z = Input(shape=self.noise_shape)
        data = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        validity = self.discriminator(data)

        self.combined = Model(z, validity)
        self.combined.compile(loss='mean_squared_error',
                              optimizer=optimizer)

    def save_model(self, version=None):
        if version is None:
            self.discriminator.save('./gan/models/toy-numerical-label-discriminator.h5')
            self.generator.save('./gan/models/toy-numerical-label-generator.h5')
            self.combined.save('./gan/models/toy-numerical-label-combined.h5')
        else:
            self.discriminator.save('./gan/models/toy-numerical-label-discriminator-{}.h5'.format(version))
            self.generator.save('./gan/models/toy-numerical-label-generator-{}.h5'.format(version))
            self.combined.save('./gan/models/toy-numerical-label-combined-{}.h5'.format(version))

    def load_model(self, version=None):
        if version is None:
            self.discriminator = load_model('./gan/models/toy-numerical-label-discriminator.h5')
            self.generator = load_model('./gan/models/toy-numerical-label-generator.h5')
            self.combined = load_model('gan/models/toy-numerical-label-combined.h5')
        else:
            self.discriminator = load_model('./gan/models/toy-numerical-label-discriminator-{}.h5'.format(version))
            self.generator = load_model('./gan/models/toy-numerical-label-generator-{}.h5'.format(version))
            self.combined = load_model('./gan/models/toy-numerical-label-combined-{}.h5'.format(version))

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(12800, input_shape=self.data_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(6400))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(6400, activation='sigmoid'))

        model.summary()

        # Discriminator takes an image as an input and outputs its validity
        data = Input(shape=self.data_shape)
        validity = model(data)

        return Model(data, validity)

    def build_generator(self):
        # BatchNormalization maintains the mean activation close to 0
        # and the activation standard deviation close to 1
        model = Sequential()
        model.add(Dense(12800, input_shape=self.noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(6400))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(6400))

        model.summary()

        # Generator takes noise as an input and outputs an image
        noise = Input(shape=self.noise_shape)
        data = model(noise)

        return Model(noise, data)

    def train(self, train_data, epochs=200, batch_size=128, save_model_interval=10000):

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):
            # -------------------
            # Train Discriminator
            # -------------------
            # Select a random half batch of images
            idx = np.random.randint(0, train_data.shape[0], half_batch)
            data = train_data[idx]

            # Generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, 6400))
            gen_data = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(data, np.ones((half_batch, 6400)))
            d_loss_fake = self.discriminator.train_on_batch(gen_data, np.zeros((half_batch, 6400)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------
            # Train Generator
            # ---------------
            noise = np.random.normal(0, 1, (batch_size, 6400))

            # The generator wants to fool the discriminator, hence trained with valid label (1)
            # valid_y = np.array([1] * batch_size)
            valid_y = np.ones((batch_size, 6400))

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Print progress
            print("{:5d} [D loss: {}, acc_real: {:2f}, acc_fake: {:2f}] [G loss: {}]".format(epoch, d_loss[0],
                                                                                             100 * d_loss_real[1],
                                                                                             100 * d_loss_fake[1],
                                                                                             g_loss))

            with open('./gan/logs/toy-numerical-label-gan.log', 'a') as log_file:
                log_file.write('{},{}\n'.format(d_loss[0], g_loss))

            # Save models at save_interval
            if epoch != 0 and epoch % save_model_interval == 0:
                self.save_model(version=str(epoch))


# x_train = load_data('numerical-label-train-x')
# y_train = load_data('numerical-label-train-y')
#
# expanded_y_train = np.expand_dims(y_train, axis=1)
# merged_train_data = np.concatenate((x_train, expanded_y_train), axis=1)
#
# gan = GANNumericalLabel()
# gan.train(merged_train_data)
# noise = np.random.normal(0, 1, (1000, 10))
# new_numerical_label_data = gan.generator.predict(noise)