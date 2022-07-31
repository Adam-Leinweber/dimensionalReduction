#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.layers import Lambda, Input, Dense, LeakyReLU, BatchNormalization, Concatenate, Reshape, Conv1D
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy, categorical_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import regularizers
from scipy.stats import multivariate_normal
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split


import time
import sys
import h5py
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import interactive

data = np.loadtxt('../data/train_data_masses.csv', delimiter = ',')

data = data[:,:4] # remove mass parameters, uncomment if you just want the model parameters.

scaler_filename = 'MSSMEW_scaler.save'

scaler = MinMaxScaler(feature_range=(0,1))
#scaler = StandardScaler()
scaler.fit(data)
joblib.dump(scaler, scaler_filename)
data = scaler.transform(data)
data[:,1] = np.power(data[:,1],1/2) #unskew M2 value

reg = None
batch_size = 1000
latent_dim = 2
epochs = 10000
# MODEL_FILENAME = 'apr21-dimensional-reduction_'+sys.argv[3]+'.h5'
MODEL_FILENAME = 'jun22-dimensional-reduction-masses-beta_1.h5'


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def kl_loss(true, pred, z_mean, z_log_var):
    loss_parameters = mse(true, pred)
    
    reconstruction_loss = K.mean(loss_parameters)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    return(K.mean((1-beta)*reconstruction_loss + beta * kl_loss))    

def wrapper(z_mean, z_log_var):
    def custom_kl_loss(true, pred):
        return kl_loss(true, pred, z_mean, z_log_var)
    return custom_kl_loss

def load_vae(nodes, beta):
    # in_parameters = Input(shape=(4,), name='in_parameters')
    in_parameters = Input(shape=(10,), name='in_parameters')
    inputs = in_parameters
    x = Dense(nodes[0], activation='tanh', kernel_regularizer=reg)(inputs)
    x = Dense(nodes[1], activation='tanh', kernel_regularizer=reg)(x)
    x = Dense(nodes[2], activation='tanh', kernel_regularizer=reg)(x)
    x = Dense(nodes[3], activation='tanh', kernel_regularizer=reg)(x)
    x = Dense(nodes[4], activation='tanh', kernel_regularizer=reg)(x)
    z_mean = Dense(latent_dim, name='z_mean', activation='linear', kernel_regularizer=reg)(x)
    z_log_var = Dense(latent_dim, name='z_log_var', activation='linear', kernel_regularizer=reg)(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = Model(in_parameters, [z_mean, z_log_var, z], name='encoder')
    #encoder.summary()
    #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(nodes[4], activation='tanh', kernel_regularizer=reg)(latent_inputs)
    x = Dense(nodes[3], activation='tanh', kernel_regularizer=reg)(x)
    x = Dense(nodes[2], activation='tanh', kernel_regularizer=reg)(x)
    x = Dense(nodes[1], activation='tanh', kernel_regularizer=reg)(x)
    x = Dense(nodes[0], activation='tanh', kernel_regularizer=reg)(x)

    # out_parameters = Dense(4, activation='linear', name='out_parameters')(x)
    out_parameters = Dense(10, activation='linear', name='out_parameters')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, out_parameters, name='decoder')
    #decoder.summary()
    #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    outputs = decoder(encoder(in_parameters)[2])
    vae = Model(
        in_parameters, 
        outputs,
        name='vae_mlp'
    )

    model_kl_loss = wrapper(z_mean, z_log_var)
    
    #vae.add_loss(vae_loss)
    vae.compile(optimizer='adam', loss=model_kl_loss)
    return encoder, vae, decoder


# n = len(sys.argv[1])
# a = sys.argv[1][1:n-1]
# a = a.split(',')
# a = [int(x) for x in a]
# nodes_list = [a] #[[8,6,4],[32,16,8],[64,32,16],[128,64,32],[256,128,64]]
# beta_list = [float(sys.argv[2])]  #[1e-4]
# encoder, vae, decoder = load_vae()

nodes_list = [[100,100,50,25,10]]
beta_list = [1e-6]

for nodes in nodes_list:
    for beta in beta_list:
        architecture = str(nodes[0])+'-'+str(nodes[1])+'-'+str(nodes[2])+'-2-'+str(nodes[2])+'-'+str(nodes[1])+'-'+str(nodes[0])+'_'+str(beta)
        
        ratio = 0.8
        train, validate = train_test_split(data, train_size = ratio, shuffle = True)

        do_train = True
        if do_train:
            earlystopper = EarlyStopping(monitor='loss', patience=50, verbose=1)
            checkpointer = ModelCheckpoint(MODEL_FILENAME, monitor='val_loss', verbose=1, save_best_only=True)
            tensorboard = TensorBoard(log_dir='./logs/test')
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50, min_lr=0.000000001)

            iteration=0
            learning_rate = 1e-3
            opt = Adam(lr=learning_rate)
            encoder, vae, decoder = load_vae(nodes, beta)
            while iteration < 10:
                iteration += 1
                opt = Adam(lr=learning_rate)
                results = vae.fit(
                    x=train,
                    y=train,
                    batch_size=batch_size, 
                    shuffle="batch", 
                    epochs=epochs, 
                    verbose=1, 
                    callbacks=[checkpointer, earlystopper, tensorboard],
                    validation_data=[validate, validate],
                )
                vae.save_weights(filepath=MODEL_FILENAME)
                encoder, vae, decoder = load_vae(nodes, beta)
                vae.load_weights(MODEL_FILENAME)
                learning_rate /= 2

        encoder, vae, decoder = load_vae(nodes, beta)
        vae.load_weights(MODEL_FILENAME)
        
