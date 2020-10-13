import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K

"""
This demo was originally based on the code found here:
https://users.fmrib.ox.ac.uk/~saad/ae1.html
"""

num_samples_per_class = 250
var_width = 0.25
cov1 = [[1, var_width, var_width],
       [.9, var_width, var_width],
       [.9, var_width, var_width]]
cov2 = [[var_width, .9, var_width],
       [var_width, 1, var_width],
       [var_width, .9, var_width]]
cov3 = [[var_width, var_width, .9],
       [var_width, var_width, .9],
       [var_width, var_width, 1]]
class1 = np.random.multivariate_normal([0.5, 0.5, 0.5], cov1, num_samples_per_class)
class2 = np.random.multivariate_normal([-0.5, 0.5, 1], cov2, num_samples_per_class)
class3 = np.random.multivariate_normal([0.5, -0.5, -0.5], cov3, num_samples_per_class)

data = np.concatenate((class1, class2, class3), axis=0)
labels = np.repeat([1, 2, 3], num_samples_per_class)



def create_autoencoder(input_dim, latent_dim=2):
    # Encoder Model
    inputs = Input(shape=(input_dim,), name='Input')
    x = Dense(100, activation='relu')(inputs)
    x = Dense(20, activation='relu')(x)
    lat = Dense(latent_dim, activation='sigmoid')(x)

    encoder = Model(inputs, lat, name='encoder')

    # Decoder model
    lat_input = Input(shape=(latent_dim,))
    x = Dense(20, activation='relu')(lat_input)
    x = Dense(100, activation='relu')(x)
    outputs = Dense(input_dim, activation='sigmoid')(x)

    decoder = Model(lat_input, outputs, name='decoder')

    # Autoencoder
    outputs = decoder(encoder(inputs))
    ae = Model(inputs, outputs, name='ae')

    return ae, encoder, decoder

def recon_loss(inputs,outputs):
    reconstruction_loss = mse(inputs, outputs)
    return K.mean(reconstruction_loss)

ae, enc, dec = create_autoencoder(input_dim=data.shape[1], latent_dim=2)
ae.summary()

# define loss function
losses = {'decoder': recon_loss}

# choose optimisation routine
ae.compile(optimizer='adam', loss=losses)

# run the fitting
ae.fit(data,
        {'decoder':data},
        epochs=200,
        batch_size=128,shuffle=True)

#plt.plot(ae.history.history['loss'])
#plt.show()

encoded_data = enc.predict(data)

unique_labels = np.unique(labels)

plt.figure(figsize=(10, 7))
plt.title('Input Data in 3D')
ax = plt.axes(projection="3d")
for label in unique_labels:
    indices = np.where(labels == label)[0]
    ax.scatter3D(data[indices, 0], data[indices, 1], data[indices, 2])

plt.figure(figsize=(10, 7))
plt.title('Encoded data in 2D')
for label in unique_labels:
    indices = np.where(labels == label)[0]
    plt.scatter(encoded_data[indices, 0], encoded_data[indices, 1])

plt.show()
