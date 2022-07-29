#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.random import set_seed
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import History 
import matplotlib.pyplot as plt
from subprocess import check_output
from keras.optimizers.schedules import ExponentialDecay
from sklearn.neighbors import kneighbors_graph
import requests

from GLayers import GConvLayer
import os

from spektral.layers.convolutional.gcn_conv import GCNConv
from spektral.datasets import mnist

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

seed = 12345
np.random.seed(seed)
set_seed(seed)

#%%

class GCN(tf.keras.Model):
    """
    This model, with its default hyperparameters, implements the architecture
    from the paper:
    > [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)<br>
    > Thomas N. Kipf and Max Welling
    **Mode**: single, disjoint, mixed, batch.
    **Input**
    - Node features of shape `([batch], n_nodes, n_node_features)`
    - Weighted adjacency matrix of shape `([batch], n_nodes, n_nodes)`
    **Output**
    - Softmax predictions with shape `([batch], n_nodes, n_labels)`.
    **Arguments**
    - `n_labels`: number of channels in output;
    - `channels`: number of channels in first GCNConv layer;
    - `activation`: activation of the first GCNConv layer;
    - `output_activation`: activation of the second GCNConv layer;
    - `use_bias`: whether to add a learnable bias to the two GCNConv layers;
    - `dropout_rate`: `rate` used in `Dropout` layers;
    - `l2_reg`: l2 regularization strength;
    - `**kwargs`: passed to `Model.__init__`.
    """

    def __init__(
        self,
        n_labels,
        channels=16,
        activation="relu",
        output_activation="softmax",
        use_bias=False,
        dropout_rate=0.5,
        l2_reg=2.5e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.n_labels = n_labels
        self.channels = channels
        self.activation = activation
        self.output_activation = output_activation
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        reg = tf.keras.regularizers.l2(l2_reg)
        self.inp = tf.keras.layers.Input(shape=(4,),name="input_1")
        self._d0 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn0 = GCNConv(
            channels, activation=activation, kernel_regularizer=reg, use_bias=use_bias
        )
        self._d1 = tf.keras.layers.Dropout(dropout_rate)
        self._gcn1 = GCNConv(
            n_labels, activation=output_activation, use_bias=use_bias
        )

    def get_config(self):
        return dict(
            n_labels=self.n_labels,
            channels=self.channels,
            activation=self.activation,
            output_activation=self.output_activation,
            use_bias=self.use_bias,
            dropout_rate=self.dropout_rate,
            l2_reg=self.l2_reg,
        )

    def call(self, inputs):
        if len(inputs) == 2:
            x, a = inputs
        else:
            x, a, _ = inputs  # So that the model can be used with DisjointLoader
        x = self.inp(x)
        x = self._d0(x)
        x = self._gcn0([x, a])
        x = self._d1(x)
        return self._gcn1([x, a])
#%%

# if (not 'datatrain' in locals() and not 'datatrain' in locals()):
#     (datatrain,_),(datatest,_) = tf.keras.datasets.fashion_mnist.load_data()

def preprocess_images(images, bn = False):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    if bn == True:
        return np.where(images > .5, 1.0, 0.0).astype('float32')
    return images
  
def data_loader():
    (datatrain,_),(datatest,_) = tf.keras.datasets.fashion_mnist.load_data()
    datatrain = preprocess_images(datatrain)
    datatest = preprocess_images(datatest)
        
    trainset = shuffle(datatrain,random_state=seed)
    testset = shuffle(datatest,random_state=seed)
    
    return trainset, testset

#%%
## Create a Sampling layer

class Sampling(tf.keras.layers.Layer):
    
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch,dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon

#%%
@tf.function
def threeD_loss(inputs, outputs):
    expand_inputs = tf.expand_dims(inputs, 2)
    expand_outputs = tf.expand_dims(outputs, 1)
    distances = tf.math.reduce_sum(tf.math.squared_difference(expand_inputs, expand_outputs), -1)
    min_dist_to_inputs = tf.math.reduce_min(distances,1)
    min_dist_to_outputs = tf.math.reduce_min(distances,2)
    return tf.math.reduce_mean(min_dist_to_inputs, 1) + tf.math.reduce_mean(min_dist_to_outputs, 1)
#%%
latent_dim = 2

encoder_inputs = tf.keras.Input(shape=(36,1),dtype=tf.float32)
adjacency_input = tf.keras.Input(shape=(36,36),dtype=tf.float32)
#x = tf.keras.layers.Dropout(0)(encoder_inputs)
x = GConvLayer(outdim=16)(encoder_inputs,adjacency_input)
#x = tf.keras.layers.Conv2D(32, 2, activation = "relu", strides = 2, padding="same")(x)
#x = tf.keras.layers.Conv2D(64, 2, activation = "relu", strides = 1, padding="same")(x)
x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(32, activation="relu")(encoder_inputs)
# x = tf.keras.layers.Dense(64, activation="relu")(x)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(16, activation="relu")(x)

z_mean = tf.keras.layers.Dense(latent_dim, name ="z_mean")(x)
z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = tf.keras.Model(inputs=(encoder_inputs, adjacency_input), outputs=[z_mean, z_log_var, z], name ="encoder")
encoder.summary()    

#%%

latent_inputs = tf.keras.Input(shape=(latent_dim,))
adjacency_input_dec  = tf.keras.Input(shape=(36,36),dtype=tf.float32)
x = tf.keras.layers.Dense(16, activation="relu")(latent_inputs)
#x = tf.keras.layers.Dense(3*3*64, activation ="relu")(latent_inputs)
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(576)(x)
#x = tf.keras.layers.Reshape((3,3,64))(x)
#x = tf.keras.layers.Conv2DTranspose(64, 2, activation = "sigmoid", strides = 1, padding = "same")(x)
#x = tf.keras.layers.Conv2DTranspose(32, 2, activation = "sigmoid", strides = 2, padding = "same")(x)
x = tf.keras.layers.Reshape((36,16), input_shape=(576,))(x)
x = GConvLayer(outdim=16)(x,adjacency_input_dec)
# x = tf.keras.layers.Dense(64, activation="sigmoid")(x)
# x = tf.keras.layers.Dense(32, activation="sigmoid")(x)
decoder_outputs = GConvLayer(outdim=1)(x,adjacency_input_dec)
# decoder_outputs = tf.keras.layers.Dense(27,activation="sigmoid")(x)
#x = tf.keras.layers.Conv2DTranspose(1, 2, activation = "sigmoid", padding ="same")(x)
#x = tf.keras.layers.Flatten()(x)
#decoder_outputs = tf.keras.layers.Dense(36)(x)
decoder = tf.keras.Model(inputs=(latent_inputs,adjacency_input_dec), outputs=decoder_outputs, name ="decoder")
decoder.summary()

#%%

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name = "reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name = "kl_loss")
        self.adj_matx = np.ones((36,36))
        self.kl_warmup = tf.Variable(0.0, trainable=False, name='beta_kl_warmup', dtype=tf.float32)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
            ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(inputs=(data,self.adj_matx))
            reconstruction = self.decoder(inputs=(z,self.adj_matx))
            reconstruction_loss =  tf.math.reduce_mean(threeD_loss(data,reconstruction))
            kl_loss = (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(kl_loss, axis = -1))
            #beta = tf.Variable(1.)
            total_loss = reconstruction_loss +kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads,self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss" : self.total_loss_tracker.result(),
            "reconstruction_loss" : self.reconstruction_loss_tracker.result(),
            "kl_loss" : self.kl_loss_tracker.result()
            
            }
    def call(self,inputs):
        data,adj_mat = inputs
        z, z_mean, z_log_var = self.encoder(inputs)
        out = self.decoder(inputs=(z,adj_mat))
        return out, z, z_mean, z_log_var
        
        
#%%

def plot_label_clusters(vae, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = vae.encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()
    
def plot_latent_space(vae, n=30, figsize=15):
    # display a n*n 2D manifold of digits
    digit_size = 6
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()
#%%

#if (not 'datatrain' in locals() and not 'datatrain' in locals()):
#    trainset, testset = data_loader()
#trainset, testset = data_loader()


outrootname = "thesisvae"
Nfolder = check_output("if [ ! -d \"" + outrootname + "\" ]; then mkdir \"" + outrootname + "\";fi",shell=True)
Nfolder = check_output("if [ ! -d \"" + outrootname + "/hist\" ]; then mkdir \"" + outrootname + "/hist\";fi",shell=True)

def data_upload(datapath,name):
    '''
    Function to load data from disk or using an URL.
    Parameters
    ----------
    datapath : String
        path (local or URL) of data in csv format.
    Returns
    -------
    pandas.dataframe
        Dataframe containing data used for training and/or inference.
    '''
    if("http" in datapath):
        print("Downloading Dataset")
        try:
            # Download
            dataset = requests.get(datapath)
            dataset.raise_for_status()
        except requests.exceptions.RequestException:
            print("Error: Could not download file")
            raise        
        # Writing dataset on disk.    
        with open(outrootname+"/"+name+".csv","wb") as o:
            o.write(dataset.content)
        datapath = outrootname+"/"+name+".csv"
    print("Loading Dataset from Disk")
    
    # Reading dataset and creating pandas.DataFrame.
    dataset = pd.read_csv(datapath,header=0)
    print("Entries ", len(dataset))        
    
    return dataset


def preprocess_features(muon_dataframe):
    """Prepares input features from Muon data set.

    Args:
        muon_dataframe: A Pandas DataFrame expected to contain data
        from muon simulations
    Returns:
        A DataFrame that contains the features to be used for the model.
  """
    selected_features = muon_dataframe[
      [#'Event',
    'n_Primitive',
    '1dtPrimitive.id_r',
    '2dtPrimitive.id_r',
    '3dtPrimitive.id_r',
    '4dtPrimitive.id_r',
    '1dtPrimitive.id_eta',
    '2dtPrimitive.id_eta',
    '3dtPrimitive.id_eta',
    '4dtPrimitive.id_eta',
    '1dtPrimitive.id_phi',
    '2dtPrimitive.id_phi',
    '3dtPrimitive.id_phi',
    '4dtPrimitive.id_phi',
    '1dtPrimitive.phiB',
    '2dtPrimitive.phiB',
    '3dtPrimitive.phiB',
    '4dtPrimitive.phiB',
    '1dtPrimitive.quality',
    '2dtPrimitive.quality',
    '3dtPrimitive.quality',
    '4dtPrimitive.quality',
    '1l1Muon.eta',
    '2l1Muon.eta',
    '3l1Muon.eta',
    '1l1Muon.phi',
    '2l1Muon.phi',
    'delta_phi12',
    'delta_phi13',
    'delta_phi14',
    'delta_phi23',
    'delta_phi24',
    'delta_phi34',
    'genParticle.pdgId',
    'genParticle.eta',
    'genParticle.phi',
    'genParticle.pt'
         ]]
    

    processed_features = selected_features.copy()
    return processed_features.astype(np.float64)


def datasets():
    '''
    Normalization of phiBending and cutting entries with a pT higher than 200 GeV

    Returns
    -------
    int
        No error code.

    '''
    out_dataframe = data_upload("https://www.dropbox.com/s/m8ahte2xcwby8ew/bxcut_full_muon.csv?dl=1","bxcut_full_muon.csv")
    # muon_dataframe_test = data_upload("https://www.dropbox.com/s/9pxb63k60cz4hif/bxcut_full_test.csv?dl=1",'bxcut_full_test.csv')
    
    out_dataframe["1dtPrimitive.phiB"] = out_dataframe["1dtPrimitive.phiB"]/512.
    out_dataframe["2dtPrimitive.phiB"] = out_dataframe["2dtPrimitive.phiB"]/512.
    out_dataframe["3dtPrimitive.phiB"] = out_dataframe["3dtPrimitive.phiB"]/512.
    out_dataframe["4dtPrimitive.phiB"] = out_dataframe["4dtPrimitive.phiB"]/512.
    
    # muon_dataframe_test["1dtPrimitive.phiB"] = muon_dataframe_test["1dtPrimitive.phiB"]/512.
    # muon_dataframe_test["2dtPrimitive.phiB"] = muon_dataframe_test["2dtPrimitive.phiB"]/512.
    # muon_dataframe_test["3dtPrimitive.phiB"] = muon_dataframe_test["3dtPrimitive.phiB"]/512.
    # muon_dataframe_test["4dtPrimitive.phiB"] = muon_dataframe_test["4dtPrimitive.phiB"]/512.
    
  
    
    out_dataframe = out_dataframe[out_dataframe['genParticle.pt'] <= 200]
    # muon_dataframe_test = muon_dataframe_test[muon_dataframe_test['genParticle.pt'] <= 200]
    # muon_dataframe_test.to_csv("muon_test.csv")
    out_dataframe.to_csv("out_data.csv")
    return 0

def preproc(test=False):
    '''
    Selection of features used in Training/Test. Changing the quality feature to a binary one.

    Parameters
    ----------
    test : Boolean, optional
        Flag to preprocess the training set (False) or the testing set (True). The default is False.

    Returns
    -------
    List of Numpy Arrays
        if test=True: numpy arrays containing input data and true labels respectively for testing.
    List of Numpy Arrays
        if test=False: numpy arrays containing input data and true labels respectively for testing.

    '''
    try:
        
        out_dataframe = pd.read_csv('out_data.csv')
    except FileNotFoundError:
        datasets()
        out_dataframe = pd.read_csv('out_data.csv')
    # try:
    #     muon_dataframe_test = pd.read_csv('muon_test.csv')
    # except FileNotFoundError:
    #     datasets()
    #     muon_dataframe_test = pd.read_csv('muon_test.csv')
    
    X = preprocess_features(out_dataframe)
    # X_test = preprocess_features(muon_dataframe_test)
    
    
    X.loc[X["1dtPrimitive.quality"] < 4, '1dtPrimitive.quality'] = 0.0
    X.loc[X["1dtPrimitive.quality"] >= 4, '1dtPrimitive.quality'] = 1.0
    X.loc[X["2dtPrimitive.quality"] < 4, '2dtPrimitive.quality'] = 0.0
    X.loc[X["2dtPrimitive.quality"] >= 4, '2dtPrimitive.quality'] = 1.0
    X.loc[X["3dtPrimitive.quality"] < 4, '3dtPrimitive.quality'] = 0.0
    X.loc[X["3dtPrimitive.quality"] >= 4, '3dtPrimitive.quality'] = 1.0
    X.loc[X["4dtPrimitive.quality"] < 4, '4dtPrimitive.quality'] = 0.0
    X.loc[X["4dtPrimitive.quality"] >= 4, '4dtPrimitive.quality'] = 1.0
    
    # X_test.loc[X_test["1dtPrimitive.quality"] < 4, '1dtPrimitive.quality'] = 0.0
    # X_test.loc[X_test["1dtPrimitive.quality"] >= 4, '1dtPrimitive.quality'] = 1.0
    # X_test.loc[X_test["2dtPrimitive.quality"] < 4, '2dtPrimitive.quality'] = 0.0
    # X_test.loc[X_test["2dtPrimitive.quality"] >= 4, '2dtPrimitive.quality'] = 1.0
    # X_test.loc[X_test["3dtPrimitive.quality"] < 4, '3dtPrimitive.quality'] = 0.0
    # X_test.loc[X_test["3dtPrimitive.quality"] >= 4, '3dtPrimitive.quality'] = 1.0
    # X_test.loc[X_test["4dtPrimitive.quality"] < 4, '4dtPrimitive.quality'] = 0.0
    # X_test.loc[X_test["4dtPrimitive.quality"] >= 4, '4dtPrimitive.quality'] = 1.0



    
    # X['filler0'] = 0
    # X['filler1'] = 0
    # X['filler2'] = 0
    # X['filler3'] = 0
    # X['filler4'] = 0
    
    scaler = MinMaxScaler()
    scaler.fit(X)
    
    X = scaler.transform(X)

    X = X.reshape(X.shape[0],36,1)

    


    return X,scaler
#%%
def scaled_predict(vae,scaler,data):
    pred = vae.decoder.predict(data)
    pred = pred.reshape(data.shape[0],36)

    repred = scaler.inverse_transform(pred)

    repred = np.delete(repred, np.s_[31:37], axis=1)
    repred = np.delete(repred, np.s_[27:29], axis=1)

    repred[:,:13] = np.round(repred[:,:13])

    return repred

#%%
dataset,scaler = preproc()
trainset, valset = train_test_split(dataset, test_size=0.2, random_state=seed)
adj_matx = np.ones((36,36))
#%%
history = History()
vae = VAE(encoder,decoder)
learning_schedule = ExponentialDecay(0.00001,decay_steps=1000,decay_rate=0.96,staircase=True)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_schedule))
vae.fit(dataset,validation_split=0.2,epochs=50,batch_size=128,callbacks=[history])

fig2,ax2 = plt.subplots()
#ax2.plot(history.history['kl_loss'], label='kl_loss')
ax2.plot(history.history['reconstruction_loss'], label='rec_loss')
#ax2.plot(history.history['loss'], label='tot_loss')
ax2.set_title('Training loss per epoch')
ax2.set_xlabel('# Epoch')
ax2.set_ylabel('loss')
plt.legend()
plt.tight_layout()
plt.show()

fig,ax = plt.subplots()
ax.plot(history.history['kl_loss'], label='kl_loss')
#ax2.plot(history.history['reconstruction_loss'], label='rec_loss')
#ax2.plot(history.history['loss'], label='tot_loss')
ax.set_title('Training loss per epoch')
ax.set_xlabel('# Epoch')
ax.set_ylabel('loss')
plt.legend()
plt.tight_layout()
plt.show()