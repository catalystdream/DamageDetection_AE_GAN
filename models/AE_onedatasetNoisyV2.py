# -*- coding: utf-8 -*-
"""
Modified on Sat May 11 19:46:53 2024 For higher amplitude

@author: Harrish Joseph
"""
import time, sys
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, regularizers, utils 
from tensorflow.keras.callbacks import EarlyStopping
plt.close('all')

mat_files = [file for file in os.listdir(os.getcwd()) if file.endswith('.mat')]
mat_files.sort()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

col = ["blue","lime","darkviolet","forestgreen","darkorange","deepskyblue","firebrick"]
column_width_mm = 100
aspect_ratio    = 4 / 3
figsize         = (column_width_mm / 25.4 * 4, column_width_mm / 25.4 * 3)

damlabel = ['Undamaged','5\% Damage','10\% Damage','15\% Damage','20\% Damage','25\% Damage','30\% Damage']

def NormalizeData(data):                    # Normalizing function
    normdat = np.zeros((data.shape))
    for i,da in enumerate(data):
        for j in range(2):
            normdat[i,:,j] = (da[:,j] - np.min(da[:,j]))/(np.max(da[:,j])-np.min(da[:,j]))
    return normdat

datadir = os.path.join(os.path.dirname(__file__), '../data') 
resultsdir = os.path.join(os.path.dirname(__file__), '../results')
filename   = 'Cubic2DOFp1top63500_2s.mat'
da         = sio.loadmat(f'{datadir}/{filename}')
dam     = da['damNLNoisy']
undam   = da['undamNLNoisy']
undamNorm  = NormalizeData(undam)
damNorm = np.zeros((dam.shape))
for j in range(dam.shape[-1]):
    damNorm[:,:,:,j] = NormalizeData(dam[:,:,:,j])
    
train_data, test_data = train_test_split(undamNorm,  test_size=0.2, random_state=21)
print(f'Training: {filename}')
start = time.time()
folder_name = time.strftime(f'{resultsdir}/AE%m%d%H%MNoisy{filename[:-4]}')#[0][0]
os.makedirs(folder_name, exist_ok=True)                    # Make directory to save plots
ks,ks1,ks2 = [int(round(50/ (2 ** i), 0)) for i in range(3)]
fi,fi1,fi2 = [int(round(100/ (2 ** i), 0)) for i in range(3)]
batchsize  = int(undam.shape[0] / 10)
epoch      = 1000
latentfilters = 10
act1 = 'leaky_relu'
l2   = 0.0000001

def encoder(input_data): 
    x1 = layers.Conv1D(fi, kernel_size=ks, activation=act1, padding='same', 
                       kernel_regularizer=regularizers.l2(l2),  # l2 regularization
                       name='Encoder1')(input_data)
    x2 = layers.MaxPooling1D(2, padding='same', name='Max1')(x1)
    x3 = layers.Conv1D(fi1, kernel_size=ks1, activation=act1, padding='same', 
                        kernel_regularizer=regularizers.l2(l2),  # l2 regularization
                        name='Encoder2')(x2)
    x4 = layers.MaxPooling1D(2, padding='same', name='Max2')(x3)
    x5 =  layers.Conv1D(fi2, kernel_size=ks2, activation=act1, padding='same', 
                              kernel_regularizer=regularizers.l2(l2),  # l2 regularization
                              name='Encoder4')(x4)
    flattened = layers.Flatten(name='Flatten')(x5)
    fc1 = layers.Dense(400, activation=act1, name='FC1')(flattened)
    return fc1

def decoder(encoded):
    # Decoder
    fc2 = layers.Dense(125*latentfilters, activation=act1, name='FC2')(encoded)
    reshaped = layers.Reshape((125, latentfilters), name='Reshape')(fc2)# Reshape layer
    y1 = layers.Conv1DTranspose(fi1, kernel_size=ks2, activation=act1, padding='same', strides=2, 
                                kernel_regularizer=regularizers.l2(l2),  # l2 regularization
                                name='Decoder1')(reshaped)
    y2 = layers.Conv1DTranspose(fi, kernel_size=ks1, activation=act1, padding='same', strides=2,
                                kernel_regularizer=regularizers.l2(l2),  # l2 regularization
                                name='Decoder2')(y1)
    y3 = layers.Conv1DTranspose(fi, kernel_size=ks1, activation=act1, padding='same', strides=1,
                                kernel_regularizer=regularizers.l2(l2),  # l2 regularization
                                name='Decoder3')(y2)
    decoded = layers.Conv1DTranspose(2, kernel_size=ks, activation='sigmoid', padding='same',strides=1,  
                                    kernel_regularizer=regularizers.l2(l2),  # l2 regularization
                                    name='Reconstructed')(y3)
    return decoded

input_data  = tf.keras.Input(shape=(train_data.shape[1], 2))
autoencoder = tf.keras.Model(input_data, decoder(encoder(input_data)))
opt         = tf.keras.optimizers.Adam(learning_rate=0.001)
autoencoder.compile(optimizer=opt, loss='mae')
autoencoder.summary()

utils.plot_model(autoencoder, to_file=os.path.join(folder_name, 'autoencoder_graph.png'),show_shapes=True, expand_nested=True, dpi=150)
# sys.exit()
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True )

history=autoencoder.fit(train_data,train_data,epochs=epoch, batch_size=batchsize,  shuffle=True,
                validation_data=(test_data,test_data),callbacks=[early_stopping] )
end = time.time()
print(f'Time for training {(end - start) / 60 :.2F} minutes')
sys.exit()
plt.figure(figsize=figsize)       #figure 1 Fit and validation loss
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
plt.savefig(os.path.join(folder_name, f"{filename[0:4]}Fit and validation loss"), dpi=300)
test_predict  = autoencoder.predict(test_data)
train1        = undamNorm[0:damNorm.shape[0],:,:]
train_predict = autoencoder.predict(train1)

# Test data reconstruction plot and loss
train_loss = np.array([tf.keras.losses.mae(train_predict[:,:,0], train1[:,:,0]),
              tf.keras.losses.mae(train_predict[:,:,1], train1[:,:,1])]).T
test_loss  = np.array([tf.keras.losses.mae(test_predict[:,:,0], test_data[:,:,0]),
              tf.keras.losses.mae(test_predict[:,:,1], test_data[:,:,1])])
damageLoss = np.zeros((damNorm.shape[0],damNorm.shape[2],damNorm.shape[3]))
for k1 in range(damNorm.shape[3]):
    recon_dam = autoencoder.predict(damNorm[:,:,:,k1])#.reshape(damNorm.shape[0],damNorm.shape[1])
    damageLoss[:,0,k1] =tf.keras.losses.mae(recon_dam[:,:,0], damNorm[:,:,0,k1]).numpy()#.reshape(damNorm.shape[0],)
    damageLoss[:,1,k1] =tf.keras.losses.mae(recon_dam[:,:,1], damNorm[:,:,1,k1]).numpy()
allLoss = np.concatenate([train_loss.reshape(damageLoss.shape[0],damageLoss.shape[1],1) ,damageLoss],axis=-1)

fig, axs = plt.subplots(4, 2, sharex=True, figsize=figsize)#Figure 2 Undamage reconstruction loss
k=0
for i, row in enumerate(axs):
    for j, ax in enumerate(row):
        if j % 1== 0:
            ax.plot(train1[k,:,0], color='r', label='Original Mass 1')
            ax.plot(train_predict[k,:,0], color='b', label='Reconstructed Mass 1')
        else:
            ax.plot(train1[k,:,1], color='r', label='Original Mass 2')
            ax.plot(train_predict[k,:,1], color='b', label='Reconstructed Mass 2')
        k += 1
fig.suptitle("Undamaged train data reconstruction")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.text(0.5, 0.04, 'Time', ha='center')
fig.text(0.04, 0.5, 'Displacement', va='center', rotation='vertical')
fig.savefig(os.path.join(folder_name, f'{filename[0:4]}Train Recon'), dpi=300)

 

end2 = time.time()
print(f'Time for training {(end - start) / 60 :.2F} minutes')
print(f'Total execution time  {(end2 - start) / 60 :.2F} minutes')
