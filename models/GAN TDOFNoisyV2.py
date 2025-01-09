# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 18:46:22 2024

@author: Harrish Joseph
"""
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os,sys
import scipy.io as sio
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
def NormalizeData(data):                    # Normalizing function
    normdat = np.zeros((data.shape))
    for i,da in enumerate(data):
        normdat[i,:,0:2] = (da - np.min(da))/(np.max(da)-np.min(da))
    return normdat
plt.close('all')
# currentdir = os.getcwd()
currentdir = os.path.join(os.path.dirname(__file__), '../data') 
resultsdir = os.path.join(os.path.dirname(__file__), '../results')
filename   = 'Cubic2DOFp1top63500_2s.mat'
da         = sio.loadmat(f'{currentdir}/{filename}')
dam        = da['damNLNoisy']
undam      = da['undamNLNoisy']
undamNorm  = NormalizeData(undam)
damNorm    = np.zeros((dam.shape))
for j in range(dam.shape[-1]):
    damNorm[:,:,:,j]  = NormalizeData(dam[:,:,:,j])
train_data, test_data = train_test_split(undamNorm, test_size=0.2, random_state=21)
batch_size = int(undam.shape[0] / 10)
train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((test_data))
val_dataset = val_dataset.batch(batch_size)

start = time.time()
folder_name = time.strftime(f'{resultsdir}/GAN%m%d%H%MNoisy{filename}v2')
os.makedirs(folder_name, exist_ok=True)

col = ["blue", "lime", "darkviolet", "forestgreen", "darkorange", "deepskyblue", "firebrick"]
column_width_mm = 100
aspect_ratio = 4 / 3
figsize = (column_width_mm / 25.4 * 4, column_width_mm / 25.4 * 3)
damlabel = ['Undamaged','5\% Damage','10\% Damage','15\% Damage','20\% Damage','25\% Damage','30\% Damage']

latent_dim = 200
epochs = 1000
# Define Generator
def make_generator_model(latent_dim):
    latent_input = tf.keras.Input(shape=(latent_dim,), name='Latent_Input')
    fc2 = layers.Dense(125 * 1, activation='relu', name='FC2')(latent_input)
    reshaped = layers.Reshape((125, 1), name='Reshape')(fc2)
    y1 = layers.Conv1DTranspose(64, kernel_size=6, activation='relu', padding='same', strides=2, name='Decoder1')(reshaped)
    y2 = layers.Conv1DTranspose(50, kernel_size=6, activation='relu', padding='same', strides=1, name='Decoder5')(y1)
    decoded = layers.Conv1DTranspose(2, kernel_size=4, activation='sigmoid', padding='same', strides=2,
                                      name='Reconstructed')(y2)
    return tf.keras.Model(inputs=latent_input, outputs=decoded)

generator = make_generator_model(latent_dim)

generator.summary()
# Define Discriminator
def make_discriminator_model(sigLen):
    input_signal = tf.keras.Input(shape=(sigLen, 2), name='Signal_Input')
    y1 = layers.Conv1D(32, kernel_size=6, activation='relu', padding='same', strides=2, name='Disc1')(input_signal)
    y1_drop = layers.Dropout(0.1)(y1)
    y2 = layers.Conv1D(32, kernel_size=6, activation='relu', padding='same', strides=2, name='Disc2')(y1_drop)
    y2_drop = layers.Dropout(0.1)(y2)
    y3 = layers.Conv1D(32, kernel_size=6, activation='relu', padding='same', strides=2, name='Disc3')(y2_drop)
    y3_drop = layers.Dropout(0.1)(y3)
    y4 = layers.Flatten()(y3_drop)
    y5 = layers.Dense(2000, activation='relu')(y4)
    output = layers.Dense(2, activation='sigmoid')(y5)
    return tf.keras.Model(inputs=input_signal, outputs=output)

discriminator = make_discriminator_model(sigLen=train_data.shape[1])

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)#
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

class GAN(tf.keras.Model):
    def __init__(self, discriminator, generator, latent_dim, discriminator_loss_fn, generator_loss_fn):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss_fn = discriminator_loss_fn
        self.g_loss_fn = generator_loss_fn

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, o_signals):
        # Sample random points in the latent space
        batch_size = tf.shape(o_signals)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_signals = self.generator(random_latent_vectors, training=True)

            real_output = self.discriminator(o_signals, training=True)
            fake_output = self.discriminator(gen_signals, training=True)

            g_loss = self.g_loss_fn(fake_output)
            d_loss= self.d_loss_fn(real_output,fake_output)

        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # Update metrics and return their value.
        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
    def get_config(self):
        config = {
            "discriminator": tf.keras.utils.serialize_keras_object(self.discriminator),
            "generator": tf.keras.utils.serialize_keras_object(self.generator),
            "latent_dim": self.latent_dim,
            "discriminator_loss_fn": tf.keras.losses.serialize(self.d_loss_fn),
            "generator_loss_fn": tf.keras.losses.serialize(self.g_loss_fn),
        }
        base_config = super(GAN, self).get_config()
        return {"d_loss": self.d_loss_tracker.result(), "g_loss": self.g_loss_tracker.result()}

gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim,discriminator_loss_fn=discriminator_loss,
    generator_loss_fn=generator_loss)
gan.compile(
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True),#
)
# Train GAN model
d_losses = []  # List to store discriminator losses for each epoch
g_losses = [] 
start_time = time.time()
for epoch in range(epochs):
    for step, x_batch_train in enumerate(train_dataset):
        loss_value = gan.train_step(x_batch_train)
    d_losses.append(gan.d_loss_tracker.result().numpy())
    g_losses.append(gan.g_loss_tracker.result().numpy())
    train_acc = gan.d_loss_tracker.result()
    gan.d_loss_tracker.reset_state()
    gan.g_loss_tracker.reset_state()
    for x_batch_val in val_dataset:# Run a validation loop at the end of each epoch.
        gan.train_step(x_batch_val)
    val_acc = gan.d_loss_tracker.result()
    gan.d_loss_tracker.reset_state()
    gan.g_loss_tracker.reset_state()
    # print("Validation acc: %.4f" % (float(val_acc),))
    print(f'Epoch: {epoch} -- Train:{train_acc:.3f} Val:{val_acc:.3f}')
print("Time taken: %.2fs" % (time.time() - start_time))
sys.exit()
epochs =epoch+1
# Plot the discriminator and generator losses
plt.figure(figsize=figsize)################################## Loss plot
plt.plot(range(epochs), d_losses, label='Discriminator Loss')
plt.plot(range(epochs), g_losses, label='Generator Loss')
plt.title('GAN Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(folder_name, f'{filename[0:2]}Train Losses'), dpi=300)


num_samples = 9
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))
generated_signals = generator.predict(random_latent_vectors)


fig, axs = plt.subplots(3, 3, sharex=True, figsize=figsize)# Generator plot
k=0
for i, row in enumerate(axs):
    for j, ax in enumerate(row):
        ax.plot(generated_signals[k,:,0], color='r', label='Mass 1')
        ax.plot(generated_signals[k,:,1], color='b', label='Mass 2')
        k += 1
fig.suptitle("Generator data")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.text(0.5, 0.04, 'Time', ha='center')
fig.text(0.04, 0.5, 'Displacement', va='center', rotation='vertical')
fig.savefig(os.path.join(folder_name, f'{filename[0:2]}GeneratedafterTraining'), dpi=300)



