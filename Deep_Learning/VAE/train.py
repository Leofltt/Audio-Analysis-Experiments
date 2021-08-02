from VAE import VAE
import numpy as np
import os


LR = 0.0005
BATCH_S = 64
N_EPOCHS = 150
SPECS_PATH =  "/Users/leofltt/Desktop/Audio-Analysis-Experiments/Deep_Learning/VAE/spectrograms/"

def load_FSDD(spectrogram_path):
    x_train = []
    file_paths = []
    for root, _, file_names in os.walk(spectrogram_path):
        for file in file_names:
            file_path = os.path.join(root, file)
            spec = np.load(file_path) # (bins, frames, 1)
            x_train.append(spec)
            file_paths.append(file_path)
    x_train = np.array(x_train)
    x_train = x_train[[...,np.newaxis]]
    return x_train, file_paths

def train(x_train, learning_rate, batch_size, num_epochs):
    autoencoder = VAE(
        input_shape=(256,64,1),
        conv_filters=(512,256,128,64,32),
        conv_kernel=(3,3,3,3,3),
        conv_strides=(2,2,2,2,(2,1)),
        latent_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train,batch_size,num_epochs)
    return autoencoder

if __name__ == "__main__":
    x_train, _ = load_FSDD(SPECS_PATH)
    # x_train = x_train[:,:,:,np.newaxis]
    ae = train(x_train,LR,BATCH_S,N_EPOCHS)
    ae.save("model")
    ae_2 = VAE.load("model")
    ae_2.summary()