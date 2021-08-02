import numpy as np
import os
import pickle 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K 

class Autoencoder:
    '''
    Deep Convolutional Autoencoder
    '''

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernel,
                 conv_strides,
                 latent_dim
                 ):
        '''
        inputs
        input_shape : shape of input array
        conv_filters : number of filters for each layer
        conv_kernel : kernel size for each layer
        conv_strides : stride value for each layer
        latent_dim : number of latent space dimensions
        '''
        
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernel = conv_kernel
        self.conv_strides = conv_strides
        self.latent_dim = latent_dim

        self.encoder = None 
        self.decoder = None 
        self.model = None 

        self._conv_layers = len(conv_filters)
        self._unflat_shape = None
        self._model_input = None

        self.__build()
    
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()
    
    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer,loss=mse_loss)
    
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True
                       )
    
    def save(self, path="."):
        self._directory_check(path)
        self._save_parameters(path)
        self._save_weights(path)
    
    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations
    
    @classmethod
    def load(self, path="."):
        params_path = os.path.join(path, "parameters.pkl")
        weights_path = os.path.join(path, "weights.h5")
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        autoencoder = Autoencoder(*params)
        autoencoder._load_weigths(weights_path)
        return autoencoder

    # ======= Build Autoencoder Methods =====================
    
    def __build(self):
        '''
        Build the autoencoder model
        '''
        self._build_encoder()
        self._build_decoder()
        self._build_ae()
    
    def _build_ae(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input,model_output,name="autoencoder")

    def _build_encoder(self):
        '''
        Build the encoder model
        '''
        encoder_input = self._add_enc_input()
        conv_layers = self._add_conv_layers(encoder_input)
        encoder_output = self._add_enc_output(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input,encoder_output, name='encoder')
    
    def _build_decoder(self):
        '''
        Build the decoder model
        '''
        decoder_input = self._add_dec_input()
        dense_layer = self._add_dec_dense(decoder_input)
        reshape_layer = self._add_reshape(dense_layer)
        conv_transpose_layers = self._add_conv_transpose(reshape_layer)
        decoder_output = self._add_dec_output(conv_transpose_layers)
        self.decoder = Model(decoder_input,decoder_output, name='decoder')
        
    # ======= Encoder Methods ============================================

    def _add_enc_input(self):
        return Input(shape=self.input_shape, name='encoder_input')
    
    def _add_conv_layers(self, x):
        '''
        Create convolution blocks of the encoder
        '''
        for i in range(self._conv_layers):
            x = self._add_conv_layer(i,x)
        return x
    
    def _add_conv_layer(self, layer_idx, layers):
        '''
        Add conv2D  + ReLU + batch norm to 'layers' graph
        '''
        conv_layer = Conv2D(filters=self.conv_filters[layer_idx],
                            kernel_size=self.conv_kernel[layer_idx],
                            strides=self.conv_strides[layer_idx],
                            padding='same',
                            name=f"encoder_conv_layer_{layer_idx+1}"
                            )
        x = conv_layer(layers)
        x = ReLU(name=f"encoder_relu_{layer_idx+1}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_idx+1}")(x)
        return x 
    
    def _add_enc_output(self, layers):
        '''
        Flatten data and add Dense layer to 'layers' graph
        '''
        self._unflat_shape = K.int_shape(layers)[1:] # shape before flat, ignore batch size

        x = Flatten()(layers)
        x = Dense(self.latent_dim, name="encoder_output")(x)
        return x 
    
    # ======= Decoder Methods ===================================

    def _add_dec_input(self):
        return Input(shape=self.latent_dim, name="decoder_input")
    
    def _add_dec_dense(self, decoder_input):
        num_neurons = np.prod(self._unflat_shape)
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return(dense_layer)
    
    def _add_reshape(self, dense_layer):
        reshape_layer = Reshape(self._unflat_shape, name="reshape_layer")(dense_layer)
        return reshape_layer
    
    def _add_conv_transpose(self, x):
        '''
        Add Convolutional Transpose blocks
        In the reverse order of the encoder
        '''
        for i in reversed(range(1, self._conv_layers)):
            x = self._add_conv_trans_layer(i,x)
        return x 
    
    def _add_conv_trans_layer(self, layer_idx, x):
        '''
        Add conv2DTranspose + ReLU + batch norm to 'x' graph
        '''
        layer_num = self._conv_layers - layer_idx
        
        conv_transpose = Conv2DTranspose(filters=self.conv_filters[layer_idx],
                                         kernel_size=self.conv_kernel[layer_idx],
                                         strides=self.conv_strides[layer_idx],
                                         padding='same',
                                         name=f"decoder_conv_transpose_{layer_num}"
                                        )
        x = conv_transpose(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x 
    
    def _add_dec_output(self, layers):
        conv_transpose_final = Conv2DTranspose(filters=1,
                                         kernel_size=self.conv_kernel[0],
                                         strides=self.conv_strides[0],
                                         padding='same',
                                         name=f"decoder_conv_transpose_{self._conv_layers}"
                                        )
        x = conv_transpose_final(layers)
        output_layer = Activation("sigmoid", name="sigmoid_output_layer")(x)
        return output_layer
    
    # =================================

    def _directory_check(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
    
    def _save_parameters(self, path):
        params = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernel,
            self.conv_strides,
            self.latent_dim
            ]
        save_path = os.path.join(path, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(params, f)
    
    def _save_weights(self, path):
        save_path = os.path.join(path, "weights.h5") 
        self.model.save_weights(save_path)
    
    def _load_weigths(self, path):
        self.model.load_weights(path)


if __name__ == "__main__":
    autoencoder = VAE(
        input_shape=(28,28,1),
        conv_filters=(32,64,64,64),
        conv_kernel=(3,3,3,3),
        conv_strides=(1,2,2,1),
        latent_dim=2
    )
    autoencoder.summary()

