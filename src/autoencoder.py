from keras.layers import Conv2D, Input, Dense, Flatten, Conv2DTranspose, Activation, BatchNormalization, ReLU, Concatenate , MaxPooling2D, UpSampling2D, Lambda, Reshape
from keras import backend as K
from keras import Model
import numpy as np


def Conv_AE(latent_dim=20):
    '''
    Just a simple convolutional autoencoder architecture
    Arguments:
        latent_dim: int
    Return:
        autoencoder: model
    '''
    input_img = Input(shape=(32, 32, 3))

    # Encoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    volume_size = K.int_shape(x)
    x = Flatten()(x)
    encoded = Dense(latent_dim)(x)

    # Decoder
    x = Dense(np.prod(volume_size[1:]))(encoded)
    x = Reshape((volume_size[1], volume_size[2], volume_size[3]))(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    return autoencoder


class Var_AE(Model):
    '''
    Variational autoencoder architecture
    Arguments:
        latent_dim: int
    Return:
        several methods to build and call the model
    '''
    def __init__(self, latent_dim=20, **kwargs):
        super(Var_AE, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        inputs = Input(shape=(32, 32, 3))
        x = Flatten()(inputs)
        x = Dense(400, activation='relu')(x)
        z_mean = Dense(self.latent_dim)(x)
        z_log_var = Dense(self.latent_dim)(x)
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        return Model(inputs, [z_mean, z_log_var, z])
    
    def build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,))
        x = Dense(400, activation='relu')(latent_inputs)
        x = Dense(32*32*3, activation='sigmoid')(x)
        outputs = Reshape((32,32,3))(x)
        return Model(latent_inputs, outputs)
    
    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0, stddev=1.)
        return z_mean + K.exp(z_log_var/2)*epsilon
    
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructured = self.decoder(z)
        kl_loss = self.get_kl_loss(z_mean, z_log_var)
        self.add_loss(kl_loss)
        return reconstructured

    def get_kl_loss(self, z_mean, z_log_var):
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(kl_loss) / (32 * 32 * 3)


# For the futher research
class Denoising_AE:
    '''
    Variational autoencoder architecture
    Arguments:
        latent_dim: int
    Return:
        several methods to build and call the model
    '''
    def conv_block(x, filters, kernel_size, strides=2): 
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x) 
        x = BatchNormalization()(x)
        x = ReLU()(x) 
        return x

    def deconv_block(x, filters, kernel_size): 
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding='same')(x) 
        x = BatchNormalization()(x)
        x = ReLU()(x)
        return x

    def denoising_autoencoder(self, conv_block, deconv_block): 
        dae_inputs = Input(shape=(32, 32, 3), name='dae_input') 
        conv_block1 = conv_block(dae_inputs, 32, 3) 
        conv_block2 = conv_block(conv_block1, 64, 3) 
        conv_block3 = conv_block(conv_block2, 128, 3) 
        conv_block4 = conv_block(conv_block3, 256, 3) 
        conv_block5 = conv_block(conv_block4, 256, 3, 1)
        deconv_block1 = deconv_block(conv_block5, 256, 3) 
        merge1 = Concatenate()([deconv_block1, conv_block3]) 
        deconv_block2 = deconv_block(merge1, 128, 3) 
        merge2 = Concatenate()([deconv_block2, conv_block2]) 
        deconv_block3 = deconv_block(merge2, 64, 3) 
        merge3 = Concatenate()([deconv_block3, conv_block1]) 
        deconv_block4 = deconv_block(merge3, 32, 3) 
        final_deconv = Conv2DTranspose(filters=3, kernel_size=3, padding='same')(deconv_block4)
        dae_outputs = Activation('sigmoid', name='dae_output')(final_deconv) 
        return Model(dae_inputs, dae_outputs, name='dae')