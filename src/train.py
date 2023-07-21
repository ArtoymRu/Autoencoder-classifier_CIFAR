from keras.optimizers.legacy import Adam
from keras.losses import MeanSquaredError
from keras.callbacks import ModelCheckpoint

def ae_train(ae, encoder_name, dataset_name, x_train, x_test, latent_dim):
    '''
    Train loop for autoencoders
    Arguments:
        ae: model
        x_train: dataframe
        x_test: dataframe
    Return:
        train history
    '''
    ae.compile(optimizer=Adam(), loss=MeanSquaredError())
    history =  ae.fit(x_train, x_train, epochs=20, batch_size=64, 
      shuffle=True, validation_data=(x_test, x_test))

    ae.save_weights(f'../models/{encoder_name}.{latent_dim}.{dataset_name}.keras')

    return history


def denoising_ae_train(autoencoder, train_noise, train, test_noise, test):
    '''
    Train loop for denoising autoencoder
    Arguments:
    Return:
    '''
    autoencoder.compile(loss='mse', optimizer='adam') 
    checkpoint = ModelCheckpoint('../models/denoising_ae_model.h5', verbose=1, save_best_only=True, save_weights_only=True) 
    autoencoder.fit(train_noise, train, validation_data=(test_noise, test), epochs=40, batch_size=128, callbacks=[checkpoint])