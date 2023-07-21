from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Input

# Load CIFAR10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# Normalizing the RGB codes by dividing it to the max RGB value
x_train, x_test = x_train / 255.0, x_test / 255.0

# Autoencoder settings
input_img = Input(shape=(32, 32, 3))  
# Encoder
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoder')(x)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional
# Decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Defining the model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))

# Now, let's train the classifier using latent representations
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoder').output)
# Reshape the encoder output to 1D
flattened = Flatten()(encoder.output)
# Add the final Dense layer for classification. Let's assume the classes are 10
dense = Dense(10, activation='softmax')(flattened)
classifier = Model(encoder.input, dense)
# Compile the model
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Training
classifier.fit(x_train, y_train,
               epochs=50,
               batch_size=128,
               shuffle=True,
               validation_data=(x_test, y_test))

# Evaluate the model
scores = classifier.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])