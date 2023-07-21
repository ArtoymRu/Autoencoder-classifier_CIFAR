from keras.layers import Input, Dense
from keras import Model
from keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential


def clf(x_train_encoded):
    #Define classifier
    inputs = Input(shape=(x_train_encoded.shape[1],))
    x = Dense(256, activation='relu')(inputs)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs, outputs)

def clf_train(model, encoder_name, latent_dim, dataset_name, x_train, y_train, x_test, y_test):

    autoencoder = model(latent_dim=latent_dim)

    if encoder_name == 'Var_AE':
        encoder = Model(autoencoder.encoder.input, autoencoder.encoder.output[2])
    else:
        encoder = Model(autoencoder.input, autoencoder.layers[-7].output)

    autoencoder.load_weights(f'../models/{encoder_name}.{latent_dim}.{dataset_name}.keras')

    # Convert labels to categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Generate latent representations
    x_train_latent = encoder.predict(x_train)
    x_test_latent = encoder.predict(x_test)

    # Define the classifier model
    classifier = Sequential([
        Dense(256, activation='relu', input_shape=(x_train_latent.shape[1],)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    classifier.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    history = classifier.fit(x_train_latent, y_train, epochs=10, validation_data=(x_test_latent, y_test))
    classifier.save_weights(f'../models/clf.{encoder_name}.{latent_dim}.{dataset_name}.keras')

    # Get the predicted classes for the test set
    y_pred = classifier.predict(x_test_latent)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Print classification report
    report = classification_report(np.argmax(y_test, axis=1), y_pred_classes) # updated to decode categorical y_test
    print(f'Classification Report for {encoder_name} ({latent_dim}):\n{report}')

    # Plot confusion matrix
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred_classes) # updated to decode categorical y_test
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {encoder_name} ({latent_dim})')
    plt.colorbar()
    tick_marks = np.arange(10)  # 10 for CIFAR10, 100 for CIFAR100
    plt.xticks(tick_marks, range(10), rotation=45)
    plt.yticks(tick_marks, range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    return classifier, history
