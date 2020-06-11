import tensorflow as tf
import keras
import numpy as np
from keras.callbacks import ModelCheckpoint


# loading data file
def load_dataset():
    with open("data_.npy", 'rb') as f:
        data = np.load(f)

    with open("target.npy", 'rb') as f:
        target = np.load(f)

    # print(data[0], target[0])

    return data, target


# Data Modelling
def create_model(data, target):
    from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
    from keras.models import Sequential

    model = Sequential()

    model.add(Conv2D(filters=200, kernel_size=(3, 3), input_shape=data.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=100, kernel_size=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(100))
    model.add(Activation('relu'))

    model.add(Dense(target.shape[1], activation='softmax'))
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy']
    )

    return model


# create test and train datasets
def create_train_test_dataset(data, target):

    from sklearn.model_selection import train_test_split as tts
    X_train, X_test, y_train, y_test = tts(data, target, test_size=0.1, random_state=64)

    return X_train, X_test, y_train, y_test


# fitting model
def fit_and_save_model(model, X_train, X_test, y_train, y_test):
    checkpoint = ModelCheckpoint('models/model-{epoch:03d}.model', monitor='val_loss', verbose=0, save_best_only=True,
                                 mode="auto")

    print("Model Fitting: ")

    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=100, callbacks=[checkpoint],
              validation_split=0.2)
    y_pred = np.array(model.predict_classes(X_test))

    print(y_pred[:10], y_test[:10])

    # Saving model
    model_name = 'final_model'
    print("Saving Model as {}".format(model_name))

    model.save(model_name)

    print("Saved Model as {}".format(
        model_name
    ))

    return model


# main function()
def train_main():

    data, target = load_dataset()
    model = create_model(data=data, target=target)
    X_train, X_test, y_train, y_test = create_train_test_dataset(data=data, target=target)
    fit_and_save_model(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print("Done... ")


if __name__ == '__main__':
    train_main()
