import logging

import click
import h5py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras as k
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Permute,
)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2

logging.basicConfig(level=logging.INFO,)

LOGGER = logging.getLogger()


def scaleAndSplit(data, labels):
    # data = data - np.mean(data) # not needed?
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    num_classes = 2
    y_train = k.utils.to_categorical(y_train, num_classes)
    y_test = k.utils.to_categorical(y_test, num_classes)
    return X_train, X_test, y_train, y_test


def buildModel(
    num_input_channels=24,
    num_timesteps=20,
    num_filters=20,
    num_Dense=200,
    drop_rate=0.0,
    reg=None,
):
    kernel_size = (3, 3, 3)
    num_classes = 2
    ac = "relu"
    opt = Adam(lr=0.001, decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    inp = Input(
        (num_timesteps, num_input_channels, 8, 8)
    )  # timesteps, channels, rows, columns
    permuted = Permute((1, 3, 4, 2))(inp)  # expects channels_last
    conv1 = Conv3D(
        num_filters, kernel_size, activation=ac, kernel_regularizer=reg, padding="valid"
    )(permuted)
    conv2 = Conv3D(
        num_filters, kernel_size, activation=ac, kernel_regularizer=reg, padding="valid"
    )(conv1)
    conv3 = Conv3D(
        num_filters, kernel_size, activation=ac, kernel_regularizer=reg, padding="valid"
    )(conv2)
    conv4 = Conv3D(
        num_filters, kernel_size, activation=ac, kernel_regularizer=reg, padding="valid"
    )(conv3)
    flat = Flatten()(conv4)
    dense = Dense(num_Dense)(flat)
    dropout = Dropout(drop_rate)(dense)
    out = Dense(num_classes, activation="softmax")(dropout)

    model = Model(inputs=[inp], outputs=[out])

    model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=opt)
    LOGGER.info("build model!")
    LOGGER.info(model.summary())
    return model


def trainModel(model, X_train, y_train, X_test, y_test):
    # early_stopping = EarlyStopping(monitor="val_loss", patience=3)
    # early_stopping = EarlyStopping(monitor="loss", patience=3)

    hist = model.fit(
        X_train,
        y_train,
        batch_size=128,
        epochs=20,
        validation_data=(X_test, y_test),
        # callbacks=[early_stopping,],
    )


@click.command()
@click.option(
    "--input-path", help="input array of training data (piece positions)", required=True
)
@click.option(
    "--input-path-attacks",
    help="input array of training data (attacked squares)",
    required=True,
)
@click.option("--input-path-labels", help="input array of labels", required=True)
@click.option("--output-path", help="where to save the model", required=True)
def main(input_path, input_path_labels, input_path_attacks, output_path):
    # data dims: (samples, time, channel, row, col)
    data_positions = np.load(input_path)["arr_0"]
    data_attacks = np.load(input_path_attacks)["arr_0"]
    data = np.concatenate((data_positions, data_attacks), axis=2)
    labels = np.load(input_path_labels)["arr_0"]
    LOGGER.info(f"Training Conv3D model on data of shape {data.shape}")

    model = buildModel(num_input_channels=data.shape[2], num_timesteps=data.shape[1])

    X_train, X_test, y_train, y_test = scaleAndSplit(data, labels)
    trainModel(model, X_train, y_train, X_test, y_test)
    model.save(output_path)


if __name__ == "__main__":
    main()
