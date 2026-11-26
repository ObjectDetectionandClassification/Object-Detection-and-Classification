from keras.models import Sequential
from keras.layers import *
import numpy as np
from Evaluation import evaluation


def VGG_16(HN, weights_path=None, num_of_class=None):
    model = Sequential()
    model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=int(HN), activation="relu"))  # 4096
    model.add(Dense(units=int(HN), activation="relu"))  # 4096
    model.add(Dense(units=num_of_class, activation="softmax"))
    weight = np.asarray(model.get_weights()[30])
    return model


def Model_VGG16(train_data, train_target, test_data, test_target, BS=None, HN=None):
    if HN is None:
        HN = 25
    if BS is None:
        BS = 4
    IMG_SIZE = [32, 32, 3]

    Train_X = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Train_X[i, :] = np.resize(train_data[i], (IMG_SIZE[0],  IMG_SIZE[1], IMG_SIZE[2]))
    Train_X = Train_X.reshape(Train_X.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Test_X[i, :] = np.resize(test_data[i], (IMG_SIZE[0],  IMG_SIZE[1], IMG_SIZE[2]))
    Test_X = Test_X.reshape(Test_X.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    model = VGG_16(HN, num_of_class=train_target.shape[1])
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.fit(x=Train_X, y=train_target, epochs=1, batch_size=BS, steps_per_epoch=1)
    pred = model.predict(Test_X)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_target)
    return Eval, pred