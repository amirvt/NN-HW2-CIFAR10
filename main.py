import os
import pickle
import numpy as np
import keras
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import resample, shuffle

save_dir = 'hists'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

plot_dir = 'plots'
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

num_classes = 10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

conv_dims_default = [32, 64, 128]
filters_default = 2
stride_default = 3
pool_default = 2
dense_dim_default = 512
dropout_default = 0.
active_default = 'relu'
batch_size_default = 64
epochs_default = 100


# %%

def build_model(conv_dims=conv_dims_default, filters=filters_default, stride=stride_default,
                pool=pool_default, active=active_default, dense_dim=dense_dim_default, dropout=dropout_default,
                opt_func=keras.optimizers.adam, loss=keras.losses.categorical_crossentropy):
    _model = Sequential()

    for i, dim in enumerate(conv_dims):
        for j in range(filters):
            if i == 0 and j == 0:
                _model.add(Conv2D(dim, (stride, stride),
                                  padding='same', input_shape=x_train.shape[1:]))
            else:
                _model.add(Conv2D(dim, (stride, stride), padding='same'))
            _model.add(Activation(active))

        _model.add(MaxPool2D(pool_size=(pool, pool)))
        _model.add(Dropout(dropout))

    _model.add(Flatten())
    _model.add(Dense(dense_dim))
    _model.add(Activation(active))
    _model.add(Dropout(dropout))
    _model.add(Dense(num_classes))
    _model.add(Activation('softmax'))

    optimizer = opt_func(1e-4, 1e-6)
    _model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return _model


def train(_model, _x_train, _y_train,
          batch_size=batch_size_default, epochs=epochs_default, verbose=1):
    _hist = _model.fit(_x_train, _y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       validation_data=(x_test, y_test),
                       shuffle=True,
                       verbose=verbose)
    return _hist


def save_hist(file_name, history):
    with open(save_dir + '/' + file_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


# %% 4B


model = build_model()
hist = train(model, x_train, y_train, verbose=2)

save_hist('4b', hist)

# %% 4C - compare active functions over 10 epochs
actives = ['sigmoid', 'tanh']
for active in actives:
    model = build_model(active=active)
    hists_4c = (
        train(model, x_train, y_train)
    )
    save_hist('4c-' + active, hists_4c)

# %% 4D - compare ADAM and SGD over 10 epochs
# noinspection PyTypeChecker
model = build_model(opt_func=keras.optimizers.SGD)
hist_4d = train(model, x_train, y_train)
save_hist('4d-sgd', hist_4d)

# %% 4E - reduce train size to 600

samples = [resample(x_train[np.argmax(y_train, 1) == i], y_train[np.argmax(y_train, 1) == i], n_samples=600)
           for i in range(10)]
x_list, y_list = list(map(list, zip(*samples)))
x_train_small, y_train_small = np.vstack(x_list), np.vstack(y_list)
x_train_small, y_train_small = shuffle(x_train_small, y_train_small)
model = build_model()
hist_small = train(model, x_train_small, y_train_small, epochs=epochs_default)
save_hist('4e', hist_small)


# %% 4E - reduce train size to 600

x_train_small, y_train_small = resample(x_train, y_train, n_samples=600)
model = build_model()
hist_small = train(model, x_train_small, y_train_small, epochs=epochs_default)
save_hist('4e', hist_small)

# %% 4f - dropout and data augmention

# dropouts = [0.1, 0.2, 0.3, 0.4, 0.6]
dropouts = [0.4]
for dropout in dropouts:
    model = build_model(dropout=dropout)
    hists_4f = train(model, x_train, y_train, epochs=100, verbose=2)
    save_hist('4f-' + str(dropout), hists_4f)

# %% 4f - augment
model = build_model()
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

# Compute quantities required for feature-wise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(x_train)
# %%
# Fit the model on the batches generated by datagen.flow().
hists_4f2 = model.fit_generator(datagen.flow(x_train, y_train,
                                             batch_size=batch_size_default),
                                steps_per_epoch=len(x_train) / batch_size_default,
                                epochs=100,
                                validation_data=(x_test, y_test),
                                workers=4)
save_hist('4f2', hists_4f2)

# %%
model = build_model(active='relu', dropout=0.25)
asd = model.fit_generator(datagen.flow(x_train, y_train,
                                       batch_size=batch_size_default),
                          steps_per_epoch=len(x_train) / batch_size_default,
                          epochs=1000,
                          validation_data=(x_test, y_test),
                          workers=4,
                          verbose=2)

# Epoch 464/500 drp .25
#  - 16s - loss: 0.2887 - acc: 0.8975 - val_loss: 0.4006 - val_acc: 0.8717
save_hist('final4', asd)
