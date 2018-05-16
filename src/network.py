import os
import numpy as np

from keras.models import Sequential
from keras.layers import (Dense, Conv2D, BatchNormalization, MaxPooling2D,
                          Dropout, Flatten)
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import plot_model

from dataFilesIO import DataFiles
from preprocessing import Preprocessor
from helper import Constants
from helper import Logging

co = Constants()
io = DataFiles()
preprocessor = Preprocessor()
logger = Logging()


def load_data(set_name):
    if preprocessor.data_exists():
        set = np.load(os.path.join(co.DATA_ROOT, '{}.npz'.format(set_name)))
    else:
        data_location = preprocessor.transform_data(set_name)
        set = np.load(data_location)

    return (set['X'], set['Y'], set['Y_string'])


def store_model(model):
    already_stored = os.listdir(co.MODELS_ROOT)
    if len(already_stored) != 0:
        # Split model name, get last part #.h5, extract only #
        ids = [int(model_name.split('_')[-1][:-3])
               for model_name in already_stored]

        model_name = 'model_{}.h5'.format(np.amax(ids)+1)
    else:
        model_name = 'model_{}.h5'.format('0')

    model.save(os.path.join(co.MODELS_ROOT, model_name))
    return os.path.join(co.MODELS_ROOT, model_name)


def add_full_convolutional_layer(model):
    model.add(Conv2D(filters=48, kernel_size=(3, 3), input_shape=input_shape,
              bias_regularizer=l2_reg,
              activation='relu'))
    # Applies batch norm to each frame
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D())
    model.add(Dropout(rate=co.DROPOUT_RATE))

    return model


train_x, train_y, accents = load_data('train')

# Turn targets to categorical, for use with categorical_crossentropy
train_y = to_categorical(train_y)

# Pad the input, so all the samples have PAD_SIZE frames
padded_input = pad_sequences(train_x, maxlen=co.PAD_SIZE)
# Add a 4th dimension, to act as the channel dimension, akin to images
padded_input = np.expand_dims(padded_input, axis=3)

# shape of padded_input = (batch, PAD_SIZE, 128, 1)
input_shape = np.shape(padded_input)[1:]

l2_reg = l2(co.L2_REG_RATE)

model = Sequential()
model = add_full_convolutional_layer(model)
model = add_full_convolutional_layer(model)

model.add(Flatten())
model.add(Dense(units=256, activation='relu', bias_regularizer=l2_reg))
model.add(Dropout(rate=co.DROPOUT_RATE))
model.add(Dense(units=co.OUTPUT_SIZE, activation='softmax'))

plot_model(model, show_shapes=True,
           to_file=os.path.join(co.FIG_ROOT, 'model.png'))

optimizer = Adam(lr=co.LEARNING_RATE)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(patience=2)

hist = model.fit(padded_input, train_y, batch_size=co.BATCH_SIZE,
                 epochs=co.EPOCHS, validation_split=0.1,
                 callbacks=[early_stopping])

model_location = store_model(model)

entry = {
    'model_location': model_location,
    'val_loss': hist.history['val_loss'],
    'loss': hist.history['loss'],
    'val_acc': hist.history['val_acc'],
    'acc': hist.history['acc'],
    # 'test_loss': evaluation[0],
    # 'test_acc': evaluation[1],
    'epochs_trained': len(hist.history['loss']),
    'training_params': co.net_params_to_dictionary(),
}

logger.store_log_entry(entry)
