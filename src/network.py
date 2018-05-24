import os
import numpy as np

from keras.models import Sequential
from keras.layers import (Dense, Conv2D, BatchNormalization, MaxPooling2D,
                          Dropout, Flatten)
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from keras.layers import LSTM
from keras.layers import Reshape
from keras.layers import TimeDistributed

from dataFilesIO import DataFiles
from preprocessing import Preprocessor
from helper import Constants
from helper import Logging

co = Constants()
io = DataFiles()
train_preprocessor = Preprocessor("dataset_scaled_train")
test_preprocessor = Preprocessor("dataset_scaled_test")
logger = Logging()


def load_data(set_name):
    if "train" in set_name:
      preprocessor = train_preprocessor
    elif "test" in set_name:
      preprocessor = test_preprocessor
    else:
      print("Set name error in function load_data!")
      exit()
    if preprocessor.data_exists():
        set = np.load(os.path.join(co.DATA_ROOT, '{}.npz'.format(set_name)))
    else:
        data_location = preprocessor.transform_data()
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
    model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=input_shape,
              #kernel_regularizer=l2_reg,
              bias_regularizer=l2_reg,
              activation='relu'))
    # Applies batch norm to each frame
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D())

    return model


train_x, train_y, _ = load_data('dataset_scaled_train')
print('origi: ', np.shape(train_x), np.shape(train_y))
train_x = train_preprocessor.add_noise(train_x)
train_x = np.array(train_x)
train_y = np.concatenate((train_y, train_y), axis=0)
print('noisy: ', np.shape(train_x), np.shape(train_y))

testOrg_x, testOrg_y, _ = load_data('dataset_scaled_test')
print('origi: ', np.shape(testOrg_x), np.shape(testOrg_y))
testOrg_x = test_preprocessor.add_noise(testOrg_x)
testOrg_x = np.array(testOrg_x)
testOrg_y = np.concatenate((testOrg_y, testOrg_y), axis=0)
print('noisy: ', np.shape(testOrg_x), np.shape(testOrg_y))

numVal = 151

idx = np.arange(len(testOrg_x))
np.random.shuffle(idx)
valInds = np.random.choice(idx, size=numVal, replace=False)
assert(len(valInds) == numVal)

testInds = []
for i in idx:
  if i not in valInds:
    testInds.append(i)
assert(len(testInds) == len(testOrg_x) - numVal)

val_x = testOrg_x[valInds]
val_y = testOrg_y[valInds]

test_x = testOrg_x[testInds]
test_y = testOrg_y[testInds]

# Shuffle the data randomly
idx = np.arange(len(train_x))
np.random.shuffle(idx)
train_x = train_x[idx]
train_y = train_y[idx]

# Turn targets to categorical, for use with categorical_crossentropy
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)
val_y = to_categorical(val_y)

# Pad the input, so all the samples have PAD_SIZE frames
padded_train = pad_sequences(train_x, maxlen=co.PAD_SIZE, padding="post")
# Add a 4th dimension, to act as the channel dimension, akin to images
padded_train = np.expand_dims(padded_train, axis=3)

input_shape = np.shape(padded_train)[1:]

# shape of padded_train = (batch, PAD_SIZE, 128, 1)

padded_test = pad_sequences(test_x, maxlen=co.PAD_SIZE, padding="post")
padded_test = np.expand_dims(padded_test, axis=3)

padded_val = pad_sequences(val_x, maxlen=co.PAD_SIZE, padding="post")
padded_val = np.expand_dims(padded_val, axis=3)

l2_reg = l2(co.L2_REG_RATE)

model = Sequential()
model = add_full_convolutional_layer(model)
model = add_full_convolutional_layer(model)
model = add_full_convolutional_layer(model)
model = add_full_convolutional_layer(model)
model.add(Dropout(rate=co.DROPOUT_RATE))

shape = model.layers[-1].output_shape

model.add(Reshape((shape[3], shape[1] * shape[2])))
model.add(LSTM(128, return_sequences=True, input_shape = (shape[3], co.PAD_SIZE, shape[1] * shape[2]), bias_regularizer=l2_reg))
model.add(LSTM(128, return_sequences=True, bias_regularizer=l2_reg))
model.add(Dropout(rate=co.DROPOUT_RATE))
model.add(Flatten())

model.add(Dense(units=co.OUTPUT_SIZE, activation='softmax'))

plot_model(model, show_shapes=True,
           to_file=os.path.join(co.FIG_ROOT, 'model.png'))

optimizer = Adam(lr=co.LEARNING_RATE)
#optimizer = SGD(lr=co.LEARNING_RATE, decay=5e-1, momentum=0.99, nesterov=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(patience=5)

hist = model.fit(padded_train, train_y, batch_size=co.BATCH_SIZE,
                 epochs=co.EPOCHS, validation_data=(padded_val, val_y),
                 callbacks=[early_stopping])

loss_and_metrics = model.evaluate(padded_test, test_y, batch_size=co.BATCH_SIZE)

print("Test statistics:\n - Loss: {a}\n - Accuracy: {b}".format(a = loss_and_metrics[0], b = loss_and_metrics[1]))

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
