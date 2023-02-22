# %% [markdown]
# # Autoencoder UNIBO Powertools Dataset

# %%
import numpy as np
import pandas as pd
import scipy.io
import math
import os
import ntpath
import sys
import logging
import time
import sys
import random
import boto3

from importlib import reload
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model

from kfp import dsl, components
from kfp.dsl import data_passing_methods
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op

import pickle

IS_TRAINING = False
RESULT_NAME = ""
IS_OFFLINE = True

if IS_OFFLINE:
    import plotly.offline as pyo
    pyo.init_notebook_mode()   

data_path = "../../"

reload(logging)
logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')

sys.path.append(data_path)
print(sys.path)
from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols
from data_processing.model_data_handler import ModelDataHandler
from data_processing.prepare_rul_data import RulHandler

# %% [markdown]
# ### Config logging

# %%


# %% [markdown]
# # Load Data

# %%
dataset = UniboPowertoolsData(
    test_types=[],
    chunk_size=1000000,
    lines=[37, 40],
    charge_line=37,
    discharge_line=40,
    base_path=data_path
)

# %%
train_names = [
    '000-DM-3.0-4019-S',#minimum capacity 1.48
    '001-DM-3.0-4019-S',#minimum capacity 1.81
    '002-DM-3.0-4019-S',#minimum capacity 2.06
    '009-DM-3.0-4019-H',#minimum capacity 1.41
    '010-DM-3.0-4019-H',#minimum capacity 1.44
    '014-DM-3.0-4019-P',#minimum capacity 1.7
    '015-DM-3.0-4019-P',#minimum capacity 1.76
    '016-DM-3.0-4019-P',#minimum capacity 1.56
    '017-DM-3.0-4019-P',#minimum capacity 1.29
    #'047-DM-3.0-4019-P',#new 1.98
    #'049-DM-3.0-4019-P',#new 2.19
    '007-EE-2.85-0820-S',#2.5
    '008-EE-2.85-0820-S',#2.49
    '042-EE-2.85-0820-S',#2.51
    '043-EE-2.85-0820-H',#2.31
    '040-DM-4.00-2320-S',#minimum capacity 3.75, cycles 188
    '018-DP-2.00-1320-S',#minimum capacity 1.82
    #'019-DP-2.00-1320-S',#minimum capacity 1.61
    '036-DP-2.00-1720-S',#minimum capacity 1.91
    '037-DP-2.00-1720-S',#minimum capacity 1.84
    '038-DP-2.00-2420-S',#minimum capacity 1.854 (to 0)
    '050-DP-2.00-4020-S',#new 1.81
    '051-DP-2.00-4020-S',#new 1.866 
]

test_names = [
    '003-DM-3.0-4019-S',#minimum capacity 1.84
    '011-DM-3.0-4019-H',#minimum capacity 1.36
    '013-DM-3.0-4019-P',#minimum capacity 1.6
    '006-EE-2.85-0820-S',# 2.621    
    '044-EE-2.85-0820-H',# 2.43
    '039-DP-2.00-2420-S',#minimum capacity 1.93
    '041-DM-4.00-2320-S',#minimum capacity 3.76, cycles 190
]

# %%
dataset.prepare_data(train_names, test_names)
dataset_handler = ModelDataHandler(dataset, [
    CycleCols.VOLTAGE,
    CycleCols.CURRENT,
    CycleCols.TEMPERATURE
])

rul_handler = RulHandler()

with open('dataset.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)


# %% [markdown]
# ## Data preparation

# %%
(train_x, train_y_soh, test_x, test_y_soh,
  train_battery_range, test_battery_range,
  time_train, time_test, current_train, current_test) = dataset_handler.get_discharge_whole_cycle_future(train_names, test_names)

train_x = train_x[:,:284,:]
test_x = test_x[:,:284,:]
print("cut train shape {}".format(train_x.shape))
print("cut test shape {}".format(test_x.shape))


x_norm = rul_handler.Normalization()
train_x, test_x = x_norm.fit_and_normalize(train_x, test_x)

# %% [markdown]
# # Model training

# %%
if IS_TRAINING:
    EXPERIMENT = "autoencoder_unibo_powertools"

    experiment_name = time.strftime("%Y-%m-%d-%H-%M-%S") + '_' + EXPERIMENT
    print(experiment_name)

# Model definition

opt = tf.keras.optimizers.Adam(learning_rate=0.0002)
LATENT_DIM = 10

class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(train_x.shape[1], train_x.shape[2])),
            #layers.MaxPooling1D(5, padding='same'),
            layers.Conv1D(filters=16, kernel_size=5, strides=2, activation='relu', padding='same'),
            layers.Conv1D(filters=8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Flatten(),
            layers.Dense(self.latent_dim, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(self.latent_dim)),
            layers.Dense(568, activation='relu'),
            layers.Reshape((71, 8)),
            layers.Conv1DTranspose(filters=8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv1DTranspose(filters=16, kernel_size=5, strides=2, activation='relu', padding='same'),
            layers.Conv1D(3, kernel_size=3, activation='relu', padding='same'),
            #layers.UpSampling1D(5),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(LATENT_DIM)
autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
autoencoder.encoder.summary()
autoencoder.decoder.summary()

# %%
if IS_TRAINING:
    history = autoencoder.fit(train_x, train_x,
                                epochs=500, 
                                batch_size=32, 
                                verbose=1,
                                validation_split=0.1
                               )

# %%
if IS_TRAINING:
    autoencoder.save_weights(data_path + 'results/trained_model/%s/model' % experiment_name)

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = data_path + 'results/trained_model/%s/history.csv' % experiment_name
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    history = history.history

# %%
if not IS_TRAINING:
    history = pd.read_csv(data_path + 'results/trained_model/%s/history.csv' % RESULT_NAME)
    autoencoder.load_weights(data_path + 'results/trained_model/%s/model' % RESULT_NAME)
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

# %%
if not IS_TRAINING:
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(history)

# %% [markdown]
# ### Testing

# %%
results = autoencoder.evaluate(test_x, test_x, return_dict = True)
print(results)
max_rmse = 0
for index in range(test_x.shape[0]):
    result = autoencoder.evaluate(np.array([test_x[index, :, :]]), np.array([test_x[index, :, :]]), return_dict = True, verbose=0)
    max_rmse = max(max_rmse, result['rmse'])
print("Max rmse: {}".format(max_rmse))

# %% [markdown]
# # Results Visualization

# %%
fig = go.Figure()
fig.add_trace(go.Scatter(y=history['loss'],
                    mode='lines', name='train'))
if 'val_loss' in history:
    fig.add_trace(go.Scatter(y=history['val_loss'],
                    mode='lines', name='validation'))
fig.update_layout(title='Loss trend',
                  xaxis_title='epoch',
                  yaxis_title='loss',
                  width=1400,
                  height=600)
fig.show()

# %%
train_predictions = autoencoder.predict(train_x)
labels = ['Voltage', 'Current', 'Temperature']

# %%
for i in range(train_x.shape[2]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_predictions[0,:,i],
                        mode='lines', name='predicted'))
    fig.add_trace(go.Scatter(y=train_x[0,:,i],
                        mode='lines', name='actual'))
    fig.update_layout(title='Results on training - battery new',
                    xaxis_title='Step',
                    yaxis_title=labels[i],
                    width=1400,
                    height=600)
    fig.show()

# %%
for i in range(train_x.shape[2]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_predictions[int(train_battery_range[0]/2),:,i],
                        mode='lines', name='predicted'))
    fig.add_trace(go.Scatter(y=train_x[int(train_battery_range[0]/2),:,i],
                        mode='lines', name='actual'))
    fig.update_layout(title='Results on training - middle life',
                    xaxis_title='Step',
                    yaxis_title=labels[i],
                    width=1400,
                    height=600)
    fig.show()

# %%
for i in range(train_x.shape[2]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_predictions[train_battery_range[0]-1,:,i],
                        mode='lines', name='predicted'))
    fig.add_trace(go.Scatter(y=train_x[train_battery_range[0]-1,:,i],
                        mode='lines', name='actual'))
    fig.update_layout(title='Results on training - end of life',
                    xaxis_title='Step',
                    yaxis_title=labels[i],
                    width=1400,
                    height=600)
    fig.show()

# %%
test_predictions = autoencoder.predict(test_x)
labels = ['Voltage', 'Current', 'Temperature']

# %%
for i in range(train_x.shape[2]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=test_predictions[0,:,i],
                        mode='lines', name='predicted'))
    fig.add_trace(go.Scatter(y=test_x[0,:,i],
                        mode='lines', name='actual'))
    fig.update_layout(title='Results on testing - battery new',
                    xaxis_title='Step',
                    yaxis_title=labels[i],
                    width=1400,
                    height=600)
    fig.show()

# %%
for i in range(train_x.shape[2]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=test_predictions[int(test_battery_range[0]/2),:,i],
                        mode='lines', name='predicted'))
    fig.add_trace(go.Scatter(y=test_x[0,:,i],
                        mode='lines', name='actual'))
    fig.update_layout(title='Results on testing - middle life',
                    xaxis_title='Step',
                    yaxis_title=labels[i],
                    width=1400,
                    height=600)
    fig.show()

# %%
for i in range(train_x.shape[2]):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=test_predictions[test_battery_range[0]-1,:,i],
                        mode='lines', name='predicted'))
    fig.add_trace(go.Scatter(y=test_x[0,:,i],
                        mode='lines', name='actual'))
    fig.update_layout(title='Results on testing - end of life',
                    xaxis_title='Step',
                    yaxis_title=labels[i],
                    width=1400,
                    height=600)
    fig.show()