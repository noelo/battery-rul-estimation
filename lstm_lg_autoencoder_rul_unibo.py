#!/usr/bin/env python
# coding: utf-8

# # RUL estimation UNIBO Powertools Dataset
import kfp
from kfp import dsl, components
from kfp.dsl import data_passing_methods
from kfp.components import InputPath, OutputPath, OutputArtifact
from kfp.components import func_to_container_op
from kubernetes.client import V1Volume, V1SecretVolumeSource, V1VolumeMount, V1EnvVar, V1PersistentVolumeClaimVolumeSource
from typing import NamedTuple
from os import environ as env
import time

def readyData(data_path:str,epoch_count:int, modelData:OutputPath(),parameterData:OutputPath()):
    import shutil
    import numpy as np
    import pandas as pd
    import sys
    import logging
    import time
    import sys
    # import random

    from importlib import reload
    import pickle
    from tensorflow import keras
    from keras import layers, regularizers
    from keras.models import Model
    from keras import backend as K
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Activation, TimeDistributed, Input, Concatenate
    from keras.optimizers import Adam
    from keras.layers import LSTM, Masking

    RESULT_NAME = "2023-02-13-11-54-47_lstm_autoencoder_rul_unibo_powertools"
    EXPERIMENT = "lstm_autoencoder_rul_unibo_powertools"

    #data_path = "../../"
    # data_path="/opt/data/pitstop/"

    sys.path.append(data_path)
    from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols
    from data_processing.model_data_handler import ModelDataHandler
    from data_processing.prepare_rul_data import RulHandler


    # ### Config logging
    reload(logging)
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.INFO, datefmt='%Y/%m/%d %H:%M:%S')


    # # Load Data

    dataset = UniboPowertoolsData(
        test_types=[],
        chunk_size=1000000,
        lines=[37, 40],
        charge_line=37,
        discharge_line=40,
        base_path=data_path
    )

    ################################################NOC

    train_names = [
        '000-DM-3.0-4019-S'#minimum capacity 1.81
    ]

    test_names = [
        '003-DM-3.0-4019-S']


    dataset.prepare_data(train_names, test_names)
    dataset_handler = ModelDataHandler(dataset, [
        CycleCols.VOLTAGE,
        CycleCols.CURRENT,
        CycleCols.TEMPERATURE
    ])

    rul_handler = RulHandler()

    # # Data preparation

    CAPACITY_THRESHOLDS = {
    3.0 : 2.7,#th 90% - min 2.1, 70%
    2.85 : 2.7,#th 94.7% - min 2.622, 92%
    2.0 : 1.93,#th 96.5% - min 1.93, 96.5%
    4.0 : 3.77,#th 94.2% - min 3.77 94.2%
    4.9 : 4.7,#th 95.9% - min 4.3, 87.7%
    5.0 : 4.5#th 90% - min 3.63, 72.6%
    }

    (train_x, train_y_soh, test_x, test_y_soh,
    train_battery_range, test_battery_range,
    time_train, time_test, current_train, current_test) = dataset_handler.get_discharge_whole_cycle_future(train_names, test_names, min_cycle_length=300)

    train_y = rul_handler.prepare_y_future(train_names, train_battery_range, train_y_soh, current_train, time_train, CAPACITY_THRESHOLDS)
    # del globals()["current_train"]
    # del globals()["time_train"]
    test_y = rul_handler.prepare_y_future(test_names, test_battery_range, test_y_soh, current_test, time_test, CAPACITY_THRESHOLDS)
    # del globals()["current_test"]
    # del globals()["time_test"]

    x_norm = rul_handler.Normalization()
    x_norm.fit(train_x)

    train_x = x_norm.normalize(train_x)
    test_x = x_norm.normalize(test_x)
    
    AUTOENCODER_WEIGHTS = '2023-02-09-15-50-22_autoencoder_gl_unibo_powertools'
    N_CYCLE = 500
    WARMUP_TRAIN = 15
    WARMUP_TEST = 30

    opt = keras.optimizers.Adam(learning_rate=0.0002)
    LATENT_DIM = 10

    class Autoencoder(Model):
        def __init__(self, latent_dim):
            super(Autoencoder, self).__init__()
            self.latent_dim = latent_dim

            encoder_inputs = layers.Input(shape=(train_x.shape[1], train_x.shape[2]))
            encoder_conv1 = layers.Conv1D(filters=8, kernel_size=10, strides=2, activation='relu', padding='same')(encoder_inputs)
            encoder_pool1 = layers.MaxPooling1D(5, padding='same')(encoder_conv1)
            encoder_conv2 = layers.Conv1D(filters=8, kernel_size=4, strides=1, activation='relu', padding='same')(encoder_pool1)
            encoder_pool2 = layers.MaxPooling1D(3, padding='same')(encoder_conv2)
            encoder_flat1 = layers.Flatten()(encoder_pool1)
            encoder_flat2 = layers.Flatten()(encoder_pool2)
            encoder_concat = layers.concatenate([encoder_flat1, encoder_flat2])
            encoder_outputs = layers.Dense(self.latent_dim, activation='relu')(encoder_concat)
            self.encoder = Model(inputs=encoder_inputs, outputs=encoder_outputs)

            decoder_inputs = layers.Input(shape=(self.latent_dim,))
            decoder_dense1 = layers.Dense(10*8, activation='relu')(decoder_inputs)
            decoder_reshape1 = layers.Reshape((10, 8))(decoder_dense1)
            decoder_upsample1 = layers.UpSampling1D(3)(decoder_reshape1)
            decoder_convT1 = layers.Conv1DTranspose(filters=8, kernel_size=4, strides=1, activation='relu', padding='same')(decoder_upsample1)
            decoder_upsample2 = layers.UpSampling1D(5)(decoder_convT1)
            decoder_convT2 = layers.Conv1DTranspose(filters=8, kernel_size=10, strides=2, activation='relu', padding='same')(decoder_upsample2)
            decoder_outputs = layers.Conv1D(3, kernel_size=3, activation='relu', padding='same')(decoder_convT2)
            self.decoder = Model(inputs=decoder_inputs, outputs=decoder_outputs)



        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    autoencoder = Autoencoder(LATENT_DIM)

    autoencoder.compile(optimizer=opt, loss='mse', metrics=['mse', 'mae', 'mape', keras.metrics.RootMeanSquaredError(name='rmse')])
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()
    autoencoder.load_weights(data_path + 'results/trained_model/%s/model' % AUTOENCODER_WEIGHTS)
    # compression
    train_x = autoencoder.encoder(train_x).numpy()
    test_x = autoencoder.encoder(test_x).numpy()
    print("compressed train x shape {}".format(train_x.shape))
    print("compressed test x shape {}".format(test_x.shape))
    test_x = test_x[:,~np.all(train_x == 0, axis=0)]#we need same column number of training
    train_x = train_x[:,~np.all(train_x == 0, axis=0)]
    print("compressed train x shape without zero column {}".format(train_x.shape))
    print("compressed test x shape without zero column {}".format(test_x.shape))


    x_norm = rul_handler.Normalization()
    x_norm.fit(train_x)
    train_x = x_norm.normalize(train_x)
    test_x = x_norm.normalize(test_x)
    train_x = rul_handler.battery_life_to_time_series(train_x, N_CYCLE, train_battery_range)
    test_x = rul_handler.battery_life_to_time_series(test_x, N_CYCLE, test_battery_range)
    train_x, train_y, train_battery_range, train_y_soh = rul_handler.delete_initial(train_x, train_y, train_battery_range, train_y_soh, WARMUP_TRAIN)
    test_x, test_y, test_battery_range, test_y_soh = rul_handler.delete_initial(test_x, test_y, test_battery_range, test_y_soh, WARMUP_TEST)

    # first one is SOH, we keep only RUL
    train_y = train_y[:,1]
    test_y = test_y[:,1]

    # # Y normalization
    y_norm = rul_handler.Normalization()
    y_norm.fit(train_y)
    train_y = y_norm.normalize(train_y)
    test_y = y_norm.normalize(test_y)  

    experiment_name = time.strftime("%Y-%m-%d-%H-%M-%S") + '_' + EXPERIMENT
    print(experiment_name)

    opt = keras.optimizers.Adam(lr=0.000003)
    model = Sequential()
    model.add(Masking(input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(LSTM(128, activation='tanh',
                    return_sequences=True,
                    kernel_regularizer=regularizers.l2(0.0002)))
    model.add(LSTM(64, activation='tanh', return_sequences=False,
                    kernel_regularizer=regularizers.l2(0.0002)))
    model.add(Dense(64, activation='selu', kernel_regularizer=regularizers.l2(0.0002)))
    model.add(Dense(32, activation='selu', kernel_regularizer=regularizers.l2(0.0002)))
    model.add(Dense(1, activation='linear'))
    model.summary()

    model.compile(optimizer=opt, loss='huber', metrics=['mse', 'mae', 'mape', keras.metrics.RootMeanSquaredError(name='rmse')])

    history = model.fit(train_x, train_y, 
                                epochs=epoch_count, 
                                batch_size=32, 
                                verbose=2,
                                validation_split=0.1
                            )

    model.save('/tmp/results/trained_model/%s.h5' % experiment_name)

    hist_df = pd.DataFrame(history.history)
    hist_csv_file = '/tmp/results/trained_model/%s_history.csv' % experiment_name
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    history = history.history
    
    #package up model and store
    shutil.make_archive('/tmp/%store' % experiment_name, "tar", "/tmp/results/")       
    shutil.copy('/tmp/%store' % experiment_name+".tar",modelData)
    print("Model data stored")
    
    dataStore = [train_x, train_y, train_battery_range, train_y_soh, test_x, test_y, test_battery_range, test_y_soh, y_norm, experiment_name,]  
    
    with open(parameterData, "b+w") as f:   
        pickle.dump(dataStore,f)    
    print('Parameter data written...')
    
    
def evaluateModel(modelData:InputPath(),paramaterData:InputPath(),outputImages:OutputPath()):
    import os  
    import shutil
    import pickle
    import numpy as np
    import pandas as pd
    from importlib import reload
    import plotly.graph_objects as go

    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, regularizers
    from keras import backend as K
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Activation, TimeDistributed, Input, Concatenate
    from keras.optimizers import Adam
    from keras.layers import LSTM, Masking
    
    f = open(paramaterData,"b+r")
    dataStore = pickle.load(f)
    train_x=dataStore[0]
    train_y=dataStore[1]
    train_battery_range=dataStore[2]
    train_y_soh=dataStore[3]
    test_x=dataStore[4]
    test_y=dataStore[5]
    test_battery_range=dataStore[6]
    test_y_soh=dataStore[7]
    y_norm=dataStore[8]
    experiment_name=dataStore[9]
    

    shutil.unpack_archive(modelData,"/tmp/model/","tar")
    res = []
    for (dir_path, dir_names, file_names) in os.walk("/tmp/model/"):
        res.extend(dir_path)
        res.extend(dir_names)
        res.extend(file_names)
    print(res)
    history = pd.read_csv('/tmp/model/trained_model/%s_history.csv' % experiment_name)
    model = keras.models.load_model('/tmp/model/trained_model/%s.h5' % experiment_name)
    model.summary(expand_nested=True)
    
    if not os.path.exists("/tmp/result-images"):
        os.mkdir("/tmp/result-images")
    
   
    
    # if not IS_TRAINING:
    #     history = pd.read_csv(data_path + 'results/trained_model/%s_history.csv' % RESULT_NAME)
    #     model = keras.models.load_model(data_path + 'results/trained_model/%s.h5' % RESULT_NAME)
    #     model.summary(expand_nested=True)

    # if not IS_TRAINING:
    #     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #         print(history)


    # ### Testing
    results = model.evaluate(test_x, test_y, return_dict = True)
    print(results)
    max_rmse = 0
    for index in range(test_x.shape[0]):
        result = model.evaluate(np.array([test_x[index, :, :]]), np.array([test_y[index]]), return_dict = True, verbose=0)
        max_rmse = max(max_rmse, result['rmse'])
    print("Max rmse: {}".format(max_rmse))


    # # Results Visualization
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
    fig.write_image("/tmp/result-images/fig1.jpeg")

    train_predictions = model.predict(train_x)

    train_y = y_norm.denormalize(train_y)
    train_predictions = y_norm.denormalize(train_predictions)


    a = 0
    for b in train_battery_range:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_y_soh[a:b], y=train_predictions[a:b,0],
                            mode='lines', name='predicted'))
        fig.add_trace(go.Scatter(x=train_y_soh[a:b], y=train_y[a:b],
                            mode='lines', name='actual'))
        fig.update_layout(title='Results on training',
                        xaxis_title='SoH Capacity',
                        yaxis_title='Remaining Ah until EOL',
                        xaxis={'autorange':'reversed'},
                        width=1400,
                        height=600)
        # fig.show()
        fig.write_image("/tmp/result-images/fig2.jpeg")
        a = b


    a = 0
    for b in train_battery_range:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=train_predictions[a:b,0],
                            mode='lines', name='predicted'))
        fig.add_trace(go.Scatter(y=train_y[a:b],
                            mode='lines', name='actual'))
        fig.update_layout(title='Results on training',
                        xaxis_title='Cycle',
                        yaxis_title='Remaining Ah until EOL',
                        width=1400,
                        height=600)
        # fig.show()
        fig.write_image("/tmp/result-images/fig3.jpeg")
        
        a = b


    test_predictions = model.predict(test_x)

    test_y = y_norm.denormalize(test_y)
    test_predictions = y_norm.denormalize(test_predictions)


    a = 0
    for b in test_battery_range:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_y_soh[a:b], y=test_predictions[a:b,0],
                            mode='lines', name='predicted'))
        fig.add_trace(go.Scatter(x = test_y_soh[a:b], y=test_y[a:b],
                            mode='lines', name='actual'))
        fig.update_layout(title='Results on testing',
                        xaxis_title='SoH Capacity',
                        yaxis_title='Remaining Ah until EOL',
                        xaxis={'autorange':'reversed'},
                        width=1400,
                        height=600)
        fig.write_image("/tmp/result-images/fig4.jpeg")
        a = b


    a = 0
    for b in test_battery_range:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=test_predictions[a:b, 0],
                            mode='lines', name='predicted'))
        fig.add_trace(go.Scatter(y=test_y[a:b],
                            mode='lines', name='actual'))
        fig.update_layout(title='Results on testing',
                        xaxis_title='Cycle',
                        yaxis_title='Remaining Ah until EOL',
                        width=1400,
                        height=600)
        fig.write_image("/tmp/result-images/fig4.jpeg")
        a = b
    
    shutil.make_archive('/tmp/%s_images' % experiment_name, "tar", "/tmp/result-images/")       
    shutil.copy('/tmp/%s_images' % experiment_name+".tar",outputImages)
    print("Image data stored")
    
def readFiles(modeldata: InputPath()):
    import os  
    import shutil
    
    shutil.unpack_archive(modeldata,"/tmp/model/","tar")
    res = []
    for (dir_path, dir_names, file_names) in os.walk("/tmp/model"):
        res.extend(file_names)
    print(res)
   
    
readyData_op= components.create_component_from_func(
    readyData, base_image='quay.io/noeloc/batterybase')

evaluateModel_op= components.create_component_from_func(
    evaluateModel, base_image='quay.io/noeloc/batterybase',packages_to_install=['kaleido'])
  
readFiles_op= components.create_component_from_func(
    readFiles, base_image='quay.io/noeloc/batterybase')    
    
@dsl.pipeline(
  name='batteryTestPipeline',
  description='Download files from minio and store'
)
def batteryTestPipeline():  
    dsl.VolumeOp 
    vol = V1Volume(
        name='batterydatavol',
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
            claim_name='batterydata',)
        )
    res = readyData_op(data_path="/opt/data/pitstop/",epoch_count=2).add_pvolumes({"/opt/data": vol})
    images = evaluateModel_op(res.outputs["modelData"],res.outputs["parameterData"]).add_pod_annotation(name="tekton.dev/output_artifacts", value=("output.outputImages"))
    # readFiles_op(images.output)

if __name__ == '__main__':
    from kfp_tekton.compiler import TektonCompiler
    from kfp_tekton.compiler import pipeline_utils
    env.setdefault("DEFAULT_STORAGE_CLASS","odf-lvm-vg1")
    env.setdefault("DEFAULT_ACCESSMODES","ReadWriteOnce")
    env.setdefault("DEFAULT_STORAGE_SIZE","10Gi")
    compiler = TektonCompiler()
    pipeline_conf = pipeline_utils.TektonPipelineConf()
    pipeline_conf.add_pipeline_annotation("tekton.dev/track_artifact", 'true')
    compiler.produce_taskspec = False
    compiler.compile(batteryTestPipeline, __file__.replace('.py', '.yaml'),tekton_pipeline_conf=pipeline_conf)  