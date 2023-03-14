from kfp import dsl, components
from kfp.dsl import data_passing_methods
from kfp.components import InputPath, OutputPath, OutputArtifact
from kfp.components import func_to_container_op
from kubernetes.client import V1Volume, V1SecretVolumeSource, V1VolumeMount, V1EnvVar, V1PersistentVolumeClaimVolumeSource
from typing import NamedTuple

# Download files to workspace
def readyData(preppedData: OutputPath()):
    import boto3
    import os
    import sys
    import logging
    import pickle
    from importlib import reload
    import shutil
    
# export DEFAULT_ACCESSMODES=ReadWriteOnce
# export DEFAULT_STORAGE_SIZE=5Gi
# export DEFAULT_STORAGE_CLASS=odf-lvm-vg1
    
    reload(logging)
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')    

    from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols
    # from data_processing.model_data_handler import ModelDataHandler
    # from data_processing.prepare_rul_data import RulHandler
    
    data_path = "/mnt/"    
    sys.path.append(data_path)
    print(sys.path)
    
    dataset = UniboPowertoolsData(
        test_types=[],
        chunk_size=1000000,
        lines=[37, 40],
        charge_line=37,
        discharge_line=40,
        base_path=data_path
    )
    
    with open(preppedData, "b+w") as f:   
        pickle.dump(dataset,f)    
    print('PrepData written...')
    
def fitNormaliseData(preppedData: InputPath(),modelData:OutputPath()):
    import logging
    import pickle
    from importlib import reload
    reload(logging)
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')    

    from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols
    from data_processing.model_data_handler import ModelDataHandler
    from data_processing.prepare_rul_data import RulHandler
    
    f = open(preppedData,"b+r")
    dataset = pickle.load(f)
        
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


    dataset.prepare_data(train_names, test_names)
    dataset_handler = ModelDataHandler(dataset, [
        CycleCols.VOLTAGE,
        CycleCols.CURRENT,
        CycleCols.TEMPERATURE
    ])

    rul_handler = RulHandler()

    (train_x, train_y_soh, test_x, test_y_soh,
    train_battery_range, test_battery_range,
    time_train, time_test, current_train, current_test) = dataset_handler.get_discharge_whole_cycle_future(train_names, test_names)

    train_x = train_x[:,:284,:]
    test_x = test_x[:,:284,:]

    x_norm = rul_handler.Normalization()
    train_x, test_x = x_norm.fit_and_normalize(train_x, test_x) 
    
    dataStore = [train_x, train_y_soh, test_x, test_y_soh,train_battery_range, test_battery_range,time_train, time_test, current_train, current_test]  
    
    with open(modelData, "b+w") as f:   
        pickle.dump(dataStore,f)    
    print('modelData written...')
    
def autoEncodeData(IS_TRAINING:bool, epochCount:int, modelData: InputPath(),weightsPath:OutputPath(),historyPath:OutputPath()):
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers, regularizers
    from keras.models import Model
    import logging
    import pickle
    import time
    from importlib import reload
    import pandas as pd
    
    reload(logging)
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')    

    # from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols
    # from data_processing.model_data_handler import ModelDataHandler
    # from data_processing.prepare_rul_data import RulHandler
    
    f = open(modelData,"b+r")
    dataStore = pickle.load(f)
    
    train_x=dataStore[0]
    train_y_soh=dataStore[1]
    test_x=dataStore[2]
    test_y_soh=dataStore[3]
    train_battery_range=dataStore[4]
    test_battery_range=dataStore[5]
    time_train=dataStore[6]
    time_test=dataStore[7]
    current_train=dataStore[8]
    current_test=dataStore[9]  
    
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

    if IS_TRAINING:
        history = autoencoder.fit(train_x, train_x,
                                    epochs=epochCount, 
                                    batch_size=32, 
                                    verbose=1,
                                    validation_split=0.1
                                )
        
        print("history",history)
        model_json = autoencoder.to_json()
        with open("weightsPath", "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5
        # model.save_weights("model.h5")
        print("Saved model to disk")
        # autoencoder.save_weights(data_path + 'results/trained_model/%s/model' % experiment_name)
        autoencoder.save_weights(weightsPath+"/model")

        hist_df = pd.DataFrame(history.history)
        print("hist_df",hist_df)
        # hist_csv_file = data_path + 'results/trained_model/%s/history.csv' % experiment_name
        with open(historyPath, mode='b+w') as f:
            hist_df.to_csv(f)
        history = history.history
        print("saving weights and history...done",history)
        
        
def readFiles(weights: InputPath(),history:InputPath()):
    import os  
    for x in [weights,history]:
        file_stats = os.stat(x)
        print(file_stats)
        print(f'File Size in Bytes is {file_stats.st_size}')
        print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')

    
    
readyData_op= components.create_component_from_func(
    readyData, base_image='quay.io/noeloc/batterybase',
    packages_to_install=['boto3'])

fitNormaliseData_op= components.create_component_from_func(
    fitNormaliseData, base_image='quay.io/noeloc/batterybase')

autoEncodeData_op= components.create_component_from_func(
    autoEncodeData, base_image='quay.io/noeloc/batterybase')
    
readFiles_op= components.create_component_from_func(
    readFiles, base_image='quay.io/noeloc/batterybase')    
    
@dsl.pipeline(
  name='batteryTestPipeline',
  description='Download files from minio and store'
)
def batteryTestPipeline():   
    vol = V1Volume(
        name='batterydatavol',
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
            claim_name='batterydata',)
        )
    res = readyData_op().add_pvolumes({"/mnt": vol})
    prep = fitNormaliseData_op(res.output)
    model = autoEncodeData_op(True,1,prep.output)
    readFiles_op(model.outputs["weightsPath"],model.outputs["historyPath"])
    
    
    # .add_pod_label('pipelines.kubeflow.org/cache_enabled', 'false')
    # readFiles_op(prep.output).add_pod_annotation(name="tekton.dev/output_artifacts", value=([])).add_pod_annotation(name="tekton.dev/artifact_items", value=([])).add_pod_label('pipelines.kubeflow.org/cache_enabled', 'false')
    
    # print('Pipeline Completed...',x.outputs['p1'])
if __name__ == '__main__':
    from kfp_tekton.compiler import TektonCompiler
    compiler = TektonCompiler()
    compiler.produce_taskspec = False
    compiler.compile(batteryTestPipeline, __file__.replace('.py', '.yaml'))
    


    