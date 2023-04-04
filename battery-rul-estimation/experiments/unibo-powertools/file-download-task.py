import os
from kfp import dsl, components
from kfp.components import InputPath, OutputPath
from kubernetes.client import V1Volume, V1EnvVar, V1PersistentVolumeClaimVolumeSource
from kfp_tekton.k8s_client_helper import env_from_secret

# Download files to workspace
def ready_data(data_path:str,prepped_data: OutputPath()):
    '''prepare model data'''
    import sys
    import logging
    import pickle
    from importlib import reload
    from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols

    reload(logging)
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')

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

    with open(prepped_data, "b+w") as f:
        pickle.dump(dataset,f)
    print('PrepData written...')

def fit_normalise_data(prepped_data: InputPath(),model_data:OutputPath()):
    '''normalise data'''
    import logging
    import pickle
    from importlib import reload
    from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols
    from data_processing.model_data_handler import ModelDataHandler
    from data_processing.prepare_rul_data import RulHandler
    
    reload(logging)
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')    
    f = open(prepped_data,"b+r")
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
    data_store = [train_x, train_y_soh, test_x, test_y_soh,train_battery_range, test_battery_range,time_train, time_test, current_train, current_test]
    with open(model_data, "b+w") as f:
        pickle.dump(data_store,f)
    print('modelData written...')
    
def auto_encode_data(is_training:bool, epoch_count:int, model_data: InputPath(),weightspath:OutputPath(),historypath:OutputPath()):
    '''train or inference model'''
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
    f = open(model_data,"b+r")
    data_store = pickle.load(f)

    train_x=data_store[0]
    train_y_soh=data_store[1]
    test_x=data_store[2]
    test_y_soh=data_store[3]
    train_battery_range=data_store[4]
    test_battery_range=data_store[5]
    time_train=data_store[6]
    time_test=data_store[7]
    current_train=data_store[8]
    current_test=data_store[9]  

    if is_training:
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

    if is_training:
        history = autoencoder.fit(train_x, train_x,
                                    epochs=epoch_count,
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
        autoencoder.save_weights(weightspath+"/model")

        hist_df = pd.DataFrame(history.history)
        print("hist_df",hist_df)
        # hist_csv_file = data_path + 'results/trained_model/%s/history.csv' % experiment_name
        with open(historypath, mode='b+w') as f:
            hist_df.to_csv(f)
        history = history.history
        print("saving weights and history...done",history)

def load_trigger_data(data_file:str,bucket_details:str,file_destination:str):
    '''load data file passed from cloud event into relevant location'''
    import boto3
    import os

    endpoint_url=os.environ["s3_host"]
    aws_access_key_id=os.environ["s3_access_key"]
    aws_secret_access_key=os.environ["s3_secret_access_key"]
    print(endpoint_url,aws_access_key_id, aws_secret_access_key)

    s3_target = boto3.resource('s3',
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=None,
        config=boto3.session.Config(signature_version='s3v4'),
        verify=False
    )

    with open(file_destination+data_file, 'wb') as f:
        s3_target.meta.client.download_fileobj(bucket_details, data_file, f)

def read_files(weights_path: InputPath(),history_path:InputPath()):
    '''print file info for debugging'''
    import os  
    for x in [weights_path,history_path]:
        file_stats = os.stat(x)
        print(file_stats)
        print(f'File Size in Bytes is {file_stats.st_size}')
        print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')

ready_data_op= components.create_component_from_func(
    ready_data, base_image='quay.io/noeloc/batterybase',
    packages_to_install=['boto3'])

fit_normalise_data_op= components.create_component_from_func(
    fit_normalise_data, base_image='quay.io/noeloc/batterybase')

auto_encode_data_op= components.create_component_from_func(
    auto_encode_data, base_image='quay.io/noeloc/batterybase')

read_files_op= components.create_component_from_func(
    read_files, base_image='quay.io/noeloc/batterybase')

load_trigger_data_op= components.create_component_from_func(
    load_trigger_data, base_image='quay.io/noeloc/batterybase',
    packages_to_install=['boto3'])

@dsl.pipeline(
  name='batterytest-pipeline',
  description='Download files from s3, train, inference'
)
def batterytest_pipeline(file_obj:str, src_bucket:str):
    '''Download files from s3, train, inference'''
    print("Params",file_obj, src_bucket)
    vol = V1Volume(
        name='batterydatavol',
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
            claim_name='batterydata',)
        )
    ## /mnt/data/unibo-powertools-dataset/unibo-powertools-dataset
    file_destination = "/mnt/data/unibo-powertools-dataset/unibo-powertools-dataset/"
    trigger_data = load_trigger_data_op(file_obj, src_bucket,file_destination).add_pvolumes({"/mnt": vol})
    trigger_data.add_env_variable(V1EnvVar(name='s3_host', value='http://rook-ceph-rgw-ceph-object-store.openshift-storage.svc:8080'))
    trigger_data.add_env_variable(env_from_secret('s3_access_key', 's3-secret', 'AWS_ACCESS_KEY_ID'))
    trigger_data.add_env_variable(env_from_secret('s3_secret_access_key', 's3-secret', 'AWS_SECRET_ACCESS_KEY'))

    res = ready_data_op("/mnt/").after(trigger_data).add_pvolumes({"/mnt": vol})

    prep = fit_normalise_data_op(res.output)

    model = auto_encode_data_op(True,1,prep.output)

    read_files_op(model.outputs["weightspath"],model.outputs["historypath"])

if __name__ == '__main__':
    from kfp_tekton.compiler import TektonCompiler
    os.environ.setdefault("DEFAULT_STORAGE_CLASS","managed-csi")
    os.environ.setdefault("DEFAULT_ACCESSMODES","ReadWriteOnce")
    os.environ.setdefault("DEFAULT_STORAGE_SIZE","10Gi")
    compiler = TektonCompiler()
    compiler.produce_taskspec = False
    compiler.compile(batterytest_pipeline, __file__.replace('.py', '.yaml'))