from kfp import dsl, components
from kfp.dsl import data_passing_methods
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
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
    from data_processing.model_data_handler import ModelDataHandler
    from data_processing.prepare_rul_data import RulHandler
    
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
    import boto3
    import os
    import sys
    import logging
    import pickle
    from importlib import reload
    import shutil
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

    # %%
    dataset.prepare_data(train_names, test_names)
    dataset_handler = ModelDataHandler(dataset, [
        CycleCols.VOLTAGE,
        CycleCols.CURRENT,
        CycleCols.TEMPERATURE
    ])

    rul_handler = RulHandler()
    # %%
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
    
def autoEncodeData(modelData: InputPath()):
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.models import Model
    import logging
    import pickle
    from importlib import reload
    reload(logging)
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')    

    from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols
    from data_processing.model_data_handler import ModelDataHandler
    from data_processing.prepare_rul_data import RulHandler
    
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
    
    # print("cut train shape {}".format(train_x.shape))
    # print("cut test shape {}".format(test_x.shape))
  
    
    
readyData_op= components.create_component_from_func(
    readyData, base_image='quay.io/noeloc/batterybase',
    packages_to_install=['boto3'])

fitNormaliseData_op= components.create_component_from_func(
    fitNormaliseData, base_image='quay.io/noeloc/batterybase',
    packages_to_install=['boto3'])

autoEncodeData_op= components.create_component_from_func(
    autoEncodeData, base_image='quay.io/noeloc/batterybase',
    packages_to_install=['boto3'])


def readFiles(infiles: InputPath())-> NamedTuple('taskOutput', [('p1', str)]):
    import os  
    import shutil
    file_stats = os.stat(infiles)
    print(file_stats)
    print(f'File Size in Bytes is {file_stats.st_size}')
    print(f'File Size in MegaBytes is {file_stats.st_size / (1024 * 1024)}')
    from collections import namedtuple
    task_output = namedtuple('taskOutput', ['p1'])
    return task_output("1")
    
readFiles_op= components.create_component_from_func(
    readFiles, base_image='registry.access.redhat.com/ubi8/python-38')
    
    
@dsl.pipeline(
  name='loadFilesTest',
  description='Download files from minio and store'
)
def download_and_store():   
    vol = V1Volume(
        name='batterydatavol',
        persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
            claim_name='batterydata',)
        )
    res = readyData_op().add_pvolumes({"/mnt": vol})
    prep = fitNormaliseData_op(res.output)
    autoEncodeData_op(prep.output)
    
    # .add_pod_label('pipelines.kubeflow.org/cache_enabled', 'false')
    # readFiles_op(prep.output).add_pod_annotation(name="tekton.dev/output_artifacts", value=([])).add_pod_annotation(name="tekton.dev/artifact_items", value=([])).add_pod_label('pipelines.kubeflow.org/cache_enabled', 'false')
    
    # print('Pipeline Completed...',x.outputs['p1'])
    


    