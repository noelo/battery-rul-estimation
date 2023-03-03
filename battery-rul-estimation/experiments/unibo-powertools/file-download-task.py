import scipy.io
import math
import os
import ntpath
import sys
import logging

from importlib import reload


from kfp import dsl, components
from kfp.dsl import data_passing_methods
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op
from kubernetes.client import V1Volume, V1SecretVolumeSource, V1VolumeMount, V1EnvVar, V1PersistentVolumeClaimVolumeSource

# Download files to workspace
def loadFilesTest(infiles: str, datafiles: OutputPath(),preppedData: OutputPath()):
    import boto3
    import os
    import sys
    import logging
    import pickle
    from importlib import reload
    
    reload(logging)
    logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')    
        
    clientArgs = {
        'aws_access_key_id': 'minio',
        'aws_secret_access_key': 'minio123',
        'endpoint_url': 'http://minio-api.minio-dev.svc.cluster.local:9000',
        'verify': False
    }

    client = boto3.resource("s3", **clientArgs)
    
    try:
        print('Retrieving buckets...')
        print()

        for bucket in client.buckets.all():
            bucket_name = bucket.name
            print('Bucket name: {}'.format(bucket_name))

            objects = client.Bucket(bucket_name).objects.all()

            for obj in objects:
                object_name = obj.key

                print('Object name: {}'.format(object_name))

            print()

        print("Starting downloading file")
        client.Bucket('battery-data').download_file( 'unibo-powertools-dataset/README.md', datafiles)
        print("Starting downloading file....done")
        basepath = '/mnt'
        with os.scandir(basepath) as entries:
            for entry in entries:
                print(entry.name)                     
                
    except ClientError as err:
        print("Error: {}".format(err))
        
    IS_TRAINING = False
    RESULT_NAME = ""
    IS_OFFLINE = True

    # if IS_OFFLINE:
    #     import plotly.offline as pyo
    #     pyo.init_notebook_mode()   


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
    
    
    
loadFilesTest_op = components.create_component_from_func(
    loadFilesTest, base_image='quay.io/noeloc/batterybase',
    packages_to_install=['boto3'])
    
    
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
    x = loadFilesTest_op("unibo-powertools-dataset").add_pvolumes({"/mnt": vol})

    