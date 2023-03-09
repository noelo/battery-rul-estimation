from kfp import dsl, components
from kfp.dsl import data_passing_methods
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op
from kubernetes.client import V1Volume, V1SecretVolumeSource, V1VolumeMount, V1EnvVar, V1PersistentVolumeClaimVolumeSource
from typing import NamedTuple

# Download files to workspace
def writeFiles(preppedData: OutputPath()):
    import boto3
    import os
    import sys
    import logging
    import pickle
    from importlib import reload
    import shutil
    
# export DEFAULT_ACCESSMODES=ReadWriteOnce
# export DEFAULT_STORAGE_SIZE=2Gi
# export DEFAULT_STORAGE_CLASS=kfp-csi-s3
    
    # reload(logging)
    # logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', level=logging.DEBUG, datefmt='%Y/%m/%d %H:%M:%S')    
        
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
        # client.Bucket('battery-data').download_file( 'unibo-powertools-dataset/README.md', datafiles)
        print("Starting downloading file....done")
        basepath = '/mnt'
        with os.scandir(basepath) as entries:
            for entry in entries:
                print(entry.name)                     
                
    except ClientError as err:
        print("Error: {}".format(err))
        
    # IS_TRAINING = False
    # RESULT_NAME = ""
    # IS_OFFLINE = True

    # # if IS_OFFLINE:
    # #     import plotly.offline as pyo
    # #     pyo.init_notebook_mode()   


    # from data_processing.unibo_powertools_data import UniboPowertoolsData, CycleCols
    # from data_processing.model_data_handler import ModelDataHandler
    # from data_processing.prepare_rul_data import RulHandler
    
    # data_path = "/mnt/"    
    # sys.path.append(data_path)
    # print(sys.path)
    
    # dataset = UniboPowertoolsData(
    #     test_types=[],
    #     chunk_size=1000000,
    #     lines=[37, 40],
    #     charge_line=37,
    #     discharge_line=40,
    #     base_path=data_path
    # )
    
    # with open(preppedData, "b+w") as f:   
    #     pickle.dump(dataset,f)
    
    # shutil.copyfile("/mnt/large_bin_file.dat",preppedData)
    ba = bytearray(os.urandom(1_000_000))
    
    with open(preppedData, "b+w") as f:   
        f.write(ba)
    
    print('Files written...')
    
    
    
writeFiles_op= components.create_component_from_func(
    writeFiles, base_image='registry.access.redhat.com/ubi8/python-38',
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
    res = writeFiles_op().add_pvolumes({"/mnt": vol}).add_pod_label('pipelines.kubeflow.org/cache_enabled', 'false')
    readFiles_op(res.output).add_pod_annotation(name="tekton.dev/output_artifacts", value=([])).add_pod_annotation(name="tekton.dev/artifact_items", value=([])).add_pod_label('pipelines.kubeflow.org/cache_enabled', 'false')
    
    # print('Pipeline Completed...',x.outputs['p1'])
    


    