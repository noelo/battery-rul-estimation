import scipy.io
import math
import os
import ntpath
import sys
import logging
import sys
import boto3

from importlib import reload


from kfp import dsl, components
from kfp.dsl import data_passing_methods
from kfp.components import InputPath, InputTextFile, OutputPath, OutputTextFile
from kfp.components import func_to_container_op

# Download files to workspace
def loadFilesTest(infiles: str, datafiles: OutputPath()):
    import boto3
    
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

    except ClientError as err:
        print("Error: {}".format(err))
    
    
loadFilesTest_op = components.create_component_from_func(
    loadFilesTest, base_image='registry.redhat.io/rhel8/python-38:1-117',
    packages_to_install=['boto3'])
    
    
@dsl.pipeline(
  name='loadFilesTest',
  description='Download files from minio and store'
)
def download_and_store():    
    loadFilesTest_op("unibo-powertools-dataset")