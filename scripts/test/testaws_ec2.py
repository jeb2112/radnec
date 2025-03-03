# demo script to create EC2 instance and try to run a training

import os
import boto3


credentials = boto3.Session().get_credentials()

os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
os.environ["AWS_SESSION_TOKEN"] = credentials.token

import boto3

region = 'central'  
job_id = 'my-experiment' # replace with unique id  
num_instances = 1  
image_id = 'ami-0240b7264c1c9e6a9' # replace with image of choice  
instance_type = 'g5.xlarge' # replace with instance of choice

ec2 = boto3.resource('ec2', region_name=region)

instances = ec2.create_instances(  
    MaxCount=num_instances,  
    MinCount=num_instances,  
    ImageId=image_id,  
    InstanceType=instance_type,  
)




def ec2_activate(instance_id, region):
    
    ec2 = boto3.client('ec2', region_name=region)
    cond = True
    while cond == True:
        response = ec2.describe_instances(InstanceIds=instance_id)
        instances = response['Reservations'][0]['Instances']
        instance_state = instances[0]['State']['Name']
        if (instance_state == 'running') or (instance_state == 'pending') or (instance_state == 'stopping'):
            continue
        else:
            ec2.start_instances(InstanceIds=instance_id)
            break

ec2_activate(instance_id, region)