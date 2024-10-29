# test ssh connection to aws instance

import os
import boto3
from paramiko import SSHClient,RSAKey
from scp import SCPClient
import logging

from src.SSHSession import SSHSession

class SSHSession(object):
    def __init__(self,username,hostname,password=False):
        self.username = username
        self.hostname = hostname
        self.client = SSHClient()
        self.dirlist = None
        if password is False:
            # self.pkey = RSAKey.from_private_key_file("/home/{}/.ssh/id_rsa_{}".format(self.username,self.hostname))
            self.pkey = RSAKey.from_private_key_file("/home/jbishop/keystores/aws/awstest.pem")
            self.client.load_system_host_keys()
            self.client.connect(self.hostname,username=self.username,pkey=self.pkey,timeout=5000)
            
        self.scp = SCPClient(self.client.get_transport())
        self.root = "//PMIFS03.pmiad.profoundmedical.com./ClinicalData$/Clinical Trial Pivotal (TACT)"
        self.site = None
        self.localdir = None
        self.log = logging.getLogger(__name__)

        # set up AWS credentials
        credentials = boto3.Session().get_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
        os.environ["AWS_SESSION_TOKEN"] = credentials.token


    # get contents of AX SAG folders at fpath
    def get_folder(self,fpath,destpath):
        self.run_command('rm /c/tmp/test.zip')
        out,err = self.run_command('cd \'{}\'; zip -r /c/tmp/test.zip Anatomy??'.format(fpath))
        if err is not None:
            self.log.error('failed to zip {}, {}'.format(fpath,err))
            raise OSError

    def run_command(self,c,block=False):
        stdin,stdout,stderr = self.client.exec_command(c)
        if block:
            exit_status = stdout.channel.recv_exit_status()
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        err = str(err) or None
        return (out,err)

    def close(self):
        if self.client:
            self.client.close()
        if self.scp:
            self.scp.close()



user = 'ec2-user'
host = 'ec2-15-222-45-20.ca-central-1.compute.amazonaws.com'
s1 = SSHSession(user,host)
command = 'ls -la'
ret = s1.run_command(command)
print(ret)

pass
