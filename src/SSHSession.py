# simple class for ssh,scp functionality

import logging
import re
import getpass
import numpy as np
import os
import stat
from paramiko import SSHClient,RSAKey,SFTPClient,Transport
import pssh
import boto3
import pssh.clients

logging.getLogger("paramiko").setLevel(logging.ERROR)

class MySFTPClient(SFTPClient):
    def put_dir(self, source, target):
        ''' Uploads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are 
            created under target.
        '''
        for item in os.listdir(source):
            if os.path.isfile(os.path.join(source, item)):
                self.put(os.path.join(source, item), '%s/%s' % (target, item))
            else:
                self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                self.put_dir(os.path.join(source, item), '%s/%s' % (target, item))

    def get_dir(self, source, target):
        ''' Downloads the contents of the source directory to the target path. The
            target directory needs to exists. All subdirectories in source are 
            created under target.
        '''
        for fileattr in self.listdir_attr(source):  
            if stat.S_ISDIR(fileattr.st_mode):
                os.mkdir('%s/%s' % (target, fileattr.filename), ignore_existing=True)
                self.get_dir(os.path.join(source, fileattr.filename), os.path.join(target, fileattr.filename))
            else:
                self.get(os.path.join(source,fileattr.filename), os.path.join(target,fileattr.filename))
                # self.get(os.path.join(source, item), '%s/%s' % (target, item))

    # rmdir for non-empty dir
    # assumes no sub-dirs
    def remove_dir(self,target):
        for item in self.listdir(target):
            self.remove(os.path.join(target,item))
        self.rmdir(target)
       

class SSHSession(object):
    def __init__(self,username,hostname,password=False):
        self.username = username
        self.hostname = hostname
        if password is False:
            self.pkey = RSAKey.from_private_key_file("/home/{}/keystores/aws/awstest.pem".format(getpass.getuser()))

        #paramiko
        self.client = SSHClient()
        self.client.load_system_host_keys()
        self.client.connect(self.hostname,username=self.username,pkey=self.pkey,timeout=5000)
        self.transport = self.client.get_transport()
        self.transport.default_window_size = np.power(2,21)
        self.sftp = MySFTPClient.from_transport(self.transport)
        # parallel ssh
        self.client = pssh.clients.SSHClient(self.hostname,user=self.username,pkey=self.pkey)

        self.dirlist = None
            
            
        self.site = None
        self.localdir = None
        self.log = logging.getLogger(__name__)

        # set up AWS credentials
        credentials = boto3.Session().get_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
        os.environ["AWS_SESSION_TOKEN"] = credentials.token

    def run_command(self,c,block=False):
        stdin,stdout,stderr = self.client.exec_command(c)
        if block:
            exit_status = stdout.channel.recv_exit_status()
            return exit_status
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        err = str(err) or None
        return (out,err)

    def close(self):
        if self.client:
            self.client.close()
        if self.sftp:
            self.sftp.close()

