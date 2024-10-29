# class for a ssh session and functions

import logging
import re
import difflib
import os
import stat
from paramiko import SSHClient,RSAKey,SFTPClient,Transport
from scp import SCPClient
import boto3

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

    # overrides rmdir for non-empty dir
    # assumes no sub-dirs
    def remove_dir(self,target):
        for item in self.listdir(target):
            self.remove(os.path.join(target,item))
        self.rmdir(target)
       

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
        self.sftp = MySFTPClient.from_transport(self.client.get_transport())
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
    
    def run_scp(self,c,block=False):
        stdin,stdout,stderr = self.scp.exec_command(c)
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

