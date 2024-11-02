# simple class for ssh,scp functionality

import logging
import re
import getpass
import numpy as np
import os
import stat
from paramiko import SSHClient,RSAKey,SFTPClient,Transport
from pssh.clients.ssh import SSHClient as SSHClient2
from pssh.clients.ssh import ParallelSSHClient
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
                # self.mkdir('%s/%s' % (target, item), ignore_existing=True)
                self.mkdir('%s/%s' % (target, item))
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
    def remove_dir(self,target):
       
        for fileattr in self.listdir_attr(target):
            if stat.S_ISDIR(fileattr.st_mode):
                self.remove_dir(os.path.join(target,fileattr.filename))
                # for item in self.listdir(os.path.join(target,fileattr.filename)):
                #     self.remove(os.path.join(target,fileattr.filename,item))
                # self.rmdir(os.path.join(target,fileattr.filename))
            else:
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
        # parallel ssh. tricky.
        # pssh.clients.SSHClient() appears in the documentation but fails to auth public key
        # pssh.clients.ssh.SSHClient() is mentioned on github issues as the solution to the above problem but doesn't exist no modul
        # from pssh.clients.ssh import SSHClient exists and auths public key
        self.client2 = SSHClient2(self.hostname,user=self.username,pkey="/home/{}/.ssh/awstest.pem".format(getpass.getuser()))
        self.client2parallel = ParallelSSHClient(list(self.hostname),user=self.username,pkey="/home/{}/.ssh/awstest.pem".format(getpass.getuser()))

        self.dirlist = None
            
        self.site = None
        self.localdir = None
        self.log = logging.getLogger(__name__)

        # set up AWS credentials
        credentials = boto3.Session().get_credentials()
        os.environ["AWS_ACCESS_KEY_ID"] = credentials.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = credentials.secret_key
        os.environ["AWS_SESSION_TOKEN"] = credentials.token

    # for paramiko
    def run_command(self,c,block=False):
        stdin,stdout,stderr = self.client.exec_command(c)
        if block:
            exit_status = stdout.channel.recv_exit_status()
            return exit_status
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        err = str(err) or None
        return (out,err)

    # for pssh
    def run_command2(self,c,block=True):
        res = self.client2.run_command(c)
        # pssh SSHClient is async, but consuming the output waits
        # until the command is complete
        if block==True:
            for line in res.stdout:
                print(line)
            return res.stdout
        else:
            return 0

    def run_command2parallel(self,c,block=False):
        res = self.client2parallel.run_command(c)
        self.client2.join()
        for host_output in res:
            hostname = host_output.host
            stdout = list(host_output.stdout)
            print("Host %s: exit code %s, output %s" % (
                hostname, host_output.exit_code, stdout))        
        return (hostname,host_output.exit_code,stdout)

    def close(self):
        if self.client:
            self.client.close()
        if self.sftp:
            self.sftp.close()

