# Process voxel radiomics for RAD NEC datasets
# Jan 16, 2024


import numpy as np
import os
import nibabel as nb
import re
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy
from datetime import datetime as dt
import time
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve
import pandas as pd
import pickle
import SimpleITK as sitk
import pysftp
from urllib.parse import urlparse


class Sftp:
    def __init__(self, hostname, username, password, port=22):
        """Constructor Method"""
        # Set connection object to None (initial value)
        self.connection = None
        self.hostname = hostname
        self.username = username
        self.password = password
        self.port = port

    def connect(self):
        """Connects to the sftp server and returns the sftp connection object"""

        try:
            # Get the sftp connection object
            self.connection = pysftp.Connection(
                host=self.hostname,
                username=self.username,
                password=self.password,
                port=self.port,
            )
        except Exception as err:
            raise Exception(err)
        finally:
            print(f"Connected to {self.hostname} as {self.username}.")

    def listdir(self, remote_path):
        """lists all the files and directories in the specified path and returns them"""
        for obj in self.connection.listdir(remote_path):
            yield obj

    def listdir_attr(self, remote_path):
        """lists all the files and directories (with their attributes) in the specified path and returns them"""
        for attr in self.connection.listdir_attr(remote_path):
            yield attr

    def disconnect(self):
        """Closes the sftp connection"""
        self.connection.close()
        print(f"Disconnected from host {self.hostname}")

def truthtable(fval_T,fval_RN):
    fmax = 2*max(fval_RN,fval_T)
    fpr = np.zeros(101)
    tpr = np.zeros(101)
    for i,t in enumerate(np.arange(0,fmax,fmax/100)):
        tn = 0
        tp = 0
        fn = 0
        fp = 0
        for c in cases_T:
            if dsets['T'][c][im][feature]['mu'] > t:
                tp += 1
            else:
                fn += 1
        for c in cases_RN:
            if dsets['RN'][c][im][feature]['mu'] < t:
                tn += 1
            else:
                fp += 1

        fpr[i] = fp/n_RN
        tpr[i] = tp/n_T
    return tpr,fpr

def fitexp(x,a,b,c):
    return a * np.exp(-b*x) + c
def fitlin(x,a,b):
    return a*x + b

def get_fvector(vname):
    v = sitk.ReadImage(vname)
    v_arr = sitk.GetArrayFromImage(v)
    return v_arr

def listdir(dir,sftp=None):
    if sftp is None:
        return os.listdir(dir)
    else:
        # f = sftp.listdir(dir)
        return [f for f in sftp.listdir(dir)]


##############
# prepare data
##############

dsets = {'RN':{'dir':None,'vv':None},'T':{'dir':None,'vv':None}}
dset_keys = dsets.keys()

if True:
    if False:
        radnecDir = '/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC'
    else:
        # for remote access of window dropbox, direct mount is easiest.
        if not os.path.isdir('/mnt/D'):
            os.system('sudo mount -t cifs //192.168.50.224/D /mnt/D -o rw,uid=jbishop,gid=jbishop,username=Chris\ Heyn\ Lab,vers=3.0')
        radnecDir = '/mnt/D/Dropbox/BLAST DEVELOPMENT/RAD NEC'
        sftp = None
    dataDir = os.path.join(radnecDir,'CONVENTIONAL MRI')
    files = os.listdir(dataDir)

else:
    # for remote windows dropbox, pysftp package mounts C:\Users at /media/OS.
    # it's easier just to use a direct mount rather than processing pysftp.connection() results 
    # for isdir etc. However, can't figure out how to get pysftp to mount D:
    sftp = Sftp('192.168.50.224','Chris Heyn Lab','jonathanbishop')
    sftp.connect()
    sftp.connection.cd('D:')
    radnecDir = '/D:/Dropbox/BLAST DEVELOPMENT/RAD NEC'
    dataDir = os.path.join(radnecDir,'CONVENTIONAL MRI')
    files = listdir(dataDir,sftp=sftp)
    
outputDir = os.path.join(radnecDir,'radiomics')
datafile = os.path.join(outputDir,'voxel_radiomics.pkl')

cases = [f for f in files if os.path.isdir(os.path.join(dataDir,f))]

files = os.listdir(os.path.join(dataDir,cases[0]))
# naming convention from pyradiomics
rfeature = re.compile('^M.*_([a-z]+_[A-Za-z]+)\.')
rcase = re.compile('^M00[0-9]{2}')
featureset_files = list(filter(rfeature.search,files))
featureset = [rfeature.search(f).group(1) for f in featureset_files if rfeature.search(f)]


if os.path.exists(datafile):
    t = time.ctime(os.path.getmtime(datafile))
    print('\n\nloading {}, last modified {}...\n\n'.format(datafile,t))
    with open(datafile,'rb') as fp:
        dsets = pickle.load(fp)   

else:

    for d in ['RN','T']:

        for c in cases:
            if d in c:
                dsets[d][c] = {'t1mprage':{},'t2flair':{}}
                for im in ['t1mprage','t2flair']:
                    for feature in featureset:
                        dsets[d][c][im][feature] = {'mu':0,'se':0}
                        f = rcase.match(c).group() + '_' + im + '_' + feature + '.nrrd'
                        fv = get_fvector(os.path.join(dataDir,c,f))
                        dsets[d][c]['dir'] = os.path.join(dataDir,c)
                        dsets[d][c][im][feature]['mu'] = np.nanmean(fv)
                        dsets[d][c][im][feature]['se'] = np.nanstd(fv)/(len(fv)-1)

    with open(datafile,'wb') as fp:
        pickle.dump(dsets,fp)


# first pass significant difference of feature
cases_RN = [key for key in dsets['RN'].keys() if key.endswith('RN')]
cases_T = [key for key in dsets['T'].keys() if key.startswith('M')]
n_T = len(cases_RN)
n_RN = len(cases_RN)
labelset_ = np.ravel(np.row_stack((np.ones((n_T,1)),np.zeros((n_RN,1)))))

fset1 = {'t1mprage':[],'t2flair':[]}
for feature in featureset:

    for im in ['t1mprage','t2flair']:

        fval_RN = [dsets['RN'][case][im][feature]['mu'] for case in cases_RN]
        fval_T = [dsets['T'][case][im][feature]['mu'] for case in cases_T]
        res = scipy.stats.ttest_ind(fval_RN,fval_T,equal_var=True)
        if res.pvalue < 0.05:
            fset1[im].append(feature)


# second pass. univariate logistic regression by auc
auc = {'t1mprage':{},'t2flair':{}}
for im in ['t1mprage','t2flair']:
    for feature in fset1[im]:
        fval_RN = [dsets['RN'][case][im][feature]['mu'] for case in cases_RN]
        fval_T = [dsets['T'][case][im][feature]['mu'] for case in cases_T]
        X = np.array(fval_T + fval_RN)[:,np.newaxis]
        clf = LogisticRegression(solver="liblinear").fit(X, labelset_)
        l_prob = clf.predict_proba(X)[:, 1]
        auc[im][feature] = roc_auc_score(labelset_, l_prob)
        print('im {}, feature {}, auc = {:.2f}'.format(im, feature,auc[im][feature]))

# results
a=1


if sftp:
    sftp.disconnect()

 



########
# output
########

plt.figure(fig1)
plt.savefig(os.path.join(output_dir,'voxel_volume2.png'))

plt.figure(fig5)
plt.savefig(os.path.join(output_dir,'voxel_volume2_separate.png'))
