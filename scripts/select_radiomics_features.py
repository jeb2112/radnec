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
from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso
from sklearn.metrics import roc_auc_score,roc_curve,r2_score,mean_squared_error
from sklearn.feature_selection import RFE,RFECV
from sklearn.model_selection import GridSearchCV,LeaveOneOut
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
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


# loocv crossval
def loocv(X,y,cv_eval=LogisticRegression(solver='liblinear')):
    l_prob = np.zeros(len(X))
    for i in range(len(X)):
        ytrain = y.copy().drop(i,axis=0)
        Xtrain = X.copy().drop(i,axis=0)
        Xtest = X.loc[[i]]
        l_prob[i] = cv_eval.fit(Xtrain, ytrain).predict_proba(Xtest)[:, 1][0]
    return roc_auc_score(y, l_prob)

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
    dataDir = os.path.join(radnecDir,'radiomics')
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

# load master spreadsheet
mdata = pd.read_excel(os.path.join(radnecDir,'MANUSCRIPT','RAD NEC MET RESULTS','02_21_2024 RAD NEC MET.xlsx'),sheet_name='Montage Jan 2021 - June 2023 ')
mrows = np.where(mdata['ID'].str.contains(pat='M00',regex=True)==True)[0]
# working dataframe
mlrdata = pd.DataFrame({'ID':mdata['ID'][mrows],'Final Dx':mdata['Final Dx'][mrows]})


if os.path.exists(datafile):
    t = time.ctime(os.path.getmtime(datafile))
    print('\n\nloading {}, last modified {}...\n\n'.format(datafile,t))
    with open(datafile,'rb') as fp:
        dsets,mlrdata = pickle.load(fp)
        featureset = mlrdata.filter(regex=('(t1|t2).*')).columns.tolist()

else:

    files = os.listdir(os.path.join(dataDir,cases[0]))
    # naming convention from pyradiomics
    rfeature = re.compile('^((?:t1mprage|t2flair)_[a-z]+_[A-Za-z]+)\.')
    featureset = [rfeature.search(f).group(1) for f in files if rfeature.search(f)]
    mlrdata = pd.concat([mlrdata,pd.DataFrame(columns=featureset)],axis=1)

    for i,c in enumerate(cases):
        print(c)
        dx = mlrdata['Final Dx'][i]
        dsets[dx][c] = {'t1mprage':{},'t2flair':{}}
        for feature in featureset:
            im,f = feature.split('_',1)
            dsets[dx][c][im][f] = {'mu':0,'se':0}
            fv = get_fvector(os.path.join(dataDir,c,feature+'.nrrd'))
            dsets[dx][c]['dir'] = os.path.join(dataDir,c)
            dsets[dx][c][im][f]['mu'] = np.nanmean(fv)
            dsets[dx][c][im][f]['se'] = np.nanstd(fv)/(len(fv)-1)
            mlrdata.loc[i,feature] = np.nanmean(fv)

    with open(datafile,'wb') as fp:
        pickle.dump((dsets,mlrdata),fp)


# for evaluations
cv_estimator = LogisticRegression(solver='liblinear',penalty='l1',max_iter=1000)
rfecv = RFECV(cv_estimator,scoring='roc_auc')
clf_eval = LogisticRegression(solver='liblinear')

# all features with significant difference
cases_RN = [key for key in dsets['RN'].keys() if key.startswith('M')]
cases_T = [key for key in dsets['T'].keys() if key.startswith('M')]
n_T = len(cases_T)
n_RN = len(cases_RN)
# convention for combined list is T,RN
labelset_ = np.ravel(np.row_stack((np.ones((n_T,1)),np.zeros((n_RN,1)))))
mlrdata['y'] = 0
# NB for rhs assignment, just need tuple. but for lhs assignment, have to extract the indices from the tuple
mlrdata.loc[np.where(mlrdata['Final Dx']=='RN')[0],'y']=1

fset1 = {'t1mprage':[],'t2flair':[]}
for feature in featureset:
    im,f = feature.split('_',1)

    fval_RN = [dsets['RN'][case][im][f]['mu'] for case in cases_RN]
    fval_T = [dsets['T'][case][im][f]['mu'] for case in cases_T]
    res = scipy.stats.ttest_ind(fval_RN,fval_T,equal_var=True)
    if res.pvalue < 0.05:
        fset1[im].append(f)
    else:
        mlrdata = mlrdata.drop(feature,axis=1)
featureset = mlrdata.filter(regex=('(t1|t2).*')).columns.tolist()
# standard the featureset
scaler = StandardScaler()
mlrdata[featureset] = scaler.fit_transform(mlrdata[featureset])


# combine into single list
t2_flist = ['t2flair_'+t for t in fset1['t2flair']]
t1_flist = ['t1mprage_'+t for t in fset1['t1mprage']]
fset1_comb = t2_flist + t1_flist


# first pass. select features by univariate logistic regression and auc value
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

# combine and sort the two lists
t2_flist = list(map(list,auc['t2flair'].items()))
t1_flist = list(map(list,auc['t1mprage'].items()))
t2_flist = [['t2flair_'+t[0],t[1]] for t in t2_flist]
t1_flist = [['t1mprage_'+t[0],t[1]] for t in t1_flist]
feature_list = t2_flist + t1_flist
n_auc = 10
featureset_univariate_auc = sorted(feature_list,key=lambda x:x[1],reverse=True)[:n_auc]

# evaluate the top univariate features. 
# iterate number of features from 10 down, to maximize the loocv auc
auc_univariate = 0
while n_auc > 0:
    fset = [t[0] for t in featureset_univariate_auc[:n_auc]]
    X_test = mlrdata[fset].copy()
    auc = loocv(X_test,mlrdata['y'])
    if auc > auc_univariate:
        auc_univariate = auc
        n_auc_univariate = n_auc
        featureset_univariate = fset
        X_univariate = X_test
    n_auc -= 1
auc_univariate_train = roc_auc_score(mlrdata['y'],clf_eval.fit(X_univariate, mlrdata['y']).predict_proba(X_univariate)[:,1])


# set up a grid search for the factor C for 
# regularized multi-variate feature selection
grid_params = {'C':np.logspace(-1,1,20)}
X = mlrdata[featureset].copy()

# second pass. multivariate feature selection using rfe with l1 penalty.

# combined gridsearch and rfe, example from stackflow.
# training auc comes in about 0.83 with 6 features which is
# pretty close to the manual grid search below. 
if False:
    clf_C_cv = GridSearchCV(rfecv,grid_params_rfecv,scoring='roc_auc')
    clf_C_cv.fit(X,mlrdata['y'])
    bestC_cv = clf_C_cv.best_params_['estimator__C']
    rfecv.fit(X,mlrdata['y'])
    rfecv_idx_comb = np.where(rfecv.support_ == True)[0]
    featureset_rfecv_comb = [featureset[t] for t in rfecv_idx_comb]
    # evaluation the selection
    clf = LogisticRegression(solver="liblinear").fit(mlrdata[featureset_rfecv_comb], mlrdata['y'])
    l_prob = clf.predict_proba(mlrdata[featureset_rfecv_comb])[:, 1]
    auc_rfecv_comb = roc_auc_score(mlrdata['y'], l_prob)

# manual grid search for C and rfecv feature selction, impleneting LOOCV evaluation
auc_rfecv_loocv = 0
for i,C in enumerate(grid_params['C']):
    cv_estimator.C = C
    rfecv.fit(X,mlrdata['y'])
    rfecv_idx = np.where(rfecv.support_ == True)[0]
    featureset_rfecv = [featureset[t] for t in rfecv_idx]
    X_test = mlrdata[featureset_rfecv].copy()
    auc = loocv(X_test,mlrdata['y'])
    if auc > auc_rfecv_loocv:
        C_rfecv = C
        auc_rfecv_loocv = auc
        featureset_rfecv_loocv = featureset_rfecv
        rfecv_idx_loocv = rfecv_idx
        X_rfecv = X_test
auc_rfecv_train = roc_auc_score(mlrdata['y'],clf_eval.fit(X_rfecv, mlrdata['y']).predict_proba(X_rfecv)[:,1])


# using the number of features returned by RFECV, 
# calculate the regular RFE selection as a check
rfe = RFE(LogisticRegression(solver='liblinear',penalty='l1',C=C_rfecv,max_iter=1000),n_features_to_select=len(featureset_rfecv_loocv))
rfe.fit(X,mlrdata['y'])
rfe_idx = np.where(rfe.support_ == True)[0]
featureset_rfe = [featureset[t] for t in rfe_idx]
X_rfe = mlrdata[featureset_rfe].copy()
# evaluation the selection
auc_rfe = loocv(X_rfe,mlrdata['y'])
auc_rfe_train = roc_auc_score(mlrdata['y'],clf_eval.fit(X_rfe, mlrdata['y']).predict_proba(X_rfe)[:,1])



########
# output
########

with open(os.path.join(outputDir,'voxel_radiomics_selected_features.pkl'),'wb') as fp:
    pickle.dump((mlrdata[['ID','Final Dx','y']+featureset_rfecv_loocv]),fp)

drow_format = "{}\t{:<40}\t{:<8.1f}({:.1f})\t{:<8.1f}({:.1f})\t{:<8.3f}\t{:<8.2f}"
drow_format_mul = "{}\t{:<40}\t{:<8.1f}({:.1f})\t{:<8.1f}({:.1f})\t{:<8.3f}"
trow_format = "{}\t{:<40}\t{:8}\t{:8}\t{:8}\t{:8}"
drow_format_f = "{}\t{:<40}\t{:<8.1f}({:.1f})\t{:<8.1f}({:.1f})\t{:<8.3f}\t{:<8.2f}\n"
drow_format_mul_f = "{}\t{:<40}\t{:<8.1f}({:.1f})\t{:<8.1f}({:.1f})\t{:<8.3f}\n"
trow_format_f = "{}\t{:<40}\t{:8}\t{:8}\t{:8}\t{:8}\n"

with open(os.path.join(outputDir,'voxel_radiomics_features.txt'),'w') as fp:

    print('\n\nTable 1: Top ten radiomic features by univariate regression\n')
    fp.write('\n\nTable 1: Top ten radiomic features by univariate regression\n')
    print(trow_format.format('Rank','Feature','RN mean (se)','TP mean(se)','P-value','AUC '))
    fp.write(trow_format_f.format('Rank','Feature','RN mean (se)','TP mean(se)','P-value','AUC '))
    for i,t in enumerate(featureset_univariate_auc):
        im,feature = t[0].split('_',1)
        fval_RN = [dsets['RN'][case][im][feature]['mu'] for case in cases_RN]
        fval_T = [dsets['T'][case][im][feature]['mu'] for case in cases_T]
        res = scipy.stats.ttest_ind(fval_RN,fval_T,equal_var=True)    
        mu_RN = np.mean(fval_RN)
        mu_TP = np.mean(fval_T)
        se_RN = np.std(fval_RN)/np.sqrt(len(fval_RN)-1)
        se_TP = np.std(fval_T)/np.sqrt(len(fval_T)-1)
            
        print(drow_format.format(i+1,t[0].replace('_',' '),mu_RN,se_RN,mu_TP,se_TP,res.pvalue,t[1]))
        fp.write(drow_format_f.format(i+1,t[0].replace('_',' '),mu_RN,se_RN,mu_TP,se_TP,res.pvalue,t[1]))



    print('\n\nTable 2 Top {} univariate features by multivariate regression\n'.format(len(featureset_univariate)))
    fp.write('\n\nTable 2: Top {} univariate features by multivariate regression\n'.format(len(featureset_univariate)))
    for i,t in enumerate(featureset_univariate):
        im,feature = t.split('_',1)
        fval_RN = [dsets['RN'][case][im][feature]['mu'] for case in cases_RN]
        fval_T = [dsets['T'][case][im][feature]['mu'] for case in cases_T]
        res = scipy.stats.ttest_ind(fval_RN,fval_T,equal_var=True)    
        mu_RN = np.mean(fval_RN)
        mu_TP = np.mean(fval_T)
        se_RN = np.std(fval_RN)/np.sqrt(len(fval_RN)-1)
        se_TP = np.std(fval_T)/np.sqrt(len(fval_T)-1)
            
        print(drow_format_mul.format(i+1,t.replace('_',' '),mu_RN,se_RN,mu_TP,se_TP,res.pvalue))
        fp.write(drow_format_mul_f.format(i+1,t.replace('_',' '),mu_RN,se_RN,mu_TP,se_TP,res.pvalue))

    print('\t{:<40}\t{:.2f}'.format('AUC (loocv)',auc_univariate))
    fp.write('\t{:<40}\t{:.2f}\n'.format('AUC (loocv)',auc_univariate))
    print('\t{:<40}\t{:.2f}'.format('AUC (train)',auc_univariate_train))
    fp.write('\t{:<40}\t{:.2f}\n'.format('AUC (train)',auc_univariate_train))


    print('\n\nTable 3: Top radiomic features by multivariate L1,RFECV\n')
    fp.write('\n\nTable 2: Top radiomic features by multivariate L1,RFECV\n')
    for i,t in enumerate(featureset_rfecv_loocv):
        im,feature = t.split('_',1)
        fval_RN = [dsets['RN'][case][im][feature]['mu'] for case in cases_RN]
        fval_T = [dsets['T'][case][im][feature]['mu'] for case in cases_T]
        res = scipy.stats.ttest_ind(fval_RN,fval_T,equal_var=True)    
        mu_RN = np.mean(fval_RN)
        mu_TP = np.mean(fval_T)
        se_RN = np.std(fval_RN)/np.sqrt(len(fval_RN)-1)
        se_TP = np.std(fval_T)/np.sqrt(len(fval_T)-1)
            
        print(drow_format_mul.format('',t.replace('_',' '),mu_RN,se_RN,mu_TP,se_TP,res.pvalue))
        fp.write(drow_format_mul_f.format('',t.replace('_',' '),mu_RN,se_RN,mu_TP,se_TP,res.pvalue))
    print('\t{:<40}\t{:.2f}'.format('AUC (loocv)',auc_rfecv_loocv))
    fp.write('\t{:<40}\t{:.2f}\n'.format('AUC (loocv)',auc_rfecv_loocv))
    print('\t{:<40}\t{:.2f}'.format('AUC (train)',auc_rfecv_train))
    fp.write('\t{:<40}\t{:.2f}\n'.format('AUC (train)',auc_rfecv_train))

    # repeating the RFECV with RFE is giving the same results
    if False:
        print('\n\nTable 3: Top radiomic features by L1,RFE\n')
        fp.write('\n\nTable 3: Top radiomic features by L1,RFE\n')
        for i,t in enumerate(featureset_rfe):
            im,feature = t.split('_',1)
            fval_RN = [dsets['RN'][case][im][feature]['mu'] for case in cases_RN]
            fval_T = [dsets['T'][case][im][feature]['mu'] for case in cases_T]
            res = scipy.stats.ttest_ind(fval_RN,fval_T,equal_var=True)    
            mu_RN = np.mean(fval_RN)
            mu_TP = np.mean(fval_T)
            se_RN = np.std(fval_RN)/np.sqrt(len(fval_RN)-1)
            se_TP = np.std(fval_T)/np.sqrt(len(fval_T)-1)
                
            print(drow_format_mul.format('',t.replace('_',' '),mu_RN,se_RN,mu_TP,se_TP,res.pvalue))
            fp.write(drow_format_mul_f.format('',t.replace('_',' '),mu_RN,se_RN,mu_TP,se_TP,res.pvalue))
        print('\t{:<40}\t{:.2f}'.format('AUC (loocv)',auc_rfe))
        fp.write('\t{:<40}\t{:.2f}\n'.format('AUC (loocv)',auc_rfe))
        print('\t{:<40}\t{:.2f}'.format('AUC (train)',auc_rfe_train))
        fp.write('\t{:<40}\t{:.2f}\n'.format('AUC (train)',auc_rfe_train))


if sftp:
    sftp.disconnect()

# if os.system('mountpoint -q /mnt/D') == 0:
#     os.system('sudo umount /mnt/D')
 
