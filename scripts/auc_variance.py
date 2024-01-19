# estimate variance of AUC statistic.
# Jan 2024


import numpy as np
import random
import os
import matplotlib.pyplot as plt
import scipy
import copy
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import roc_auc_score,roc_curve
import pickle
import compare_auc_delong_xu


# modification of scipy.loadmat to work around nested structures in matlab
def loadmat(filename):
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

# scipy bootstrap only supports 1D data vectors.
# workaround for a 2-dimensional (ie labelled) data vector is to map a 1D 
# index to a dict containing the 2D dataset, and have that 
# dict be in scope of the callable as a global, since there is no
# optional arg available in scipy bootstrap to pass it in with 
def do_bootstrap(X,labelset_,cl=0.95):
    ids = [ i for i in range(len(X))]
    data = zip(X,labelset_)
    dataid_dict = {id:(mu,label) for id, (mu,label) in enumerate(data)}
    # callable for bootstrap
    def get_auc(z,axis=0):
        # LR requires 2d X vector
        X = np.array([dataid_dict[zz][0] for zz in z])[:,np.newaxis]
        labelset_ = np.ravel(np.array([dataid_dict[zz][1] for zz in z]))
        clf = LogisticRegression(solver="liblinear").fit(X, labelset_)
        l_prob = clf.predict_proba(X)[:, 1]
        auc = roc_auc_score(labelset_, l_prob)
        return auc
    auc0 = get_auc(ids)
    res = scipy.stats.bootstrap((ids,),get_auc,confidence_level=cl,random_state=np.random.default_rng(),vectorized=False)
    return auc0,res

def get_auc_prob(X,labelset_):
    clf = LogisticRegression(solver="liblinear").fit(X, labelset_)
    l_prob = clf.predict_proba(X)[:, 1]
    return l_prob

##############
# prepare data
##############

if False:
    radnecDir = '/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC'
else:
    # remote access of window dropbox
    if os.system('mountpoint -q /mnt/D') != 0:
        # do this manually until i get password figured out
        # os.system('sudo mount -t cifs //192.168.50.224/D /mnt/D -o rw,uid=jbishop,gid=jbishop,username=Chris\ Heyn\ Lab,vers=3.0')
        a=1
    radnecDir = '/mnt/D/Dropbox/BLAST DEVELOPMENT/RAD NEC'
    
# load output from matlab RADNEC processing script
datafile = os.path.join(radnecDir,'radnec_processed_blast_python.mat')
dset = loadmat(datafile)


##########
# DSC data
##########

X = dset['cbv_rCBV']
n_cbv = len(X)
labelset_ = np.where(dset['cbv_groundtruth']=='TP',1,0)
# bootstrap CI
(auc0_cbv,res_cbv) = do_bootstrap(X,labelset_)
# fast de Long's
n_trials = 1000
auc_delong_cbv = np.zeros(n_trials)
variances_cbv = np.zeros(n_trials)
np.random.seed(1234235)
idx = [i for i in range(n_cbv)]
for trial in range(n_trials):
    idx_trial = random.choices(idx,k=n_cbv)
    Xtrial = copy.deepcopy(X[idx_trial])[:,np.newaxis]
    labelset_trial = np.ravel(copy.deepcopy(labelset_[idx_trial]))
    l_prob = get_auc_prob(Xtrial,labelset_trial)
    auc_delong_cbv[trial], variances_cbv[trial] = compare_auc_delong_xu.delong_roc_variance(labelset_trial, l_prob)


##################
# blast FLAIR data
##################

fval_RN = dset['fits_RN_mu'][1] # index 1 is ET
n_RN = len(fval_RN)
fval_T = dset['fits_T_mu'][1]
n_T = len(fval_T)
X = np.concatenate((fval_T,fval_RN))
n_flair = n_T + n_RN
labelset_ = np.row_stack((np.ones((n_T,1)),np.zeros((n_RN,1))))
# bootstrap
(auc0_flair,res_flair) = do_bootstrap(X,labelset_)
# fast de Long
n_trials = 1000
auc_delong_flair = np.zeros(n_trials)
variances_flair = np.zeros(n_trials)
np.random.seed(1234235)
idx = [i for i in range(n_flair)]
for trial in range(n_trials):
    idx_trial = random.choices(idx,k=n_flair)
    Xtrial = copy.deepcopy(X[idx_trial])[:,np.newaxis]
    labelset_trial = np.ravel(copy.deepcopy(labelset_[idx_trial]))
    l_prob = get_auc_prob(Xtrial,labelset_trial)
    auc_delong_flair[trial], variances_flair[trial] = compare_auc_delong_xu.delong_roc_variance(labelset_trial, l_prob)




########
# output
########
print('T2 FLAIR')
print('scipy bootstrap')
print('auc = {:.2f}, CI = [{:.2f},{:.2f}], stddev = {:.3f}'.format(auc0_flair,
                                                                   res_flair.confidence_interval.low,
                                                                   res_flair.confidence_interval.high,
                                                                   res_flair.standard_error))
print('fast de Long')
print('auc = {:2f}, stddev = {:.3f}'.format(np.mean(auc_delong_flair),np.sqrt(variances_flair.mean())))


print('CBV')
print('scipy bootstrap')
print('auc = {:.2f}, CI = [{:.2f},{:.2f}], stdev = {:.3f}'.format(auc0_cbv,
                                                                   res_cbv.confidence_interval.low,
                                                                   res_cbv.confidence_interval.high,
                                                                   res_cbv.standard_error))

print('fast de Long')
print('auc = {:2f}, stddev = {:.3f}'.format(np.mean(auc_delong_cbv),np.sqrt(variances_cbv.mean())))

if False:
    if os.system('mountpoint -q /mnt/D') == 0:
        os.system('sudo umount /mnt/D')
 
