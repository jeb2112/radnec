# script for fitting and scoring RAD NEC Logistic classifier

import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import copy
import pickle

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import cross_val_score,LeaveOneOut

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

# create a train,test split
# doesn't handle data that isn't evenly divisible
def kfold_split(x,y,k,m):
    ns = int(y.shape[0]/m)
    nfeature = int(x.shape[1])
    s = []
    for i in range(m):
        s.append([x[(ns*i):(ns*i+ns)],y[(ns*i):(ns*i+ns)]])
    x_test, y_test = s[k]
    x_train = []
    y_train = []
    for i in range(m):
        if (i==k):
            continue
        else:
            a,b = s[i]
            x_train.append(a)
            y_train.append(b)
    x_train = np.array(x_train).reshape(((m-1)*ns,nfeature))
    y_train = np.array(y_train).reshape((m-1)*ns)
    return [x_train, y_train, x_test, y_test]

def pp(z,s):
    m = z.shape[0]
    print("%-19s: %0.4f +/- %0.4f | " % (s, z.mean(), z.std()/np.sqrt(m)), end='')
    for i in range(m):
        print("%0.4f " % z[i], end='')
    print()

def run(x_train, y_train, x_test, y_test, clf):
    clf.fit(x_train, y_train)
    if False:
        print("    predictions  :", clf.predict(x_test))
        print("    actual labels:", y_test)
        print("    score = %0.4f" % clf.score(x_test, y_test))
        print()
    return clf.score(x_test,y_test),clf.predict_proba(x_test)

def main():

    if True:
        radnecDir = '/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC'
    else:
        # for remote access of window dropbox, direct mount is easiest.
        if os.system('mountpoint -q /mnt/D') == 0:
            os.system('sudo mount -t cifs //192.168.50.224/D /mnt/D -o rw,uid=jbishop,gid=jbishop,username=Chris\ Heyn\ Lab,vers=3.0')
        radnecDir = '/mnt/D/Dropbox/BLAST DEVELOPMENT/RAD NEC'
    sftp = None
    datadir = os.path.join(radnecDir,'logistic')
    datafile = os.path.join(datadir,'svm_results.pkl')
    if os.path.exists(datafile):
        with open(datafile,'rb') as fp:
            (rprob,lset_,dset_) = pickle.load(fp)
    else:
        # order of vectors in the .mat file from radnec2_final.m
        # 1-4: fnames = {'t2flairvector_et_slice_bias','t2flairvector_et_slice_nonbias','t1cevector_et_slice_bias','adcvector'};
        # 5-6: fnames_raw = {'t2flair_register','t1ce'};
        d = loadmat(os.path.join(datadir,'radnec_processed_blast_python.mat'))
        dset_T = d['fits_T_mu'].T
        dset_RN = d['fits_RN_mu'].T
        if False:
            cats = ['tumour','RN']
            dset_vals = np.column_stack(dset_T[:,1],dset_RN[:,1])
            plt.scatter(cats*23,dset_vals)
            plt.show()
        n_T = np.shape(dset_T)[0]
        n_RN = np.shape(dset_RN)[0]
        fset = [0,1,2] # select vectors according to the ordering above
        nfeature = len(fset)
        dset_ = np.row_stack((dset_T[:,fset],dset_RN[:,fset]))
        lset_ = np.ravel(np.row_stack((np.ones((n_T,1)),np.zeros((n_RN,1)))))

        # randomize the data
        np.random.seed()
        ndata = n_T + n_RN
        idx = np.ravel(np.argsort(np.random.random(dset_.shape[0])))
        X = copy.deepcopy(dset_[idx])
        y = copy.deepcopy(lset_[idx])


        # cross_val_score can't work leave one out
        # kfold = ndata
        kfold = 5
        crossval_scores = cross_val_score(LogisticRegression(), X,y,scoring='roc_auc',cv=kfold)

        # attempt a leave one out
        cv = LeaveOneOut()
        y_true, r_auc = list(), list()
        for train_ix, test_ix in cv.split(X):
            # split data
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            # fit model
            model = LogisticRegression()
            model.fit(X_train, y_train)
            # evaluate model
            y_score = model.predict_proba(X)[:, 1]
            r = roc_auc_score(y, y_score)
            # store
            y_true.append(y_test[0])
            r_auc.append(r)
            if True:
                plt.plot(r_auc,'.')
                plt.ylim([0,1])


        print('AUC = {:.4f} +/- {:.4f}'.format(np.mean(r_auc),np.std(r_auc)/np.sqrt(ndata)))
        with open(datafile,'wb') as fp:
            pickle.dump((rprob,lset_,dset_),fp)
        if False:
            plt.scatter(lset_,rprob)
            plt.show()

    # construct roc
    # label convention must be reversed, using 1-
    auc = roc_auc_score(1-lset_,rprob)
    roc = roc_curve(1-lset_,rprob)
    plt.plot(roc[0],roc[1],marker='.',color='b')
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.plot([0,1],[0,1],color='k',linestyle=':')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.text(0.5,0.3,'SVM(linear),T1,Flair,ADC')
    plt.text(0.5,0.25,'auc = {:.2f}'.format(auc))
    plt.xticks([0,.5,1])
    plt.yticks([0,.5,1])
    plt.xlim([0,1])
    plt.ylim([0,1])
    # plt.show()
    # plt.savefig(os.path.join(datadir,'svm_results.png'))
main()

