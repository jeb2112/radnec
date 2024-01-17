# demonstration script for training SVM classifier for image radiomics

import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import copy
import pickle

from sklearn.svm import SVC 
from sklearn.metrics import roc_auc_score,roc_curve

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

    datadir = 'D:\Dropbox\BLAST DEVELOPMENT\RAD NEC'
    datafile = os.path.join(datadir,'svm_results.pkl')
    if os.path.exists(datafile):
        with open(datafile,'rb') as fp:
            (rprob,lset_,dset_) = pickle.load(fp)
    else:
        d = loadmat(os.path.join(datadir,'radnec_fits_python.mat'))
        dset_T = d['fits_T_mu'].T
        dset_RN = d['fits_RN_mu'].T
        if False:
            cats = ['tumour','RN']
            dset_vals = np.column_stack(dset_T[:,1],dset_RN[:,1])
            plt.scatter(cats*23,dset_vals)
            plt.show()
        n_T = np.shape(dset_T)[0]
        n_RN = np.shape(dset_RN)[0]
        fset = [0,1,2]
        nfeature = len(fset)
        dset_ = np.row_stack((dset_T[:,fset],dset_RN[:,fset]))
        lset_ = np.ravel(np.row_stack((np.ones((n_T,1)),np.zeros((n_RN,1)))))

        # randomize the data
        np.random.seed()
        ndata = n_T + n_RN
        idx = np.ravel(np.argsort(np.random.random(dset_.shape[0])))
        dset = copy.deepcopy(dset_[idx])
        lset = copy.deepcopy(lset_[idx])
        kfold = 4
        cset = [.01,.1,1,10,100]
        gset = 1./30 * 2.**np.array([-2,-1,0,1,2])
        zmax = 0.0
        nreps = 10

        if False:
            for c in cset:
                for g in [gset[2]]:
                    z = np.zeros((nreps,kfold))
                    for i in range(nreps):
                        print(c,i)
                        # randomize the data
                        np.random.seed()
                        idx = np.ravel(np.argsort(np.random.random(dset_.shape[0])))
                        dset = copy.deepcopy(dset_[idx])
                        lset = copy.deepcopy(lset_[idx])
                        for k in range(kfold):
                            (x_train,y_train,x_test,y_test) = kfold_split(dset,lset,k,kfold)
                            z[i,k],_ = run(x_train, y_train, x_test, y_test, SVC(kernel='linear', probability=True, C=c))
                    if z.mean() > zmax:
                        zmax = z.mean()
                        bestC = c
                    pp(z[i],'\n\nSVM (linear)\n\n')

            print('best c = {:.3f}, accuracy = {:.2f}'.format(bestC,zmax))

        bestC = 1
        nreps = 25
        rz = np.zeros(nreps)
        pset = np.zeros((nreps,ndata))
        # dummy data set test
        if False:
            dset_ = np.tile(copy.deepcopy(lset_),(2,1)).T
            dset_ = dset_ + np.random.randn(ndata,2)*.5
        for i in range(nreps):
            # randomize the data
            np.random.seed()
            idx = np.ravel(np.argsort(np.random.random(dset_.shape[0])))
            # hard-coded dataset is not kfold evenly
            dset = copy.deepcopy(dset_[idx[:-2]])
            lset = copy.deepcopy(lset_[idx[:-2]])
            z = np.zeros(kfold)
            for k in range(kfold):
                (x_train,y_train,x_test,y_test) = kfold_split(dset,lset,k,kfold)
                z[k],p = run(x_train, y_train, x_test, y_test, SVC(kernel='linear', probability=True, C=bestC))
                p0 = np.atleast_2d(p[:,0])
                if k == 0:
                    probs = p0
                else:
                    probs = np.column_stack((probs,p0))
            pset[i,idx[:-2]] = probs
            rz[i] = z.mean()
            pp(z,'SVM (linear)')
        pset[pset == 0] = 'nan'
        rprob = np.nanmean(pset,axis=0)
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
    plt.savefig(os.path.join(datadir,'svm_results.png'))
main()

