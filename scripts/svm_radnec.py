# script for training RAD NEC SVM classifier

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

    if True:
        radnecDir = '/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC'
    else:
        # for remote access of window dropbox, direct mount is easiest.
        # if os.system('mountpoint -q /mnt/D') == 0:
        #     os.system('sudo mount -t cifs //192.168.50.224/D /mnt/D -o rw,uid=jbishop,gid=jbishop,username=Chris\ Heyn\ Lab,vers=3.0')
        radnecDir = '/mnt/D/Dropbox/BLAST DEVELOPMENT/RAD NEC'
    datadir = os.path.join(radnecDir,'radiomics')
    datafile = os.path.join(datadir,'voxel_radiomics_selected_features.pkl')
    if os.path.exists(datafile):
        with open(datafile,'rb') as fp:
            mlrdata = pickle.load(fp)
            featureset = mlrdata.filter(regex=('(t1|t2).*')).columns.tolist()
    else:
        print('No data file found')
        return



    # randomize the data
    np.random.seed()
    ndata = len(mlrdata)
    # idx = np.ravel(np.argsort(np.random.random(dset_.shape[0])))
    # dset = copy.deepcopy(dset_[idx])
    # lset = copy.deepcopy(lset_[idx])
    kfold = 4
    cset = np.logspace(0,2,60)
    gset = 1./30 * 2.**np.array([-2,-1,0,1,2])
    gset = np.logspace(-2,2,5,base=2) / 30

    # dummy data set test
    if False:
        dset_ = np.tile(copy.deepcopy(lset_),(2,1)).T
        dset_ = dset_ + np.random.randn(ndata,2)*.5
    kernel = 'rbf'
    rs = 11
    g = .03

    if False:
        auc_best = 0
        for c in cset:
            z = np.zeros(ndata)
            probs = np.zeros((ndata,2))
            for k in range(ndata):
                x_train = np.array(mlrdata[featureset].copy().drop(k,axis=0))
                x_test = np.reshape(np.array(mlrdata.loc[k,featureset]),(1,-1))
                y_train = np.array(mlrdata['y'].drop(k,axis=0))
                y_test = np.reshape(np.array(mlrdata['y'].loc[k]),(1,-1))
                z[k],probs[k] = run(x_train, y_train, x_test, y_test, SVC(kernel=kernel, probability=True, random_state = rs,C=c, gamma=g))
            auc = roc_auc_score(np.array(mlrdata['y']),probs[:,1])
            if auc > auc_best:
                auc_best = auc
                bestC = c

    bestC = 20

    z = np.zeros(ndata)
    probs = np.zeros((ndata,2))
    for k in range(ndata):
        x_train = np.array(mlrdata[featureset].copy().drop(k,axis=0))
        x_test = np.reshape(np.array(mlrdata.loc[k,featureset]),(1,-1))
        y_train = np.array(mlrdata['y'].drop(k,axis=0))
        y_test = np.reshape(np.array(mlrdata['y'].loc[k]),(1,-1))
        z[k],probs[k] = run(x_train, y_train, x_test, y_test, SVC(kernel=kernel, probability=True, random_state = rs,C=bestC, gamma=g))


    # pp(z,'SVM (linear)')
    # with open(datafile,'wb') as fp:
    #     pickle.dump((rprob,lset_,dset_),fp)
    if False:
        plt.scatter(lset_,rprob)
        plt.show()

    # construct roc
    auc = roc_auc_score(np.array(mlrdata['y']),probs[:,1])
    roc = roc_curve(mlrdata['y'],probs[:,1])
    plt.plot(roc[0],roc[1],marker='.',color='b')
    ax = plt.gca()
    ax.set_aspect('equal')
    # plt.plot([0,1],[0,1],color='k',linestyle=':')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.text(0.5,0.3,'SVM(linear),radiomics')
    plt.text(0.5,0.25,'auc = {:.2f}'.format(auc))
    plt.xticks([0,.5,1])
    plt.yticks([0,.5,1])
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.show()
    plt.savefig(os.path.join(datadir,'svm_results.png'))
main()

