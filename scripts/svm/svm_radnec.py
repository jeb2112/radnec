# script for training RAD NEC SVM classifier

import numpy as np
import scipy
from scipy.io import loadmat
import os
import matplotlib.pyplot as plt
import copy
import pickle
import seaborn as sns
import pandas as pd

from sklearn.svm import SVC 
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import GridSearchCV,LeaveOneOut,train_test_split,ShuffleSplit,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# parse matlab struct in to dataframe
def loadstruct(filename,key):
    s = loadmat(filename)
    d = s[key]
    dd = []
    for i in range(len(d)):
        dd.append([d[i][0][j].ravel()[0] for j in range(len(d[i][0]))])
    df = pd.DataFrame(dd,columns = d.dtype.fields.keys())
    return df

# modification of scipy.loadmat to work around nested structures in matlab
def loadmat2(filename):
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

# try to make the custom grid-searchable kernel with precomputed in a pipeline
# wrapper class for the estimator with custom kernel isomap
class IsoMap(BaseEstimator,TransformerMixin):
    def __init__(self, n_neighbors=21,n_components=5):
        super(IsoMap,self).__init__()
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def transform(self, X):
        kernel = Isomap(n_components=self.n_components,n_neighbors=self.n_neighbors)
        Xt = kernel.fit_transform(X)
        Yt = kernel.fit_transform(self.X_train_)
        return np.dot(Xt,Yt.T)

    def fit(self, X, y=None, **fit_params):
        self.X_train_ = X
        return self
    


# add hyperparams to SVM for grid search with callable SVM kernel (eg isomap)
# this did not work, gridsearch deepcopies the estimator, and the inheritance
# of the clone is broken somehow because maybe the super() init doesn't get
# run by deepcopy, and then when setting the kernel attribute
# to be the callable function, get a recursion error.
class IsoSVM(SVC):

    def __init__(
            self,
            *,
            n_components=2,
            n_neighbors=5,
            kernel=None
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        super().__init__(
            probability=True
        )
        # SVC.__init__(self,probability=True)
        if False:
            self.kernel=self.isomap
        else:
            setattr(self,'kernel',self.isomap)

    def isomap(self,X,Y):
        kernel = Isomap(n_components=self.n_components,n_neighbors=self.n_neighbors)
        Xt = kernel.fit_transform(X)
        Yt = kernel.fit_transform(Y)
        return np.dot(Xt,Yt.T)


# loocv crossval
def loocv_log(X,y,C=1):
    cv_eval = LogisticRegression(solver='liblinear',C=C)
    l_prob = np.zeros(len(X))
    for i in range(len(X)):
        ytrain = y.copy().drop(i,axis=0)
        Xtrain = X.copy().drop(i,axis=0)
        Xtest = X.loc[[i]]
        l_prob[i] = cv_eval.fit(Xtrain, ytrain).predict_proba(Xtest)[:, 1][0]
    return roc_auc_score(y, l_prob)

def isomap(X,Y,n_components=4,n_neighbors=10):
    embedding = Isomap(n_components=n_components,n_neighbors=n_neighbors)

    Xt = embedding.fit_transform(X)
    Yt = embedding.fit_transform(Y)
    return np.dot(Xt,Yt.T)

# kfold validation for isomap kernel SVM
def kfold_isosvm(X,y,kernel=isomap,rs=None,n_components=2,n_neighbors=5,kfold=4):
    auc = np.zeros(kfold)
    probs = np.zeros((len(X),2))
    z = np.zeros(len(X))
    estimator = IsoSVM(kernel=kernel,n_components=n_components,n_neighbors=n_neighbors)
    ss = ShuffleSplit(n_splits = kfold,test_size=.25)
    # try to maintain equal class balance in the train test splits
    skf = StratifiedKFold(n_splits=4)
    k = 0
    for train,test in skf.split(X,y):
        X_train,y_train = np.array(X.loc[train,:]),np.array(y[train])
        X_test,y_test = np.array(X.loc[test,:]),np.array(y[test])
        z[test],probs[test] = run(X_train, y_train, X_test, y_test, estimator)
        k += 1
    auc = roc_auc_score(np.array(y), probs[:,1])
    return auc

# Peng et al appeared to use LOOCV for the isomap kernel.
# however, for isomap the number of samples must be > n_neighbours,
# and in LOOCV there is just one test sample.
def loocv_isosvm(X,y,kernel=isomap,rs=None,nreps=25,n_components=2,n_neighbors=5):
    if rs is not None:
        nreps = 1
    auc = np.zeros(nreps)
    estimator = IsoSVM(kernel=kernel,n_components=n_components,n_neighbors=n_neighbors)
    for n in range(nreps):
        z = np.zeros(len(X))
        probs = np.zeros((len(X),2))
        for k in range(len(X)):
            x_train = np.array(X.copy().drop(k,axis=0))
            x_test = np.reshape(np.array(X.loc[k,:]),(1,-1))
            y_train = np.array(y.drop(k,axis=0))
            y_test = np.reshape(np.array(y.loc[k]),(1,-1))
            z[k],probs[k] = run(x_train, y_train, x_test, y_test, estimator)
        auc[n] = roc_auc_score(np.array(y), probs[:,1])
    return np.mean(auc)

def loocv_svm(X,y,C=100,g=.01,kernel='rbf',rs=None,nreps=25):
    if rs is not None:
        nreps = 1
    auc = np.zeros(nreps)
    for n in range(nreps):
        z = np.zeros(len(X))
        probs = np.zeros((len(X),2))
        for k in range(len(X)):
            x_train = np.array(X.copy().drop(k,axis=0))
            x_test = np.reshape(np.array(X.loc[k,:]),(1,-1))
            y_train = np.array(y.drop(k,axis=0))
            y_test = np.reshape(np.array(y.loc[k]),(1,-1))
            z[k],probs[k] = run(x_train, y_train, x_test, y_test, SVC(kernel=kernel, probability=True, random_state = rs,C=C, gamma=g))
        auc[n] = roc_auc_score(np.array(y), probs[:,1])
    return np.mean(auc)

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
    paramfile = os.path.join(datadir,'hyperparams.pkl')
    if os.path.exists(datafile):
        with open(datafile,'rb') as fp:
            mlrdata = pickle.load(fp)
    else:
        print('No data file found')
        return
    # additionally load blast feature vectors from matlab output file
    blast_data = loadstruct(os.path.join(datadir,'radnec_mlrdata.mat'),'s_mlrdata')
    bvec = blast_data.filter(regex=('(MeanBlast.*|Field.*|rCBV)')).columns.tolist()
    for b in bvec:
        mlrdata[b] = blast_data[b]
    # scaled version of the blast vectors 
    scaler = StandardScaler()
    bvec = blast_data.filter(regex=('MeanBlast.*|rCBV')).columns.tolist()
    for b in bvec:
        mlrdata['Std'+b] = scaler.fit_transform(mlrdata[[b]])

    # reduced dataset for rCBV fits
    idx_nan = np.where(mlrdata['rCBV'].isnull().values)[0]
    mlrdata_rcbv = mlrdata.drop(idx_nan,axis=0).reset_index()


    # feature selections for evaluation
    fset = {}
    fset['FLAIR_radiomics_nstd'] = mlrdata.filter(regex=('(t1|t2|MeanBlastFlairETBias|FieldStrength_3T1_15T0).*')).columns.tolist()
    fset['FLAIR_radiomics'] = mlrdata.filter(regex=('(t1|t2|StdMeanBlastFlairETBias|FieldStrength_3T1_15T0).*')).columns.tolist()
    fset['radiomics'] = mlrdata.filter(regex=('(t1|t2|FieldStrength_3T1_15T0).*')).columns.tolist()
    fset['FLAIR_rCBV'] = mlrdata_rcbv.filter(regex=('(MeanBlastFlairETBias|FieldStrength_3T1_15T0|rCBV).*')).columns.tolist()
    fset['all'] = mlrdata_rcbv.filter(regex=('(t1|t2|StdMeanBlastFlairETBias|FieldStrength_3T1_15T0|StdrCBV).*')).columns.tolist()
    fset['Std_FLAIR_rCBV'] = mlrdata_rcbv.filter(regex=('(StdMeanBlastFlairETBias|FieldStrength_3T1_15T0|StdrCBV).*')).columns.tolist()
    fset_eval = ['FLAIR_radiomics','radiomics','FLAIR_rCBV','Std_FLAIR_rCBV','all']


    # test for linearity
    if False:
        C_lin = 2**32
        svm_lin = SVC(kernel='linear',C=C_lin,probability=True)
        X_lin = mlrdata_rcbv[fset['lin']]
        y_lin = mlrdata_rcbv['y']
        svm_lin.fit(X_lin,y_lin)
        acc_lin = svm_lin.score(X_lin,y_lin)    # .788
        auc_lin = roc_auc_score(y_lin,svm_lin.predict_proba(X_lin)[:,1]) # .852

    np.random.seed()
    ndata = len(mlrdata)
    cset = np.logspace(-1,3,40)
    gset = np.logspace(-2,2,40,base=2) / 30
    rs = 11

    y = mlrdata['y'].copy()

    if os.path.exists(paramfile):
        with open(paramfile,'rb') as fp:
            (C_auto,g_auto) = pickle.load(fp)
    else:
        # sklearn grid search, default cross-validation for regular SVC with gaussian RBF
        C_auto = {'rbf':{},'linear':{},'log':{}}
        g_auto = {'rbf':{},'linear':{}}
        if True:
            svm_grid = SVC(kernel='rbf')
            params_grid = {'gamma':gset,'C':cset}
            searcher = GridSearchCV(svm_grid,params_grid)
            for f in fset_eval:
                idx_nan = np.where(mlrdata['rCBV'].isnull().values)[0]
                X_train = mlrdata[fset[f]].drop(idx_nan,axis=0)
                y_train = mlrdata['y'].drop(idx_nan,axis=0)
                searcher.fit(X_train,y_train)
                C_auto['rbf'][f],g_auto['rbf'][f] = tuple(map(searcher.best_params_.get,('C','gamma')))

        # sklearn grid search for regular SVC linear
        if True:
            cset_lin = np.logspace(-2,1,50)
            svm_grid = SVC(kernel='linear')
            params_grid = {'C':cset_lin}
            searcher = GridSearchCV(svm_grid,params_grid)
            for f in fset_eval:
                idx_nan = np.where(mlrdata['rCBV'].isnull().values)[0]
                X_train = mlrdata[fset[f]].drop(idx_nan,axis=0)
                y_train = mlrdata['y'].drop(idx_nan,axis=0)
                searcher.fit(X_train,y_train)
                C_auto['linear'][f] = searcher.best_params_['C']
                g_auto['linear'][f] = 0

        # grid search for logistic
        if True:
            cset_log = np.logspace(-3,1,50)
            log_grid = LogisticRegression(solver='liblinear')
            params_grid = {'C':cset_log}
            searcher = GridSearchCV(log_grid,params_grid)
            for f in fset_eval:
                idx_nan = np.where(mlrdata['rCBV'].isnull().values)[0]
                X_train = mlrdata[fset[f]].drop(idx_nan,axis=0)
                y_train = mlrdata['y'].drop(idx_nan,axis=0)
                searcher.fit(X_train,y_train)
                C_auto['log'][f] = searcher.best_params_['C']


        # grid search isomap custom SVC kernel using precomputed in pipeline
        if False:
            featureset_rad_iso = mlrdata.filter(regex=('(t1|t2).*')).columns.tolist()
            X = mlrdata[featureset_rad_iso]
            dset = list(range(3,11))
            nset = list(range(3,13))
            if False:
                params_grid = {'n_components':dset,'n_neighbors':nset}
                estimator = IsoSVM()
                searcher = GridSearchCV(estimator,params_grid)
            else:
                params_grid = dict([('isomap__n_components',dset),('isomap__n_neighbors',nset)])
                pipe = Pipeline([('isomap',IsoMap()),\
                                ('svm',SVC())
                                ])
                # can't use LOO with isomap
                # searcher = GridSearchCV(pipe,params_grid,verbose=1,cv=LeaveOneOut())
                searcher = GridSearchCV(pipe,params_grid,verbose=1,cv=2)
            searcher.fit(X,y)
            (n_comp_opt,n_neighbors_opt) = searcher.best_params_.values()


        # manual grid search for gaussian RBF, explicit cross-validation. seems to give 
        # similar results to default search above
        if False:
            auc_best = 0
            for c in cset:
                print(c)
                for g in gset:
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
                        C_man = c
                        g_man = g

        with open(os.path.join(datadir,'hyperparams.pkl'),'wb') as fp:
            pickle.dump((C_auto,g_auto),fp)
        # grid results for 'rbf', average of manual and auto searches
        # bestC_rbf = np.mean([107,71])
        # bestg_rbf = np.mean([.011,.017])


    # svm
    if False: # iso svm not finished / not working
        auc_svm_radiomics_iso = kfold_isosvm(mlrdata[featureset_rad],y,n_components=n_comp_opt,n_neighbors=n_neighbors_opt,kernel=isomap)
    auc_svm = {'rbf':{},'linear':{}}
    for k in auc_svm.keys():
        for f in fset_eval:
            idx_nan = np.where(mlrdata['rCBV'].isnull().values)[0]
            X_train = mlrdata[fset[f]].drop(idx_nan,axis=0).reset_index(drop=True)
            y_train = mlrdata['y'].drop(idx_nan,axis=0).reset_index(drop=True)
            auc_svm[k][f] = loocv_svm(X_train,y_train,C=C_auto[k][f],g=g_auto[k][f],kernel=k)
    
    # logistic    
    auc_log = {}
    for f in fset_eval:
        idx_nan = np.where(mlrdata['rCBV'].isnull().values)[0]
        X_train = mlrdata[fset[f]].drop(idx_nan,axis=0).reset_index(drop=True)
        y_train = mlrdata['y'].drop(idx_nan,axis=0).reset_index(drop=True)
        auc_log[f] = loocv_log(X_train,y_train,C=C_auto['log'][f])



    # output


    drow_format = "{:<40}\t{:<8.2f}\t{:<8.2f}\t{:<8.2f}"
    lrow_format = "{:<40}\t{:<10}\t{:<10}\t{:<10}\n"
    with open(os.path.join(datadir,'svm_log_radnec.txt'),'w') as fp:
        print(lrow_format.format('Feature Set','SVM (rbf)','SVM (lin)','Logistic'))
        fp.write((lrow_format+'\n').format('Feature Set','SVM (rbf)','SVM (lin)','Logistic'))
        for f in fset_eval:
            print(drow_format.format(f,auc_svm['rbf'][f],auc_svm['linear'][f],auc_log[f]))
            fp.write((drow_format+'\n').format(f,auc_svm['rbf'][f],auc_svm['linear'][f],auc_log[f]))


    # check feature correlation
    corr = mlrdata[fset['all']].corr()
    sns_plot = sns.clustermap(np.abs(corr))
    sns_plot.savefig(os.path.join(datadir,'feature_correlation.png'))

    # construct roc
    plt.figure(2)
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

