# Plot segmented volumes from INDIGO Vorasidenib longitudinal study
# Jan 2, 2024


import numpy as np
import os
import nibabel as nb
import re
import matplotlib.pyplot as plt
from matplotlib import ticker
import scipy
from datetime import datetime as dt
import time
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle


def voxel_volume(img_arr):
    n = np.shape(np.where(img_arr > 0))[1]
    return n

def fitexp(x,a,b,c):
    return a * np.exp(-b*x) + c
def fitlin(x,a,b):
    return a*x + b

# Return an axes of confidence bands using the theoretical approximation.
def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None, color=None,alpha=1):
    
    if color is None:
        color = "#b9cfe7"
    if ax is None:
        ax = plt.gca()
    ci = t * s_err * np.sqrt(1/n + (x2 - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color=color, edgecolor=None,alpha=alpha)

    return ax

# get (1-alpha) 2-sided confidence intervals for sklearn.LinearRegression coefficients
def get_conf_int(alpha, lr, X, y):
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    coefs = np.r_[[lr.intercept_], lr.coef_]
    X_aux = X.copy()
    X_aux.insert(0, 'const', 1)
    dof = -np.diff(X_aux.shape)[0]
    mse = np.array(np.sum((y - lr.predict(X)) ** 2) / dof)
    var_params = np.diag(np.linalg.inv(X_aux.T.dot(X_aux)))
    t_val = scipy.stats.t.isf(alpha/2, dof)
    gap = np.reshape(t_val * np.sqrt(mse * var_params),(-1,1))

    return coefs - gap, coefs + gap




##############
# prepare data
##############

dsets = {'blast':{'dir':None,'vv':None},'slicer':{'dir':None,'vv':None},'itk':{'dir':None,'vv':None},'unet':{'dir':None,'vv':None}}
dset_keys = dsets.keys()

data_dir = '/media/jbishop/WD4/brainmets/raw/Dataset137_BraTS2021'
for k in dset_keys:
    dsets[k]['dir'] = os.path.join(data_dir,'results_'+k)
output_dir = os.path.join(data_dir,'plots')
datafile = os.path.join(output_dir,'indigo_volumes.pkl')
ntime = 11

if os.path.exists(datafile):
    t = time.ctime(os.path.getmtime(datafile))
    print('\n\nloading {}, last modified {}...\n\n'.format(datafile,t))
    with open(datafile,'rb') as fp:
        # (dx,tags,slicer_vv,unet_pre_vv,blast_vv,vv,itk_vv) = pickle.load(fp)    
        (dx,tags,dsets) = pickle.load(fp)    

else:

    tags=[]
    xtags = range(ntime)
    dx = np.zeros(ntime,dtype='float64')
    for k in ['unet','slicer','blast','itk']:
        print('\n{}\n'.format(k))
        files = os.listdir(dsets[k]['dir'])
        files = [f for f in files if 'nii' in f]
        files = sorted(files,key=lambda x:(re.match('^.*([0-9]{4})',x).group(1),re.match('^([0-9]{2})',x).group(1)))
        # skip 02_23_2021 result
        if k == 'unet':
            files = files[1:]
            d0 = dt.strptime(re.match('^(.*)\.nii',files[0]).group(1),'%m_%d_%Y').toordinal()
        dsets[k]['vv'] = np.zeros(ntime,dtype='float64')

        for i,f in enumerate(files):
            if k == 'unet':
                tags.append(re.match('(^.*)\.nii',f).group(1))
                dx[i] = dt.strptime(tags[i],'%m_%d_%Y').toordinal()-d0
            img_nb = nb.load(os.path.join(dsets[k]['dir'],f))
            img_arr = np.array(img_nb.dataobj)
            # remove non-zero background
            if k == 'slicer':
                img_arr[img_arr == 2] = 0
            dsets[k]['vv'][i] = voxel_volume(img_arr)/1000
            print('dataset {}, volume = {}'.format(f,dsets[k]['vv'][i]))

    with open(os.path.join(output_dir,'indigo_volumes.pkl'),'wb') as fp:
        pickle.dump((dx,tags,dsets),fp)



###############
# plot raw data
###############
            
# overlay figure
fig1 = plt.figure(1)
ax = fig1.add_subplot(111)
plt.plot(dx,dsets['unet']['vv'],'r',markersize=6,marker='o',linestyle='')
plt.plot(dx,dsets['blast']['vv'],'b',markersize=6,marker='^',linestyle='')
plt.plot(dx,dsets['slicer']['vv'],'g',markersize=6,marker='s',linestyle='None')
plt.xticks(ticks=dx,labels=tags,rotation=45)
ax = plt.gca()
ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
# ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
ax.yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=True))
ax.ticklabel_format(axis='y',style='plain',useOffset=False)
plt.ylabel('volume (cm^3)')
plt.xlim((dx[0]-10,dx[-1]+10))
plt.ylim((5,25))
plt.tight_layout()

# separate plots
fig5 = plt.figure(5)
ax1 = fig5.add_subplot(1,3,1)
plt.plot(dx,dsets['unet']['vv'],'r',markersize=6,marker='o',linestyle='')
plt.ylim((5,25))
plt.ylabel('volume (cm^3)')
ax2 = fig5.add_subplot(1,3,2)
plt.plot(dx,dsets['blast']['vv'],'b',markersize=6,marker='^',linestyle='')
plt.ylim((5,25))
plt.xlabel('time (days)')
ax2.set_yticklabels([])
ax3 = fig5.add_subplot(1,3,3)
plt.plot(dx,dsets['slicer']['vv'],'g',markersize=6,marker='s',linestyle='')
plt.ylim((5,25))
ax3.set_yticklabels([])



#########
# do fits
#########

w = list(np.ones(10)) + [1]
xfit = np.linspace(0,dx[-1],100)
xdata = dx

s_err = {}
lfit = {}
r2 = {}
yfit = {}
for k in dset_keys:
    if False:
        # not using splines for now
        sfit = scipy.interpolate.UnivariateSpline(xdata,dsets[k]['vv'],w=w)
        sfit.set_smoothing_factor(50)
    if False:
        # fit exp directly wasn't reliable on these datasets
        efit,cov = scipy.optimize.curve_fit(fitexp,xdata,dsets[k]['vv'])
    # fit log ydata to linear instead
    log_ydata = np.log(dsets[k]['vv'])
    lfit[k],cov = scipy.optimize.curve_fit(fitlin,xdata,log_ydata)
    log_ydata_hat = fitlin(xdata,*lfit[k])
    resid_exp = dsets[k]['vv'] - np.exp(log_ydata_hat)
    resid = log_ydata - log_ydata_hat 
    resid0 = log_ydata - np.mean(log_ydata)                       # residuals; diff. actual data from predicted values
    r2[k] = 1 - np.sum(resid**2) / np.sum(resid0**2)                       # chi-squared; estimates error in data
    yfit[k] = np.exp(fitlin(xfit,*lfit[k]))

    dof = ntime - lfit[k].size
    t = scipy.stats.t.ppf(0.975, dof)                           # ie for two-sided 95%
    s_err[k] = np.sqrt(np.sum(resid_exp**2) / dof)     

    if False:
        # using sklearn
        xdata = dx.reshape(-1,1)
        log_ydata = np.log(dsets['blast']['vv']).reshape(-1,1)
        lfit = LinearRegression().fit(xdata,log_ydata)
        ci_lower,ci_upper = get_conf_int(0.05, lfit, xdata, log_ydata)    



###############
# plot the fits
###############
        
plt.figure(fig1)
plt.plot(xfit,np.exp(fitlin(xfit,*lfit['slicer'])),'g')
plt.plot(xfit,np.exp(fitlin(xfit,*lfit['unet'])),'r')
plt.plot(xfit,np.exp(fitlin(xfit,*lfit['blast'])),'b')
ax=plt.gca()
plot_ci_manual(t,s_err['slicer'],ntime,xdata,xfit,yfit['slicer'],ax=ax,color="#e6f1e5")
plot_ci_manual(t,s_err['blast'],ntime,xdata,xfit,yfit['blast'],ax=ax,color='#b9cfe7',alpha=0.5)
plot_ci_manual(t,s_err['unet'],ntime,xdata,xfit,yfit['unet'],ax=ax,color="#eac4c4",alpha=0.5)
plt.text(400,22,'BLAST, r$^2$ = {:.2f}'.format(r2['blast']),backgroundcolor='#b9cfe7',fontsize='small')
plt.text(400,20,'UNet, r$^2$ = {:.2f}'.format(r2['unet']),backgroundcolor="#eac4c4",fontsize='small')
plt.text(400,18,'3DSlicer, r$^2$ = {:.2f}'.format(r2['slicer']),backgroundcolor="#e6f1e5",fontsize='small')

plt.figure(fig5)
ax1.plot(xfit,np.exp(fitlin(xfit,*lfit['unet'])),'r')
ax1.text(240,23,'UNet, r$^2$ = {:.2f}'.format(r2['unet']),backgroundcolor="#eac4c4",fontsize='small')
plot_ci_manual(t,s_err['unet'],ntime,xdata,xfit,yfit['unet'],ax=ax1,color="#eac4c4",alpha=0.5)
ax2.plot(xfit,np.exp(fitlin(xfit,*lfit['blast'])),'b')
ax2.text(200,23,'BLAST, r$^2$ = {:.2f}'.format(r2['blast']),backgroundcolor='#b9cfe7',fontsize='small')
plot_ci_manual(t,s_err['blast'],ntime,xdata,xfit,yfit['blast'],ax=ax2,color='#b9cfe7',alpha=0.5)
ax3.plot(xfit,np.exp(fitlin(xfit,*lfit['slicer'])),'g')
ax3.text(120,23,'3DSlicer, r$^2$ = {:.2f}'.format(r2['slicer']),backgroundcolor="#e6f1e5",fontsize='small')
plot_ci_manual(t,s_err['slicer'],ntime,xdata,xfit,yfit['slicer'],ax=ax3,color="#e6f1e5",alpha=0.5)
    

########
# output
########

plt.figure(fig1)
plt.savefig(os.path.join(output_dir,'voxel_volume2.png'))

plt.figure(fig5)
plt.savefig(os.path.join(output_dir,'voxel_volume2_separate.png'))
