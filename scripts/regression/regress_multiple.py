# script to check field strength relationship to FLAIR signal in RAD NEC data

# reads multiple regression sheet from master radnec spreadsheet, populates the mean pixel values,
# and saves to a separate .xlsx file to streamline the processing.

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import pickle
import sys
import pandas as pd
import seaborn as sb
import scipy
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

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

if os.system('mountpoint -q /mnt/D') != 0:
    # do this manually until i get key auth figured out
    # os.system('sudo mount -t cifs //192.168.50.224/D /mnt/D -o rw,uid=jbishop,gid=jbishop,username=Chris\ Heyn\ Lab,vers=3.0')
    a=1
if False:
    radnecDir = '/mnt/D/Dropbox/BLAST DEVELOPMENT/RAD NEC/MANUSCRIPT/'
    localDir = '/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC/regression'
else:
    radnecDir = '/media/jbishop/WD4/brainmets/sunnybrook/RAD NEC/'
    localDir = radnecDir

resultsfile = os.path.join(localDir,'regression_results.pkl')

# load master spreadsheet
df = pd.read_excel(os.path.join(radnecDir,'02_15_2024 RAD NEC MET.xlsx'),sheet_name='Multiple Linear Regression')

# load blast data. This is the order of feature vectors in the matlab file 
# saved for python
# 1-4: idxnames = {'FLAIR_ET','FLAIR_ET_nonbias','T1ce_ET','T1ce_ET_nonbias','ADC'};
# 5-6: idxnames_raw = {'FLAIR_ET_raw','T1ce_ET_raw'};

d = loadmat(os.path.join(radnecDir,'logistic','radnec_processed_blast_python.mat'))
dset_T = d['fits_T_mu'].T
dset_RN = d['fits_RN_mu'].T
if False:
    cats = ['tumour','RN']
    dset_vals = np.column_stack(dset_T[:,1],dset_RN[:,1])
    plt.scatter(cats*23,dset_vals)
    plt.show()
n_T = np.shape(dset_T)[0]
n_RN = np.shape(dset_RN)[0]

# copy blast values to excel daframe columns
idx_T = np.where(df['Diagnosis (1=RN 0=T)'] == 0)
idx_RN = np.where(df['Diagnosis (1=RN 0=T)'] == 1)
# loc syntax for SettingWithCopywarning doesn't work
df['Mean blast flair et bias'][idx_T] = d['fits_T_mu'][0,:]
df['Mean blast flair et bias'][idx_RN] = d['fits_RN_mu'][0,:]
df['Mean blast flair et nonbias'][idx_T] = d['fits_T_mu'][1,:]
df['Mean blast flair et nonbias'][idx_RN] = d['fits_RN_mu'][1,:]
df['Mean raw flair et nonbias'][idx_T] = d['fits_T_mu'][5,:]
df['Mean raw flair et nonbias'][idx_RN] = d['fits_RN_mu'][5,:]

# save data
if False:   
    df.to_excel(os.path.join(localDir,'logistic','regression_output.xlsx'),index=False)
    
#Setting the value for X and Y
if True:
    x = df[['Field Strength (1=3T 0=1.5T)', 'Diagnosis (1=RN 0=T)']] 
else:
    x = df[['Field Strength (0=3T 1=1.5T)', 'Diagnosis (1=RN 0=T)']] 
x = sm.add_constant(x)
y = df['Mean blast flair et bias']

#Fitting the Multiple Linear Regression model
if False:
    mlr = LinearRegression()   
    mlr.fit(x, y)
    #Intercept and Coefficient
    print("Intercept: ", mlr.intercept_) 
    print("Coefficients:") 
    print(list(zip(x, mlr.coef_)))
    print('R squared: {:.2f}'.format(mlr.score(x,y))) 
else:
    mlr = sm.OLS(y,x).fit()
    print(mlr.summary())


#Model Evaluation
y_pred_mlr= mlr.predict(x)
meanAbErr = metrics.mean_absolute_error(y, y_pred_mlr) 
meanSqErr = metrics.mean_squared_error(y, y_pred_mlr) 
rootMeanSqErr = np.sqrt(metrics.mean_squared_error(y, y_pred_mlr))
print('Mean Absolute Error:', meanAbErr) 
print('Mean Square Error:', meanSqErr) 
print('Root Mean Square Error:', rootMeanSqErr)

# plot results

if False:
    sm.graphics.plot_partregress(df['Mean blast flair et bias'],df['Field Strength (1=3T 0=1.5T)'],df['Diagnosis (1=RN 0=T)'],obs_labels=True)
    sm.graphics.plot_partregress(df['Mean blast flair et bias'],df['Diagnosis (1=RN 0=T)'],df['Field Strength (1=3T 0=1.5T)'],obs_labels=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x['Field Strength (1=3T 0=1.5T)'], x['Diagnosis (1=RN 0=T)'], y)
    plt.xlabel('field strength')
    plt.ylabel('diagnosis')


#Fitting the Multiple Logistic Regression model
x = df[['Field Strength (1=3T 0=1.5T)', 'Mean blast flair et bias']] 
x = sm.add_constant(x)
y = df['Diagnosis (1=RN 0=T)']
mlogitr = sm.Logit(y,x).fit()
m = mlogitr.summary()
print(mlogitr.summary())
with open(os.path.join(localDir,'logit_summary.txt'),'w') as fp:
    fp.write(mlogitr.summary().as_text())


