# quick script to process stats.json results from SAM viewer

# experiment 2.
# 15 random cases, 1 or more lesions per case
# compare speed and accuracy of 3d mult-slice SAM to 3d nnUNet
# also include 2d point prompt result from the clicked slice in each lesion.

import os
import json
import matplotlib.pyplot as plt
import imageio
import numpy as np
import nibabel as nb
import pandas as pd
import json
import copy
import seaborn as sns
import pickle

# load a single nifti file
def loadnifti(t1_file,dir,type=None):
    img_arr_t1 = None
    try:
        img_nb_t1 = nb.load(os.path.join(dir,t1_file))
    except IOError as e:
        print('Can\'t import {}'.format(t1_file))
        return None,None
    nb_header = img_nb_t1.header.copy()
    # nibabel convention will be transposed to sitk convention
    img_arr_t1 = np.transpose(np.array(img_nb_t1.dataobj),axes=(2,1,0))
    if type is not None:
        img_arr_t1 = img_arr_t1.astype(type)
    affine = img_nb_t1.affine

    return img_arr_t1,affine


datadir = '/media/jbishop/WD4/brainmets/sunnybrook/metastases/SAM_BraTS_2024/brats2nifti'
expdir = '/media/jbishop/WD4/brainmets/sunnybrook/metastases/SAM_BraTS_2024/experiments'

res_file = os.path.join(expdir,'experiment2','results.pkl')
if os.path.exists(res_file):
    with open(res_file,'rb') as fp:
        (mean_res,se_res,res) = pickle.load(fp)
    prompts = list(res['dsc'].keys())

else:
    casedirs = os.listdir(datadir)
    casedirs = sorted([c for c in casedirs if c.startswith('M')])
    casedirs2 = copy.copy(casedirs)
    d = {} # blast/sam data
    u = {} # unet data
    for c in casedirs:
        studydirs = os.listdir(os.path.join(datadir,c))
        studydirs = [s for s in studydirs if s.startswith('S')]
        for s in studydirs:
            sdir = os.path.join(datadir,c,s)
            print(sdir)
            statsfile = os.path.join(sdir,'stats.json')
            if os.path.exists(statsfile):
                with open(statsfile,'r') as fp:
                    sdict = json.load(fp)
                d[c] = sdict
            else:
                print('No stats file, skipping...')
                casedirs2.remove(c)
            statsfile_unet = os.path.join(sdir,'stats_unet.json')
            if os.path.exists(statsfile_unet):
                with open(statsfile_unet,'r') as fp:
                    sdict_unet = json.load(fp)
                u[c] = sdict_unet
            else:
                print('No unet stats file, skipping...')
                casedirs2.remove(c)

    casedirs = casedirs2
    prompts = list(d[c].keys())
    res = {'dsc':{p:[] for p in prompts},'speed':{p:[] for p in prompts},'coords':[]}
    res['dsc']['unet'] = []
    res['speed']['unet'] = []
    for c in casedirs:
        print(c)
        # coordinates are potentially different in the nnUNet and blast/SAM spaces
        # reload the nifti to check. 
        _,affine = loadnifti('ET_blast.nii.gz',os.path.join(datadir,c,'S0001'))
        b1 = np.reshape(affine[:3,3],(-1,1))
        _,affine2 = loadnifti('ET_unet.nii.gz',os.path.join(datadir,c,'S0001'))
        b2 = np.reshape(affine2[:3,3],(-1,1))

        for p in prompts:
            for r in d[c][p].keys(): # list or roi's for this case
                res['dsc'][p].append(d[c][p][r]['stats']['dsc']['TC'])
                res['speed'][p].append(d[c][p][r]['stats']['elapsedtime'])
                if p == 'SAM2d_point':    
                    slice = list(d[c][p][r]['bbox'].keys())[0]
                    # bbox coords were stored x,y,slice
                    coords =  np.reshape(np.array(d[c][p][r]['bbox'][slice]['p0']+[int(slice)] ),(-1,1))
                    if not all(b1 == b2):
                        coords = np.matmul(affine[:3,:3],(coords-b1)) - b2                  # ucoords = ucoords[[1,2,0]]
                # find matching roi in unet
                    cmin = 1000
                    unet_rois = [k for k in u[c].keys() if 'roi' in k]
                    for r2 in unet_rois:
                        # unet coords were stored slice,y,x so flip
                        if u[c][r2]['coords_gt']['TC'] is not None:
                            ucoords = np.reshape(np.flip(np.array(u[c][r2]['coords_gt']['TC'])),(-1,1))
                        else:
                            continue # ET/TC is not 1:1 correspondence
                                    # if there is no TC for this roi1, skip it. this may need further work.

                        cdist = np.linalg.norm(ucoords - coords)
                        if cdist < cmin:
                            cmin = np.copy(cdist)
                            rmin = copy.copy(r2)
                    # if u[c][r2]['dsc']['TC'] is None:
                    #     continue
                    res['dsc']['unet'].append(u[c][rmin]['dsc']['TC'])
                    res['speed']['unet'].append(u[c]['elapsed_time'])
                    pass

    mean_res = {'dsc':{},'speed':{}}
    se_res = {'dsc':{},'speed':{}}
    prompts = list(res['dsc'].keys())
    for stat in ['dsc','speed']:
        for p in prompts:
            mean_res[stat][p] = np.mean(res[stat][p])
            se_res[stat][p] = np.std(res[stat][p]) / np.sqrt(len(res[stat][p]))

    with open(res_file,'wb') as fp:
        pickle.dump((mean_res,se_res,res),fp)


pass
defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig,ax = plt.subplots(2,2,figsize=(6,6))
# plt.clf()
m = [mean_res['dsc'][k] for k in prompts]
se = [se_res['dsc'][k] for k in prompts]
plt.sca(ax[0,0])
ax[0,0].cla()
for i,p in enumerate(prompts):
    plt.errorbar(p,m[i],yerr=se[i],fmt='+',c=defcolors[i])
ax[0,0].set_ylim((0,1))
plt.ylabel('mean DSC+/- se')
plt.xticks(fontsize=8)

plt.sca(ax[0,1])
ax[0,1].cla()
sdata = {k:res['dsc'][k] for k in prompts}
sns.boxplot(data=sdata)
plt.ylabel('DSC')
plt.xticks(fontsize=8)

m = [mean_res['speed'][k] for k in prompts]
se = [se_res['speed'][k] for k in prompts]
plt.sca(ax[1,0])
ax[1,0].cla()
for i,p in enumerate(['SAM_bbox','unet']):
    plt.errorbar(p,m[i+2],yerr=se[i+2],fmt='+',c=defcolors[i])
ax[1,0].set_ylim((0,800))
plt.ylabel('mean time +/- se')
plt.xticks(fontsize=8)

plt.sca(ax[1,1])
ax[1,1].cla()
sdata = {k:res['speed'][k] for k in ['SAM_bbox','unet']}
sns.boxplot(data=sdata)
ax[1,1].set_ylim((0,800))
plt.ylabel('time (sec)')
plt.xticks(fontsize=8)

plt.tight_layout()
plt.show(block=False)
plt.savefig(os.path.join(expdir,'experiment2','results.png'))
pass