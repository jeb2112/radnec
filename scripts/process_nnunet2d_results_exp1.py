# quick script to process radnec2 predictions with nnunet 2d model

# experiment 1.
# initial attempt at training RN/T segmentation from flair+,t1+ in radnec 2 cases.

import os
import json
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.spatial.distance import dice
import skimage
import imageio
import numpy as np
import nibabel as nb
import pandas as pd
import json
import copy
# import seaborn as sns
import pickle


def crop2d(cpt,img_arr,cdim = None):

    if cdim is None:
        cdim = 30
    ndim = [cdim,cdim]
    pdim = [cdim,cdim]

    dim = img_arr.shape
    for i in range(2):
        if cpt[i]-ndim[i] < 0:
            ndim[i] = cpt[i]
            pdim[i] += ndim[i]-cpt[i]
        elif cpt[i]+pdim[i] >= dim[i]:
            pdim[i] = dim[i]-cpt[i]-1
            ndim[i] += (pdim[i] - (dim[i]-cpt[i]-1))

    crop_img_arr = img_arr[cpt[0]-ndim[0]:cpt[0]+pdim[0],
                        cpt[1]-ndim[1]:cpt[1]+pdim[1],
                        :]

    return crop_img_arr

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


datadir = os.path.join('D:','data','radnec2')
predictiondir = os.path.join(datadir,'nnUNet_predictions')
labeldir = os.path.join(datadir,'nnUNet_raw/Dataset139_RadNec/labelsTs')
imagedir = os.path.join(datadir,'nnUNet_raw/Dataset139_RadNec/imagesTs')
resultsdir = os.path.join(predictiondir,'experiment1')
defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']


os.makedirs(resultsdir,exist_ok=True)
res_file = os.path.join(resultsdir,'results.pkl')
if os.path.exists(res_file):
    print('Reloading prior results')
    with open(res_file,'rb') as fp:
        (mean_res,se_res,res) = pickle.load(fp)
    prompts = list(res['dsc'].keys())

else:
    imgs = sorted([f for f in os.listdir(imagedir) if f.startswith('img')])
    preds = sorted([f for f in os.listdir(predictiondir) if f.startswith('img')])
    lbls = sorted([f for f in os.listdir(labeldir) if f.startswith('img')])
    if len(preds) != len(lbls):
        raise ValueError
    stats = {}
    for t1,flair,p,l in zip(imgs[::2],imgs[1::2],preds,lbls):
        inum = int(re.search('[0-9]{6}',t1)[0])
        case = re.search('(M|DSC)_?[0-9]*',t1)[0]
        if False:
            if inum != 4619:
                continue
        pfile = os.path.join(predictiondir,p)
        t1file = os.path.join(imagedir,t1)
        flairfile = os.path.join(imagedir,flair)
        lfile = os.path.join(labeldir,l)
        img_arr_t1 = imageio.v3.imread(t1file)
        img_arr_flair = imageio.v3.imread(flairfile)
        pred_arr = imageio.v3.imread(pfile)
        lbl_arr = imageio.v3.imread(lfile)

        lesion = {'pred':{},'lbl':{}}
        stats[inum] = {}
        for cmpt in range(1,3):
            pred_lesion = np.zeros_like(pred_arr)
            pred_lesion[np.where(pred_arr==cmpt)] = 1
            lbl_lesion = np.zeros_like(lbl_arr)
            if cmpt < 3:
                lbl_lesion[np.where(lbl_arr==cmpt)] = 1
            else:
                lbl_lesion[np.where(lbl_arr)] = 1

            if np.max(lbl_lesion) == 1:
                stats[inum][cmpt] = (1-dice(lbl_lesion.flatten(),pred_lesion.flatten()))
            else:
                stats[inum][cmpt] = np.nan

        # sample results plots
        if True:
            plt.figure(8),plt.clf()
            # img_arr_rgb = np.tile(img_arr,(1,1,3))
            img_arr_t1_rgb = skimage.color.gray2rgba(img_arr_t1)/255
            img_arr_flair_rgb = skimage.color.gray2rgba(img_arr_flair)/255
            pred_arr_rgb = skimage.color.gray2rgba(pred_arr)
            lbl_arr_rgb = skimage.color.gray2rgba(lbl_arr)
            pred_ros = np.where(pred_arr_rgb[:,:,0])
            pred_rost = np.where(pred_arr_rgb[:,:,0] == 1)
            pred_rosrn = np.where(pred_arr_rgb[:,:,0] == 2)
            lbl_ros = np.where(lbl_arr_rgb[:,:,0])
            lbl_rost = np.where(lbl_arr_rgb[:,:,0] == 1)
            lbl_rosrn = np.where(lbl_arr_rgb[:,:,0] == 2)

            fp_ros = np.where((pred_arr_rgb[:,:,0] == 1) &  (lbl_arr_rgb[:,:,0] != 1))
            fn_ros = np.where((pred_arr_rgb[:,:,0] != 1) & (lbl_arr_rgb[:,:,0] == 1))
            err_rosrn = np.where(((pred_arr_rgb[:,:,0] != 1) & (lbl_arr_rgb[:,:,0] == 2))
                                 | ((pred_arr_rgb[:,:,0] == 2) & (lbl_arr_rgb[:,:,0] == 0)))
            tptn_ros = np.where((pred_arr == lbl_arr) & (lbl_arr > 0))

            # pred_arr_rgb = pred_arr_rgb.astype(float)/2
            # lbl_arr_rgb = pred_arr_rgb.astype(float)/2
            pred_arr_rgb[pred_rost] = [0.5,0,0,0.5]
            pred_arr_rgb[pred_rosrn] = [1,0,0,0.5]
            lbl_arr_rgb[lbl_rost] = [0.0,0.5,0,0.5]
            lbl_arr_rgb[lbl_rosrn] = [0,1,0,0.5]

            err_ovly = np.copy(img_arr_flair_rgb)
            err_ovly[fp_ros] = [1,0,0,0.5] # fp is red
            err_ovly[fn_ros] = [0,0,1,0.5] # fn is blue
            err_ovly[err_rosrn] = [1,1,0,0.5] # err is yellow
            err_ovly[tptn_ros] = [0,0.5,0,0.5]

            lbl_ovly = np.copy(img_arr_flair_rgb)
            lbl_ovly[lbl_rost] = colors.to_rgb(defcolors[0]) +(0.5,)
            lbl_ovly[lbl_rosrn] = colors.to_rgb(defcolors[1]) +(0.5,)
            
            pred_ovly = np.copy(img_arr_flair_rgb)
            pred_ovly[pred_rost] = colors.to_rgb(defcolors[0]) +(0.5,)
            pred_ovly[pred_rosrn] = colors.to_rgb(defcolors[1]) +(0.5,)

            ctrd = np.mean(np.array(lbl_ros),axis=1).astype(int)
            cdim = 40
            img_arr_t1_rgb_crop = crop2d(ctrd,img_arr_t1_rgb,cdim=cdim)
            img_arr_flair_rgb_crop = crop2d(ctrd,img_arr_flair_rgb,cdim=cdim)
            err_ovly_crop = crop2d(ctrd,err_ovly,cdim = cdim)
            lbl_ovly_crop = crop2d(ctrd,lbl_ovly,cdim=cdim)
            pred_ovly_crop = crop2d(ctrd,pred_ovly,cdim = cdim)

            comp1 = np.concatenate((img_arr_t1_rgb_crop,img_arr_flair_rgb_crop,lbl_ovly_crop,pred_ovly_crop,err_ovly_crop,),axis=1)
            ax = plt.imshow(comp1)
            plt.text(4*cdim+1,5,'label',color='w')
            plt.text(6*cdim+1,5,'prediction',color='w')
            plt.text(8*cdim+1,5,'error',color='w')
            plt.axis('off')
            if False:
                plt.show(block=False)
            plt.tight_layout()
            plt.savefig(os.path.join(resultsdir,'comp','comp_' + case + '_' + str(inum) + '.png'),bbox_inches='tight',pad_inches=0.0)
            a=1

    # with open(res_file,'wb') as fp:
    #     pickle.dump((stats),fp)

mdice = {}
sedice = {}
for cmpt in range(1,3):
    mdice[cmpt] = np.nanmean([stats[k][cmpt] for k in stats.keys()])
    sedice[cmpt] = np.nanstd([stats[k][cmpt] for k in stats.keys()])

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
for i,p in enumerate(['sam_blast','unet']):
    plt.errorbar(p,m[i+2],yerr=se[i+2],fmt='+',c=defcolors[i])
ax[1,0].set_ylim((0,800))
plt.ylabel('mean time +/- se')
plt.xticks(fontsize=8)

plt.sca(ax[1,1])
ax[1,1].cla()
sdata = {k:res['speed'][k] for k in ['sam_blast','unet']}
sns.boxplot(data=sdata)
ax[1,1].set_ylim((0,800))
plt.ylabel('time (sec)')
plt.xticks(fontsize=8)

plt.tight_layout()
plt.show(block=False)
plt.savefig(os.path.join(expdir,'experiment2','results.png'))
pass