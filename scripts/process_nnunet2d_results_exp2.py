# quick script to process radnec2 predictions with nnunet 2d model
# script to assemble 2d predictions back into 3d files

# experiment 2.
# Second attempt at training RN/T segmentation from flair+,t1+ in radnec 2 cases,
# now including some studies pre-treatment for tumor-only segmentation, and
# oblique slices

import os
import json
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.spatial.distance import dice
from scipy.spatial.transform import Rotation
from scipy.ndimage import affine_transform
import skimage
import imageio
import numpy as np
import nibabel as nb
import pandas as pd
import json
import copy
# import seaborn as sns
import pickle
import cc3d



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

# write a single nifti file. use uint8 for masks 
def writenifti(img_arr,filename,header=None,norm=False,type='float64',affine=None):
    img_arr_cp = copy.deepcopy(img_arr)
    if norm:
        img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
    # using nibabel nifti coordinates
    img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
    nb.save(img_nb,filename)
    if True:
        os.system('gzip --force "{}"'.format(filename))

def recycle_dims(alist):
    while True:
        for item in alist:
            yield item

if os.name == 'posix':
    datadir = '/media/jbishop/WD4/brainmets/sunnybrook/radnec2/'
else:
    datadir = os.path.join('D:','data','radnec2')
niftidir = os.path.join(datadir,'seg')
predictiondir = os.path.join(datadir,'nnUNet_predictions','experiment2')
labeldir = os.path.join(datadir,'nnUNet_raw/Dataset139_RadNec/labelsTs')
imagedir = os.path.join(datadir,'nnUNet_raw/Dataset139_RadNec/imagesTs')
resultsdir = os.path.join(predictiondir,'comp3d')
defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# arbitrary rotation for oblique slicing, from nnunet2d_trainer_preprocess
refvec = np.ones((3))/np.sqrt(3)
target = np.array([1,0,0])
Raxis = np.cross(refvec,target)
Raxis /= np.linalg.norm(Raxis)
Rangle = np.arccos(np.dot(refvec,target))
skewsim_matrix = np.array([[0,-Raxis[2],Raxis[1]],[Raxis[2],0,-Raxis[0]],[-Raxis[1],Raxis[0],0]])
# R = np.eye(3) + np.sin(Rangle) * skewsim_matrix + (1-np.cos(Rangle)) * np.matmul(skewsim_matrix,skewsim_matrix)
r_obl = Rotation.from_rotvec(Rangle*Raxis,degrees=False).as_matrix()

# this ordering is currently hard-coded in nnunet2d_trainer_preprocess.py
olist = [(0,'obl_ax'),(1,'obl_sag'),(2,'obl_cor'),(0,'ax'),(1,'sag'),(2,'cor')]
orients = recycle_dims(olist)

os.makedirs(resultsdir,exist_ok=True)

cases = sorted(set([re.search('(M|DSC)_?[0-9u]*',f)[0] for f in os.listdir(imagedir)]))

for case in cases:

    if False: # debugging
        if case != 'M0012':
            continue

    imgs = sorted([f for f in os.listdir(imagedir) if case in f])
    preds = sorted([f for f in os.listdir(predictiondir) if case in f])
    lbls = sorted([f for f in os.listdir(labeldir) if case in f])

    studies = sorted(set([re.search('[0-9]{8}',f)[0] for f in imgs]))

    for s in studies:

        # try to load original nifti volume for reference
        imgs_nii = {}
        for ik in ['flair+','t1+','flair','t1']:
            filename = glob.glob(os.path.join(niftidir,case,s,ik+'_processed*'))
            if len(filename):
                # will use 8 bit now for png, but could be 32bit tiffs
                imgs_nii[ik],affine = loadnifti(os.path.split(filename[0])[1],os.path.join(niftidir,case,s),type='uint8')
                image_dim = np.shape(imgs_nii[ik])
                break # ie processed images are all resampled to same matrix

        lbl_T_RN = {}
        lbl_TC = {}

        # if labels are available, they can be loaded and used for error stats
        if len(lbls):

            lbl_file_TC = glob.glob(os.path.join(datadir,'seg',case,'*','mask_TC*.nii'))

            for lfile in lbl_file_TC:
                try:
                    lesion = re.search('_([1-9])',lfile).group(1)
                except AttributeError:
                    lesion = '1'
                lbl_TC[lesion],_ = loadnifti(os.path.split(lfile)[1],os.path.split(lfile)[0])
                lbl_file_T = glob.glob(os.path.join(datadir,'seg',case,'*','mask_T_' + lesion + '*.nii'))
                if len(lbl_file_T):
                    lbl_file_T = lbl_file_T[0]
                    lbl_T,_ = loadnifti(os.path.split(lbl_file_T)[1],os.path.split(lbl_file_T)[0])
                    lbl_RN = lbl_TC[lesion] - lbl_T
                else:
                    lbl_T = np.zeros_like(lbl_TC[lesion])
                    lbl_RN = lbl_TC[lesion]
                lbl_T_RN[lesion] = np.zeros_like(lbl_TC[lesion])
                lbl_T_RN[lesion][np.where(lbl_RN)] = 5# default itksnap colormap
                lbl_T_RN[lesion][np.where(lbl_T)] = 6

        else: 
            lbl_TC[1] = np.ones(image_dim)

        study_imgs = sorted([f for f in imgs if s in f])
        study_preds = sorted([f for f in preds if s in f])
        if len(lbls):
            study_lbls = sorted([f for f in lbls if s in f])
            lesions = sorted(set([re.search('_([1-9])\.',f).group(1) for f in study_preds]))
        else:
            study_lbls = [None]*len(study_imgs)
            lesions = ['1'] # for now any inference without matching labels is assumed ot produce only 1 lesion

        for lesion in lesions:

            print(case,s,lesion)

            pred_3d = {'obl_ax':None,'obl_sag':None,'obl_cor':None,'ax':None,'sag':None,'cor':None,'comp_OR':None,'comp_AND':None}
            t1_3d = {'obl_ax':None,'obl_sag':None,'obl_cor':None,'ax':None,'sag':None,'cor':None,'comp_OR':None,'comp_AND':None}
            # pred_3d = {}

            for k in pred_3d.keys():
                pred_3d[k] = np.zeros(image_dim)
                t1_3d[k] = np.zeros(image_dim)

            # in nnunet2d_trainer_preprocess, the training images are labelled with a numeric
            # linear index while cycling through the slice dimensions. in this script, the cycled dimensions
            # have to be assigned from the linear indexing as it is read back in. these conventions would 
            # be better organized in a json dict
            orient = next(recycle_dims(orients))
            pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],0,orient[0])
            islice = np.min(np.where(np.moveaxis(lbl_TC[lesion],orient[0],0))[0])
            idimold = image_dim[1:] # ie because the images were originally created and labelled starting with 'ax'

            lesion_imgs = sorted([f for f in study_imgs if '_'+lesion+'_' in f])
            lesion_preds = sorted([f for f in study_preds if '_'+lesion+'.' in f])
            lesion_lbls = sorted([f for f in study_lbls if '_'+lesion+'.' in f])
            assert(len(lesion_imgs)/2 == len(lesion_preds) == len(lesion_lbls))
            a=1

            for t1,flair,p,l in zip(lesion_imgs[::2],lesion_imgs[1::2],lesion_preds,lesion_lbls):
                inum = int(re.search('[0-9]{6}',t1)[0])
                jslice = int(re.search('[0-9]{8}_([0-9]{1,3})_',t1).group(1))
                if False:
                    print(case,s,inum,lesion)

                pfile = os.path.join(predictiondir,p)
                pred_arr = imageio.v3.imread(pfile)
                t1_arr = imageio.v3.imread(os.path.join(imagedir,t1))
                idim = np.array(np.shape(pred_arr))

                # change in the image shape means a dimension cycle has been reached. this simple device
                # is hard-coded to the fact that all images are currently registered and sampled
                # to the MNI template reference which has three unique matrix dimensions.
                if any(idim != idimold):
                        pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],0,orient[0])
                        t1_3d[orient[1]] = np.moveaxis(t1_3d[orient[1]],0,orient[0])

                        if False:
                            output_fname = os.path.join(predictiondir,'pred_3d','pred_' + case + '_' + s + '_' + orient[1] + '.nii')
                            writenifti(pred_3d[orient[1]],output_fname,affine=affine)
                        orient = next(recycle_dims(orients))
                        pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],orient[0],0)
                        t1_3d[orient[1]] = np.moveaxis(t1_3d[orient[1]],orient[0],0)
                        # islice = np.min(np.where(np.moveaxis(lbl_TC[lesion],orient[0],0))[0])+1
                        islice = np.min(np.where(lbl_TC[lesion])[orient[0]])
                        idimold = np.copy(idim)
                else:
                    pass

                if islice >= np.shape(pred_3d[orient[1]])[0]:
                    raise IndexError
                pred_3d[orient[1]][jslice] = np.copy(pred_arr)
                t1_3d[orient[1]][jslice] = np.copy(t1_arr)
                islice += 1

            # output the final 'cor' 3d. 
            pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],0,orient[0])
            t1_3d[orient[1]] = np.moveaxis(t1_3d[orient[1]],0,orient[0])
            if False:
                output_fname = os.path.join(predictiondir,'pred_3d','pred_' + case + '_' + s + '_' + orient[1] + '.nii')
                writenifti(pred_3d[orient[1]],output_fname,affine=affine)

            # de-rotate if oblique volume
            if True:
                for o in olist[:3]:
                    center = np.array(np.shape(pred_3d[o[1]]))/2
                    offset = center - np.matmul(r_obl,center)
                    pred_3d[o[1]] = affine_transform(pred_3d[o[1]],r_obl,offset=offset,order=0)

            if True: # output composite 3d
                compT_OR = np.zeros_like(pred_3d['ax'])
                compRN_OR = np.zeros_like(pred_3d['ax'])
                for _,o in olist:
                    compT_OR = (pred_3d[o]==1) | (compT_OR==1)
                    compRN_OR = (pred_3d[o]==2) | (compRN_OR==2)
                pred_3d['compOR'] = np.zeros_like(lbl_TC[lesion])
                pred_3d['compOR'][np.where(compRN_OR)] = 5 
                pred_3d['compOR'][np.where(compT_OR)] = 6 # T overwrites RN

                # if multiple lesions appear in one slice, filter out the redundant ones
                # to create output files of a single predicted lesion, to match the input
                # label files
                if True:
                    pred_3d_mask = np.zeros_like(pred_3d['compOR'])
                    pred_3d_mask[np.where(pred_3d['compOR'])] = 1
                    lbl_T_RN_mask = np.where(lbl_T_RN[lesion])
                    lbl_T_RN_ctrd = np.mean(np.array(lbl_T_RN_mask),axis=1)
                    CC_pred = cc3d.connected_components(pred_3d_mask,connectivity=26)
                    mindist = 1e6
                    for i in range(1,len(np.unique(CC_pred))):
                        ctrd = np.mean(np.array(np.where(CC_pred == i)),axis=1)
                        dist = np.linalg.norm(lbl_T_RN_ctrd-ctrd)
                        if dist < mindist:
                            mindist = np.copy(dist)
                            labeled = np.copy(i)
                            filt_3d_mask = np.copy(pred_3d_mask)
                            filt_3d_mask[np.where(CC_pred != i)] = 0
                    pred_3d['compOR']  *= filt_3d_mask

                output_fname = os.path.join(resultsdir,'pred_' + case + '_' + s + '_' + lesion + '_compOR.nii')
                writenifti(pred_3d['compOR'],output_fname,affine=affine)
                if False:
                    compRN_AND = (pred_3d['ax']==2) & (pred_3d['sag']==2) & (pred_3d['cor']==2)
                    compT_AND = (pred_3d['ax']==1) & (pred_3d['sag']==1) & (pred_3d['cor']==1)
                    pred_3d['compAND'] = np.zeros_like(lbl_TC)
                    pred_3d['compAND'][np.where(compRN_AND)] = 5
                    pred_3d['compAND'][np.where(compT_AND)] = 6 # T overwrites RN
                    output_fname = os.path.join(resultsdir,'pred_' + case + '_compAND.nii')
                    writenifti(pred_3d['compAND'],output_fname,affine=affine)

        # create composite 3d errors
        if False:
            fp_ros = np.where((pred_3d['compOR'] == 6) &  (lbl_T_RN != 6))
            fn_ros = np.where((pred_3d['compOR'] != 6) & (lbl_T_RN == 6))
            err_rosrn = np.where(((pred_3d['compOR'] != 6) & (lbl_T_RN == 5))
                                    | ((pred_3d['compOR'] == 5) & (lbl_T_RN == 0)))
            tptn_ros = np.where((pred_3d['compOR'] == lbl_T_RN) & (lbl_T_RN > 0))

            # labels according to default itksnap segmentation colorsmap
            err_ovly = np.zeros_like(lbl_T_RN)
            err_ovly[fp_ros] = 1 # fp, red
            err_ovly[fn_ros] = 3 # fn, blue
            err_ovly[err_rosrn] = 4 # RN err, yellow
            err_ovly[tptn_ros] = 2 # correct pixels, green

            output_fname = os.path.join(predictiondir,'pred_3d','err_' + case + '_compOR.nii')
            writenifti(err_ovly,output_fname,affine=affine)
            a=1


