# quick script to process radnec2 predictions with nnunet 2d model
# additional script to assemble 2d predictions back into 3d files

# experiment 1.
# initial attempt at training RN/T segmentation from flair+,t1+ in radnec 2 cases.

import os
import json
import glob
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
niftidir = os.path.join(datadir,'dicom2nifti')
predictiondir = os.path.join(datadir,'nnUNet_predictions')
labeldir = os.path.join(datadir,'nnUNet_raw/Dataset139_RadNec/labelsTs')
imagedir = os.path.join(datadir,'nnUNet_raw/Dataset139_RadNec/imagesTs')
resultsdir = os.path.join(predictiondir,'experiment1')
defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']

olist = [(0,'ax'),(1,'sag'),(2,'cor')]
orients = recycle_dims(olist)

os.makedirs(resultsdir,exist_ok=True)

cases = sorted(set([re.search('(M|DSC)_?[0-9]*',f)[0] for f in os.listdir(imagedir)]))

for case in cases:

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

        if len(lbls):

            lbl_file_TC = glob.glob(os.path.join(datadir,'seg',case,'*','mask_TC.nii'))[0]
            lbl_TC,affine = loadnifti(os.path.split(lbl_file_TC)[1],os.path.split(lbl_file_TC)[0])
            lbl_file_T = glob.glob(os.path.join(datadir,'seg',case,'*','mask_T.nii'))
            if len(lbl_file_T):
                lbl_file_T = lbl_file_T[0]
                lbl_T,affine = loadnifti(os.path.split(lbl_file_T)[1],os.path.split(lbl_file_T)[0])
                lbl_RN = lbl_TC - lbl_T
            else:
                lbl_T = np.zeros_like(lbl_TC)
                lbl_RN = lbl_TC
            lbl_T_RN = np.zeros_like(lbl_TC)
            lbl_T_RN[np.where(lbl_RN)] = 5# default itksnap colormap
            lbl_T_RN[np.where(lbl_T)] = 6

        else:
            lbl_TC = np.ones(image_dim)

        pred_3d = {'ax':None,'sag':None,'cor':None,'comp_OR':None,'comp_AND':None}

        # t1_file = glob.glob(os.path.join(datadir,'seg',case,'*','t1+_processed.nii*'))[0]
        # t1_arr,_ = loadnifti(os.path.split(t1_file)[1],os.path.split(t1_file)[0])    

        for k in pred_3d.keys():
            pred_3d[k] = np.zeros(image_dim)

        orient = next(recycle_dims(orients))
        pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],0,orient[0])
        islice = np.min(np.where(np.moveaxis(lbl_TC,orient[0],0))[0])
        idimold = image_dim[1:] # ie because the images were originally created and labelled starting with 'ax'

        study_imgs = sorted([f for f in imgs if s in f])
        study_preds = sorted([f for f in preds if s in f])
        if len(lbls):
            study_lbls = sorted([f for f in lbls if s in f])
        else:
            study_lbls = [None]*len(study_imgs)


        for t1,flair,p,l in zip(study_imgs[::2],study_imgs[1::2],study_preds,study_lbls):
            inum = int(re.search('[0-9]{6}',t1)[0])
            print(case,s,inum)

            pfile = os.path.join(predictiondir,p)
            pred_arr = imageio.v3.imread(pfile)
            idim = np.array(np.shape(pred_arr))

            if any(idim != idimold):
                    pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],0,orient[0])
                    output_fname = os.path.join(predictiondir,'experiment1','pred_3d','pred_' + case + '_' + s + '_' + orient[1] + '.nii')
                    if False:
                        writenifti(pred_3d[orient[1]],output_fname,affine=affine)
                    orient = next(recycle_dims(orients))
                    pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],orient[0],0)
                    islice = np.min(np.where(np.moveaxis(lbl_TC,orient[0],0))[0])
                    idimold = np.copy(idim)
            else:
                pass

            if islice >= np.shape(pred_3d[orient[1]])[0]:
                raise IndexError
            pred_3d[orient[1]][islice] = np.copy(pred_arr)
            islice += 1

        # output the final 'cor' 3d
        pred_3d[orient[1]] = np.moveaxis(pred_3d[orient[1]],0,orient[0])
        output_fname = os.path.join(predictiondir,'experiment1','pred_3d','pred_' + case + '_' + s + '_' + orient[1] + '.nii')
        if False:
            writenifti(pred_3d[orient[1]],output_fname,affine=affine)

        if True: # output composite 3d
            compT_OR = (pred_3d['ax']==1) | (pred_3d['sag']==1) | (pred_3d['cor']==1)
            compRN_OR = (pred_3d['ax']==2) | (pred_3d['sag']==2) | (pred_3d['cor']==2)
            pred_3d['compOR'] = np.zeros_like(lbl_TC)
            pred_3d['compOR'][np.where(compRN_OR)] = 5 
            pred_3d['compOR'][np.where(compT_OR)] = 6 # T overwrites RN
            output_fname = os.path.join(predictiondir,'experiment1','pred_3d','pred_' + case + '_' + s + '_compOR.nii')
            writenifti(pred_3d['compOR'],output_fname,affine=affine)
            if False:
                compRN_AND = (pred_3d['ax']==2) & (pred_3d['sag']==2) & (pred_3d['cor']==2)
                compT_AND = (pred_3d['ax']==1) & (pred_3d['sag']==1) & (pred_3d['cor']==1)
                pred_3d['compAND'] = np.zeros_like(lbl_TC)
                pred_3d['compAND'][np.where(compRN_AND)] = 5
                pred_3d['compAND'][np.where(compT_AND)] = 6 # T overwrites RN
                output_fname = os.path.join(predictiondir,'experiment1','pred_3d','pred_' + case + '_compAND.nii')
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

            output_fname = os.path.join(predictiondir,'experiment1','pred_3d','err_' + case + '_compOR.nii')
            writenifti(err_ovly,output_fname,affine=affine)
            a=1


