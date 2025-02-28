# script processes nnunet results from test cases
# and picks out poor results with DSC < 0.8

from multiprocessing import Pool
import nibabel as nb
from nibabel.processing import resample_from_to
import numpy as np
import matplotlib.pyplot as plt
import shutil
import re
import os
import ants
import copy
from sklearn.cluster import KMeans
import subprocess
from scipy.spatial.distance import dice,directed_hausdorff
import cc3d
import json

# reproducing the same cc3d connected components as in 
# brats_nifti.py such that the roi numbering is the same, 
# calculate new segmentation masks for only 
# the lesions which nnUNet missed detecting or Dice < 0.8.
# eventually this functionality should be moved into brats_nifti.py
# for any new dataset being processed from the top.
def new_mask(groundtruth,roistats):
      
    dtlabel = {'ET':3,'TC':2}
    new_mask = np.zeros_like(groundtruth['ET'],dtype='uint8')
    # pull out the matching lesion using cc3d
    n_gt = 1
    for dt in ['TC']:
        gt_mask = np.copy(groundtruth[dt])
        CC_gt_labeled = cc3d.connected_components(gt_mask,connectivity=26)
        n_gt = max(n_gt,len(np.unique(CC_gt_labeled)))

    missed = False
    for i in range(1,n_gt):

        # not doing 'wt for now
        for dt in ['TC','ET']:

            gt_mask = np.copy(groundtruth[dt])
            CC_gt_labeled = cc3d.connected_components(gt_mask,connectivity=26)

            if i in np.unique(CC_gt_labeled):
                if roistats['roi'+str(i)]['dsc'][dt] < 0.8:

                    gt_lesion = (CC_gt_labeled == i).astype('uint8')
                    lesion_roi = np.where(gt_lesion)
                    new_mask[lesion_roi] = 1 * dtlabel[dt]
                    missed = True

    if missed:
        return new_mask
    else:
        return None


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


######
# main
######

if __name__ == '__main__':

    if os.name == 'posix':
        # nifti destination dir for BLAST
        blast_data_dir = '/media/jbishop/WD4/brainmets/sunnybrook/metastases/BraTS_2024'
    elif os.name == 'nt':
        # nifti destination dir for BLAST
        blast_data_dir = os.path.join(os.path.expanduser('~'),'data','radnec_sam')

    cases = sorted(os.listdir(blast_data_dir))
    cases = [c for c in cases if c.startswith('M')]

    # edit range of cases to process here
    missedcaselist = []
    for C in cases:
        print('case {}'.format(C))
        casenum = re.search('([0-9]{5})',C).group(1)
        ddir = os.path.join(blast_data_dir,C,'S0001')
        files = os.listdir(ddir)
        seg_fname = [f for f in files if 'seg' in f][0]

        # load ground truth seg
        gt_mask = {}
        gt_seg,affine = loadnifti(seg_fname,ddir,type='uint8')
        gt_mask['ET'] = gt_seg == 3
        gt_mask['TC'] = (gt_seg == 1) | (gt_seg == 3)

        # load stats 
        filename = os.path.join(ddir,'stats_unet.json')
        with open(filename,'r') as fp:
            roistats = json.load(fp)

        new_segmask = new_mask(gt_mask,roistats)
        if new_segmask is not None:
            newseg_fname = seg_fname.replace('seg','seg_dice').removesuffix('.gz')
            writenifti(new_segmask,os.path.join(ddir,newseg_fname),type='uint8',affine=affine)
            missedcaselist.append(C)

    with open(os.path.join(blast_data_dir,'nnunet_missedtestcases.txt'),'w') as fp:
        fp.write('\n'.join(missedcaselist))
