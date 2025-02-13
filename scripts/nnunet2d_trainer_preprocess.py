# script selects random slices from each radnec segmented volume
# and copies to a nnunet file and directory format

import numpy as np
from sklearn.model_selection import train_test_split
import os
import re
import cv2
import nibabel as nb
import shutil
import matplotlib.pyplot as plt
from skimage.io import imsave
import cc3d
import random
import glob

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
        if np.max(img_arr_t1) > 255:
            img_arr_t1 = (img_arr_t1 / np.max(img_arr_t1) * 255)
        img_arr_t1 = img_arr_t1.astype(type)
    affine = img_nb_t1.affine
    return img_arr_t1,affine

def load_dataset(cpath,type='t1c'):
    ifiles = os.listdir(cpath)
    t1c_file = [f for f in ifiles if type in f][0]
    seg_file  = [f for f in ifiles if 'seg' in f][0]    
    img_arr_t1c,_ = loadnifti(t1c_file,cpath,type='float64')
    img_arr_seg,_ = loadnifti(seg_file,cpath,type='uint8')

    return img_arr_t1c,img_arr_seg

# main

if os.name == 'posix':
    datadir = "/media/jbishop/WD4/brainmets/sunnybrook/radnec2/"
else:
    datadir = "D:\\data\\radnec2\\"

segdir = os.path.join(datadir,'seg')
nnunetdir = os.path.join(datadir,'nnUNet_raw','Dataset139_RadNec')

caselist = os.listdir(os.path.join(datadir,'seg'))
cases = {}

# pre-read the segmentation dir to tally 
# T vs RN. this is needed to stratify the train/test split
finaldx = []
for c in caselist:
    cdir = os.path.join(segdir,c)
    os.chdir(cdir)
    filename = glob.glob(os.path.join('**','mask_T.nii*'))
    if filename:
        finaldx.append(1)
    else:
        finaldx.append(0)

cases['casesTr'],cases['casesTs'],y_train,y_test = train_test_split(caselist,finaldx,stratify=finaldx,test_size=0.2,random_state=42)

img_idx = 1

for ck in cases.keys():
    print(ck)

    output_imgdir = os.path.join(nnunetdir,ck.replace('cases','images'))
    output_lbldir = os.path.join(nnunetdir,ck.replace('cases','labels'))
    try:
        shutil.rmtree(output_imgdir)
        shutil.rmtree(output_lbldir)
    except FileNotFoundError:
        pass
    os.makedirs(output_imgdir,exist_ok=True)
    os.makedirs(output_lbldir,exist_ok=True)

    for c in cases[ck]:
        print('case = {}'.format(c))

        if False: #debugging
            if c != 'M0066':
                continue

        cdir = os.path.join(segdir,c)
        os.chdir(cdir)
        masks = {}
        imgs = {}

        for tk in ['TC','T']:
            try:
                filename = glob.glob(os.path.join('**','mask_'+tk+'.nii*'))[0]
                masks[tk],_ = loadnifti(os.path.split(filename)[1],os.path.join(cdir,os.path.split(filename)[0]),type='uint8')
            except IndexError:
                if tk == 'T': # currently, if no tumor has been segmented, then there is no T mask file at all.
                    masks['T'] = np.zeros_like(masks['TC'])
                else:
                    raise FileNotFoundError('TC mask file not found')
            # masks[tk][masks[tk] == 255] = 0
        # no easy way to display low contrast mask in a ping for spot verification. can't use anything
        # non-continguous like 127,255. so, for spot checking the pings can temporarily run this code using 127,255
        # but otherwise use 0,1,2 for nnunet. 

        # check for error pixels. according to convention, 'T' should be entirely 
        # a subset of 'TC'
        errpixels = np.where(masks['TC'].astype(int) - masks['T'].astype(int) < 0)[0]
        if len(errpixels):
            masks['TC'] = masks['TC'] | masks['T']
            print('error mask pixels detected, correcting...')
        masks['lbl'] = 1*masks['T'] + 2*(masks['TC'] - masks['T'])

        if np.any(masks['lbl'] > 2):
            raise ValueError

        imgs = {}
        for ik in ['flair+','t1+']:
            filename = glob.glob(os.path.join('**',ik+'_processed*'))[0]
            # will use 8 bit now for png, but could be 32bit tiffs
            imgs[ik],_ = loadnifti(os.path.split(filename)[1],os.path.join(cdir,os.path.split(filename)[0]),type='uint8')

        pset = np.where(masks['lbl'])
        npixels = len(pset[0])

        for dim in range(3):
            slices = np.unique(pset[dim])
            for slice in slices:
                imgslice = {}
                lblslice = np.moveaxis(masks['lbl'],dim,0)[slice]
                if np.max(lblslice) > 2:
                    raise ValueError
                for ik in ('flair+','t1+'):
                    imgslice[ik] = np.moveaxis(imgs[ik],dim,0)[slice]
                if len(np.where(lblslice)[0]) > 49:
                    fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '.png'
                    imsave(os.path.join(output_lbldir,fname),lblslice,check_contrast=False)
                    # cv2.imwrite(os.path.join(output_lbldir,fname),lblslice)
                    for ktag,ik in zip(('0003','0001'),('flair+','t1+')):
                        fname = 'img_' + str(img_idx).zfill(6) + '_' + c + '_' + ktag + '.png'
                        imsave(os.path.join(output_imgdir,fname),imgslice[ik],check_contrast=False)
                    img_idx += 1
        
    a=1   
