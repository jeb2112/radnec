# script selects two random slices from each BraTS 2024 volume
# for use in fine-tuning the sam
# in addition, just one met is selected from each slice.
# in the future, if fine-tuning sam works, this can be scaled to use all slices and all mets

# note. the tkagg backend of matplotlib in the pytorch env on windows but not linux is somehow broken
# and interactive plotting does not work on windows. but, it did work once in another script
# after resetting matplotlib.use('tkagg') to the very same default backend it already was. 
# but that only worked once, and couldn't be repeated. instead, if on windows run this script in blast or
# other env if debugging interactive plots are needed.

import numpy as np
import skimage
import random
from sklearn.model_selection import train_test_split
import PIL
import os
import re
import nibabel as nb
import shutil
import matplotlib
import matplotlib.pyplot as plt
from skimage.io import imsave
import cc3d
# matplotlib.use('TkAgg')

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

def load_dataset(cpath):
    ifiles = os.listdir(cpath)
    t1c_file = [f for f in ifiles if 't1c' in f][0]
    seg_file  = [f for f in ifiles if 'seg' in f][0]    
    img_arr_t1c,_ = loadnifti(t1c_file,cpath,type='float64')
    img_arr_seg,_ = loadnifti(seg_file,cpath,type='uint8')

    return img_arr_t1c,img_arr_seg

# main

datadir = "C:\\Users\\Chris Heyn Lab\\data\\brats2024\\raw"

with open(os.path.join(datadir,'BraTS_2024_testcases.txt'),'r') as fp:
    testcases = list(map(int,[re.search('([0-9]{5})',c).group(1) for c in fp.readlines()]))
casedirs = os.listdir(os.path.join(datadir,'training'))
casenumbers = list(map(int,[re.search('([0-9]{5})',c).group(1) for c in casedirs]))
traincases = [c for c in casenumbers if c not in testcases]
traincasedirs = [d for c,d in zip(casenumbers,casedirs) if c in traincases]
cases_train,cases_val,casedirs_train,casedirs_val = train_test_split(traincases,traincasedirs,test_size=0.1,random_state=4)
cases_train = sorted(cases_train)
cases_val = sorted(cases_val)
casedirs_train = sorted(casedirs_train)
casedirs_val = sorted(casedirs_val)

tv_path = ['training_slice','validation_slice']
tv_set = {'training_slice':{'casedirs':casedirs_train,'cases':cases_train},
          'validation_slice':{'casedirs':casedirs_val,'cases':cases_val}}

for tv in tv_set.keys():

    try:
        shutil.rmtree(os.path.join(datadir,tv))
    except FileNotFoundError:
        pass
    os.makedirs(os.path.join(datadir,tv,'images'),exist_ok=True)
    os.makedirs(os.path.join(datadir,tv,'labels'),exist_ok=True)

    for idx,C in enumerate(tv_set[tv]['casedirs']):
        if False: # debugging
            if idx != 52:
                continue
        case = tv_set[tv]['cases'][idx]
        cpath = os.path.join(datadir,'training',C)
        opath = os.path.join(datadir,tv)
        img_arr_t1c,img_arr_seg = load_dataset(cpath)

        nslice = np.shape(img_arr_seg)[0]
        nslice_seg = np.zeros(nslice,dtype='int')
        for i in range(nslice):
            npixels = len(np.where((img_arr_seg[i]==3)|(img_arr_seg[i]==1))[0])
            if npixels > 9:
                nslice_seg[i] = npixels
        
        c_index = np.argsort(nslice_seg)[-1::-1]
        n_index = len(np.where(nslice_seg[c_index]>0)[0])
        # pick one large and one small lesion
        s1 = int(n_index/8)
        s2 = int(n_index/2)
        c_slices = c_index[s1],c_index[s2]
        idx_ct = len(c_slices) * idx
        for idx_plus,c in enumerate(c_slices):
            ofile = 'img_' + str(idx_ct+idx_plus).zfill(4) + '_case_' + str(case).zfill(3) + '_slice_' + str(c).zfill(3) + '.png'
            # new_p = PIL.Image.fromarray(img_arr_t1c[c])
            # new_p.convert("L")
            # new_p.save(os.path.join(opath,ofile))
            # skimage save can't work unless mode F (32bit float) is converted to uint8
            img = (img_arr_t1c[c] / np.max(img_arr_t1c[c]) * 255).astype('uint8')
            imsave(os.path.join(opath,'images',ofile),img)
            ofile = 'img_' + str(idx_ct+idx_plus).zfill(4) + '_case_' + str(case).zfill(3) + '_slice_' + str(c).zfill(3) + '.png' 
            mask = ((img_arr_seg[c] == 3) | (img_arr_seg[c] == 1)).astype('uint8')*255
            CC_mask = cc3d.connected_components(mask,connectivity=4)
            nCC_mask = len(np.unique(CC_mask))
            # select largest component. occasionally, the mask of a single large and 
            # complicated lesion will have a few disconnected pixels in a particular slice,
            # and the general size has been specified by s1,s2, so don't need to make
            # a random choice here
            if nCC_mask > 2:
                # sel = random.randint(1,nCC_mask-1)
                sel = 1
            else:
                sel = 1
            mask = (CC_mask == sel).astype('uint8') * 255
            imsave(os.path.join(opath,'labels',ofile),mask,check_contrast=False)
