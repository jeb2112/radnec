# methods for pre-processing dicom input dirs

import os,sys
import numpy as np
import glob
import copy
import re
import logging
import subprocess
import tkinter as tk
import nibabel as nb
from nibabel.processing import resample_from_to
import pydicom as pd
from pydicom.fileset import FileSet
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from cProfile import Profile
from pstats import SortKey,Stats
from enum import Enum
import ants

from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice


class PreProcess():
    def __init__():
        return

    # load up to four dicom series directories in the provided parent directory
    def loaddicom(self,d):
        dset = {'t1pre':{'d':None,'ex':False},'t1':{'d':None,'ex':False},'t2':{'d':None,'ex':False},
                'flair':{'d':None,'ex':False},'ref':{'d':None,'mask':None,'ex':False}}
        # assume a dicomdir is a parent of several series directories
        self.casedir = d
        seriesdirs = os.listdir(d)
        # special case subdir for providing an externally generated mask
        if 'mask' in seriesdirs:
            seriesdirs.pop(seriesdirs.index('mask'))
            dpath = os.path.join(d,'mask')
            img_nb = nb.load(os.path.join(dpath,'img_mask.nii.gz'))
            dset['ref']['mask'] = np.transpose(np.array(img_nb.dataobj),axes=(2,1,0))
            img_nb = nb.load(os.path.join(dpath,'img_reference.nii.gz'))
            dset['ref']['d'] = np.transpose(np.array(img_nb.dataobj),axes=(2,1,0))
        for sd in seriesdirs:
            dpath = os.path.join(d,sd)
            files = sorted(os.listdir(dpath))
            ds0 = pd.dcmread(os.path.join(dpath,files[0]))
            slice0 = ds0[(0x0020,0x0032)].value[2]
            ds = pd.dcmread(os.path.join(dpath,files[-1]))
            dslice = ds[(0x0020,0x0032)].value[2] - slice0
            if dslice < 0:
                files = sorted(files,reverse=True)
            print(ds0.SeriesDescription)
            if 't1' in ds0.SeriesDescription.lower():
                if 'pre' in ds0.SeriesDescription.lower():
                    dset['t1pre']['ex'] = True
                    dset['t1pre']['d'] = np.zeros((len(files),ds0.Rows,ds0.Columns))
                    self.ui.affine['t1pre'] = self.get_affine(ds0,dslice)
                    dset['t1pre']['d'][0,:,:] = ds0.pixel_array
                    for i,f in enumerate(files[1:]):
                        data = pd.dcmread(os.path.join(dpath,f))
                        dset['t1pre']['d'][i+1,:,:] = data.pixel_array
                else:
                    dset['t1']['ex'] = True
                    dset['t1']['d'] = np.zeros((len(files),ds0.Rows,ds0.Columns))
                    self.ui.affine['t1'] = self.get_affine(ds0,dslice)
                    dset['t1']['d'][0,:,:] = ds0.pixel_array
                    # also create isotropic target affine for resampling based on the t1 affine
                    nslice = int( ds0[(0x2001,0x1018)].value * float(ds0[(0x0018,0x0088)].value) )
                    if pd.tag.Tag(0x2001,0x1018) in ds0.keys() and pd.tag.Tag(0x0018,0x0088) in ds0.keys(): # philips. siemens?
                        nslice = int( ds0[(0x2001,0x1018)].value * float(ds0[(0x0018,0x0088)].value) )
                    elif pd.tag.Tag(0x0018,0x0050) in ds0.keys(): # possible alternate for siemens?
                        nslice = int(float(ds0[(0x0018,0x0050)].value) * len(files))
                    else:
                        raise Exception('number of 1mm slices cannot be parsed from header')
                    nx = int( float(ds0[(0x0028,0x0030)].value[0]) * ds0[(0x0028,0x0010)].value )
                    ny = int( float(ds0[(0x0028,0x0030)].value[1]) * ds0[(0x0028,0x0011)].value )
                    affine =  np.diag(np.ones(4),k=0)
                    affine[:3,3] = self.ui.affine['t1'][:3,3]
                    self.ui.affine['target'] = affine
                    dset['target'] = np.zeros((nslice,ny,nx))
                    for i,f in enumerate(files[1:]):
                        data = pd.dcmread(os.path.join(dpath,f))
                        dset['t1']['d'][i+1,:,:] = data.pixel_array
            elif any([f in ds0.SeriesDescription.lower() for f in ['flair','fluid']]):
                dset['flair']['ex'] = True
                dset['flair']['d'] = np.zeros((len(files),ds0.Rows,ds0.Columns))
                self.ui.affine['flair'] = self.get_affine(ds0,dslice)
                dset['flair']['d'][0,:,:] = ds0.pixel_array
                for i,f in enumerate(files[1:]):
                    data = pd.dcmread(os.path.join(dpath,f))
                    dset['flair']['d'][i+1,:,:] = data.pixel_array
            elif 't2' in ds0.SeriesDescription.lower():
                dset['t2']['ex'] = True
                dset['t2']['d'] = np.zeros((len(files),ds0.Rows,ds0.Columns))
                self.ui.affine['t2'] = self.get_affine(ds0,dslice)
                for i,f in enumerate(files[1:]):
                    data = pd.dcmread(os.path.join(dpath,f))
                    dset['t2']['d'][i+1,:,:] = data.pixel_array
        return dset
    
    # use for an input dicom directory
    # preprocessing nifti files is handled by the checkbox settings
    def preprocess(self,dirs):
        for d in dirs:

            dset = self.loaddicom(d)

            # resample to target isotropic matrix
            for t in ['t1pre','t2']:
                if dset[t]['ex']:
                    print('Resampling ' + t + ' into target space...')
                    dset[t]['d'],self.ui.affine[t] = self.resamplet2(dset['target'],dset[t]['d'],
                                                                     self.ui.affine['target'],self.ui.affine[t])
                    dset[t]['d']= np.clip(dset[t]['d'],0,None)

            # assuming t1,flair always present in any possible dataset
            for t in ['t1','flair']:
                print('Resampling ' + t + ' into target space...')
                dset[t]['d'],self.ui.affine[t] = self.resamplet2(dset['target'],dset[t]['d'],
                                                                 self.ui.affine['target'],self.ui.affine[t])
                dset[t]['d'] = np.clip(dset[t]['d'],0,None)

            if False:
                for t in ['t1pre','t1','t2','flair']:
                    if dset[t]['ex']:
                        self.ui.roiframe.WriteImage(dset[t]['d'],os.path.join(d,'img_'+t+'_resampled.nii.gz'),
                                            type='float',affine=self.ui.affine['t1'])
                        

            # skull strip. for now assuming only needed on input dicoms

            if dset['ref']['mask'] is not None:
                # use optional mask if available. 

                # pre-registration. for now assuming mask is a T1post so register T1pre,T2,flair
                # reference
                fixed_image = itk.GetImageFromArray(dset['t1']['d'])

                for t in ['t1pre','t2','flair']:
                    fname = os.path.join(d,'img_'+t+'_resampled.nii.gz')
                    if dset[t]['ex']:
                        moving_image = itk.GetImageFromArray(dset[t]['d'])
                        moving_image_res = self.elastix_affine(fixed_image,moving_image,type='rigid')
                        dset[t]['d'] = itk.GetArrayFromImage(moving_image_res)
                        if False:
                            self.ui.roiframe.WriteImage(dset[t]['d'],os.path.join(d,'img_'+t+'_preregistered.nii.gz'),
                                                    type='float',affine=self.ui.affine['t1'])

                # apply mask
                for t in ['t1','t1pre','t2','flair']:
                    if dset[t]['ex']:
                        dset[t]['d'] = np.where(dset['ref']['mask'],dset[t]['d'],0)

            else:    
                # attempt model extraction
                img_arr_t1_ex,mask = self.extractbrain(dset['t1']['d'])
                if True:
                    self.ui.roiframe.WriteImage(mask,os.path.join(d,'img_'+'t1'+'_mask.nii.gz'),
                                                type='uint8',affine=self.ui.affine['t1'])
                nmask = len(mask[mask>0])
                for t in ['t1pre','t2','flair']:
                    if dset[t]['ex']:
                        img_arr_t_ex,_ = self.extractbrain(dset[t]['d'])
                        # if extraction mask is significantly smaller than t1 mask, assume it's not good
                        # and just use the t1 mask
                        if len(img_arr_t_ex[img_arr_t_ex>0]) > 0.98*nmask:
                            dset[t]['d'] = copy.deepcopy(img_arr_t_ex)
                        else:
                            # to use t1 mask have to preregister to t1 first
                            fixed_image = itk.GetImageFromArray(dset['t1']['d'])

                            fname = os.path.join(d,'img_'+t+'_resampled.nii.gz')
                            if dset[t]['ex']:
                                moving_image = itk.GetImageFromArray(dset[t]['d'])
                                moving_image_res = self.elastix_affine(fixed_image,moving_image,type='rigid')
                                dset[t]['d'] = itk.GetArrayFromImage(moving_image_res)
                                if False:
                                    self.ui.roiframe.WriteImage(dset[t]['d'],os.path.join(d,'img_'+t+'_preregistered.nii.gz'),
                                                            type='float',affine=self.ui.affine['t1'])
                            # apply t1 mask
                            dset[t]['d'] = np.where(mask,dset[t]['d'],0)
                # finally record extracted t1
                dset['t1']['d'] = img_arr_t1_ex

            if False:
                for t in ['t1','t1pre','t2','flair']:
                    if dset[t]['ex']:
                        self.ui.roiframe.WriteImage(dset[t]['d'],os.path.join(d,'img_'+t+'_extracted.nii.gz'),
                                                    type='float',affine=self.ui.affine['t1'])

            # registration
            print('register T2, flair')

            # reference
            fixed_image = itk.GetImageFromArray(dset['t1']['d'])

            for t in ['t1pre','t2','flair']:
                fname = os.path.join(d,'img_'+t+'_resampled.nii.gz')
                if dset[t]['ex']:
                    moving_image = itk.GetImageFromArray(dset[t]['d'])
                    moving_image_res = self.elastix_affine(fixed_image,moving_image,type='rigid')
                    dset[t]['d'] = itk.GetArrayFromImage(moving_image_res)
                    if False:
                        self.ui.roiframe.WriteImage(dset[t]['d'],os.path.join(d,'img_'+t+'_registered.nii.gz'),
                                                type='float',affine=self.ui.affine['t1'])

            # bias correction.
            for t in ['t1pre','t1','t2','flair']:
                if dset[t]['ex']:   
                    dset[t]['d'] = self.n4bias(dset[t]['d'])

            # rescale the data
            # if necessary clip any negative values introduced by the processing
            for t in ['t1pre','t1','t2','flair']:
                if dset[t]['ex']:
                    if np.min(dset[t]['d']) < 0:
                        dset[t]['d'][dset[t]['d'] < 0] = 0
                    dset[t]['d'] = self.rescale(dset[t]['d'])

            # save nifti files for future use
            for t in ['t1pre','t1','t2','flair']:
                if dset[t]['ex']:
                    self.ui.roiframe.WriteImage(dset[t]['d'],os.path.join(d,'img_'+t+'_processed.nii.gz'),
                                                type='float',affine=self.ui.affine['t1'])

        if len(dirs) == 1:
            return dset
        return



    def normalslice_callback(self,event=None):
        print('normal stats')
        # do kmeans
        # Creates a matrix of voxels for normal brain slice
        # Gating Routine

        if self.slicevolume_norm.get() == 0:
            self.normalslice=self.ui.get_currentslice()
            region_of_support = np.where(self.ui.data['raw'][0,self.normalslice]*self.ui.data['raw'][1,self.normalslice]>0) 
            vset = np.zeros_like(region_of_support,dtype='float')
            for i in range(3):
                vset[i] = np.ravel(self.ui.data['raw'][i,self.normalslice][region_of_support])
        else:
            self.normalslice = None
            region_of_support = np.where(self.ui.data['raw'][0]*self.ui.data['raw'][1]*self.ui.data['raw'][2] >0)
            vset = np.zeros_like(region_of_support,dtype='float')
            for i in range(3):
                vset[i] = np.ravel(self.ui.data['raw'][i][region_of_support])
            # t1channel_normal = self.ui.data['raw'][0][region_of_support]
            # flairchannel_normal = self.ui.data['raw'][1][region_of_support]
            # t2channel_normal = self.ui.data['raw'][2][region_of_support]

        # kmeans to calculate statistics for brain voxels
        # X_et = np.column_stack((flair,t1))
        # X_net = np.column_stack((flair,t2))
        X={}
        X['ET'] = np.column_stack((vset[1],vset[0]))
        X['T2 hyper'] = np.column_stack((vset[1],vset[2]))

        for i,layer in enumerate(['ET','T2 hyper']):
            np.random.seed(1)
            kmeans = KMeans(n_clusters=2,n_init='auto').fit(X[layer])
            background_cluster = np.argmin(np.power(kmeans.cluster_centers_[:,0],2)+np.power(kmeans.cluster_centers_[:,1],2))

            # Calculate stats for brain cluster
            self.ui.data['blast']['params'][layer]['stdt12'] = np.std(X[layer][kmeans.labels_==background_cluster,1])
            self.ui.data['blast']['params'][layer]['stdflair'] = np.std(X[layer][kmeans.labels_==background_cluster,0])
            self.ui.data['blast']['params'][layer]['meant12'] = np.mean(X[layer][kmeans.labels_==background_cluster,1])
            self.ui.data['blast']['params'][layer]['meanflair'] = np.mean(X[layer][kmeans.labels_==background_cluster,0])

            if False:
                plt.figure(7)
                ax = plt.subplot(1,2,i+1)
                plt.scatter(X[layer][kmeans.labels_==1-background_cluster,0],X[layer][kmeans.labels_==1-background_cluster,1],c='b',s=1)
                plt.scatter(X[layer][kmeans.labels_==background_cluster,0],X[layer][kmeans.labels_==background_cluster,1],c='r',s=1)
                ax.set_aspect('equal')
                ax.set_xlim(left=0,right=1.0)
                ax.set_ylim(bottom=0,top=1.0)
                plt.text(0,1.02,'{:.3f},{:.3f}'.format(self.ui.data['blast']['params'][layer]['meanflair'],self.ui.data['blast']['params'][layer]['stdflair']))

                plt.savefig('/home/jbishop/Pictures/scatterplot_normal.png')
                plt.clf()
                # plt.show(block=False)

        # automatically run BLAST
            self.ui.roiframe.layer_callback(layer=layer,updateslice=False,overlay=False)
            self.ui.runblast(currentslice=None,layer=layer)

            # activate thresholds only after normal slice stats are available
            for s in ['t12','flair','bc']:
                self.ui.roiframe.sliders[layer][s]['state']='normal'
                self.ui.roiframe.sliders[layer][s].bind("<ButtonRelease-1>",Command(self.ui.roiframe.updateslider,layer,s))
        # since we finish the on the T2 hyper layer, have this slider disabled to begin with
        # self.ui.roiframe.sliders['ET']['t12']['state']='disabled'


