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
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from cProfile import Profile
from pstats import SortKey,Stats
from enum import Enum
import ants

from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice


class PreProcess():
    def __init__(self):
        self.dtag = ['t0','t1r'] 
        self.dset = {'t0':{'d':None,'d2':None,'time':None,'affine':None},'t1r':{'d':None,'d2':None,'time':None,'affine':None}}
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
    

    # load single dicom series directory 
    # from temporal regression
    def loaddicom2(self,dpath):
        # list of time attributes to check
        timeattrs = {'AcquisitionDate':None,'AcquisitionTime':None,'StudyDate':None,'StudyTime':None}

        files = sorted(os.listdir(dpath))
        ds0 = pd.dcmread(os.path.join(dpath,files[0]))
        try:
            slice0 = ds0[(0x0020,0x0032)].value[2]
            ds = pd.dcmread(os.path.join(dpath,files[-1]))
            dslice = ds[(0x0020,0x0032)].value[2] - slice0
            if dslice < 0:
                files = sorted(files,reverse=True)
        except KeyError:
            pass
        print(dpath)
        print(ds0.SeriesDescription)
        for t in timeattrs.keys():
            if hasattr(ds0,t):
                timeattrs[t] = getattr(ds0,t)
        dset = np.zeros((len(files),ds0.Rows,ds0.Columns))
        affine = self.get_affine(ds0)
        dset[0,:,:] = ds0.pixel_array
        for ii,f in enumerate(files[1:]):
            data = pd.dcmread(os.path.join(dpath,f))
            dset[ii+1,:,:] = data.pixel_array
        return dset,affine,timeattrs,ds0



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


    # newer from temporal regression
    def process_dicom(self,dcmdirs,case,rc):

        dtag = ['t0','t1r'] 
        dset = {'t0':{'d':None,'d2':None,'time':None,'affine':None},'t1r':{'d':None,'d2':None,'time':None,'affine':None}}

        for i,d in enumerate(dcmdirs):
            print('dicom dir {}'.format(d))
            dpath = os.path.split(d)[0]
            npath = os.path.join(d.split(case)[0],case)
            if not os.path.exists(os.path.join(npath,'t0')):
                os.mkdir(os.path.join(npath,'t0'))
                os.mkdir(os.path.join(npath,'t1r'))
            dset[dtag[i]]['d'],dset[dtag[i]]['affine'],dset[dtag[i]]['time'],metadata = self.loaddicom2(d,dcmtype=rc[case]['dcmtype'])
            # convert to hounfield. not verified yet.
            if rc[case]['dcmtype'] == 'CT':
                dset[dtag[i]]['d'] = dset[dtag[i]]['d']*metadata.RescaleSlope + metadata.RescaleIntercept

        # check temporal order. partially implemented
        if True:
            # different date time checks. add more as required
            # Acquisition date appears to be per Series
            if all(dset[v]['time']['AcquisitionDate'] is not None for v in ['t0','t1r']):
                if dset['t0']['time']['AcquisitionDate'] > dset['t1r']['time']['AcquisitionDate']:
                    dset['t1r'],dset['t0'] = dset['t0'],dset['t1r']
                elif dset['t0']['time']['AcquisitionDate'] == dset['t1r']['time']['AcquisitionDate']:
                    if dset['t0']['time']['AcquisitionTime'] > dset['t1r']['time']['AcquisitionTime']:
                        dset['t1r'],dset['t0'] = dset['t0'],dset['t1r']
            # for a study that crosses midnight, the study date could remain the same while acquisition dates change
            elif all(dset[v]['time']['StudyDate'] is not None for v in ['t0','t1r']):
                if dset['t0']['time']['StudyDate'] > dset['t1r']['time']['StudyDate']:
                    dset['t1r'],dset['t0'] = dset['t0'],dset['t1r']
                elif dset['t0']['time']['StudyDate'] == dset['t1r']['time']['StudyDate']:
                    if dset['t0']['time']['AcquisitionTime'] > dset['t1r']['time']['AcquisitionTime']:
                        dset['t1r'],dset['t0'] = dset['t0'],dset['t1r']
            else:
                print('T0,T1 times not detected')


        # pre-registration. 
        if rc[case]['dcmtype'] == 'MR':
            dset['t1r']['d'] = self.register(dset['t0']['d'],dset['t1r']['d'],type='ants')
            for dt in dtag:
                niftifile = os.path.join(npath,dt,dt+'.nii')
                self.writenifti(dset[dt]['d'],niftifile,type=dset[dt]['d'].dtype.name,affine=dset['t0']['affine'])

        # skull extraction
        if rc[case]['dcmtype'] == 'MR':
            for dt in dtag:
                command = 'conda run -n hdbet hd-bet '
                command += ' -i ' + os.path.join(npath,dt,dt+'.nii')
                res = os.system(command)
                dset[dt]['d'],_ = self.loadnifti(dt+'_bet.nii.gz',os.path.join(npath,dt))


        # post-registration
        if rc[case]['dcmtype'] == 'MR':
            dset['t1r']['d'] = self.register(dset['t0']['d'],dset['t1r']['d'],type='ants')

        # N4 bias correction
        if rc[case]['dcmtype'] == 'MR':
            for dt in dtag:     
                dset[dt]['d'] = self.n4bias(dset[dt]['d'])
                niftifile = os.path.join(npath,dt,dt+'_bet_n4.nii')
                self.writenifti(dset[dt]['d'],niftifile,type=dset[dt]['d'].dtype.name,affine=dset['t0']['affine'])        

        # remove skull files
        if False:
            for dt in dtag:
                os.remove(os.path.join(npath,dt,dt+'.nii'))

        return dset



    def normalstats_callback(self,event=None):
        print('normal stats')
        # do kmeans
        # Creates a matrix of voxels for normal brain slice
        # Gating Routine

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


    # create nb affine from dicom 
    def get_affine(self,dicomdata,dslice):
        dircos = np.array(list(map(float,dicomdata.ImageOrientationPatient)))
        affine = np.zeros((4,4))
        affine[:3,0] = dircos[0:3]*float(dicomdata.PixelSpacing[0])
        affine[:3,1] = dircos[3:]*float(dicomdata.PixelSpacing[1])
        d3 = np.cross(dircos[:3],dircos[3:])
        # not entirely sure if these three tags all work the same across the 3 vendors
        if dicomdata[(0x0018,0x0023)].value == '3D':
            if hasattr(dicomdata,'SpacingBetweenSlices'):
                slthick = float(dicomdata.SpacingBetweenSlices)
            elif hasattr(dicomdata,'SliceThickness'):
                slthick = float(dicomdata.SliceThickness)
            else:
                raise ValueError('Slice thickness not parsed')
        else:
            if hasattr(dicomdata,'SliceThickness'):
                slthick = float(dicomdata.SliceThickness)
            else:
                raise ValueError('Slice thickness not parsed')

        affine[:3,2] = d3*slthick
        affine[:3,3] = dicomdata.ImagePositionPatient
        affine[3,3] = 1
        # print(affine)
        return affine
    

    # replace with hd-bet
    def extractbrain(self,img_arr_input):
        print('extract brain')
        img_arr = copy.deepcopy(img_arr_input)
        img_arr = self.brainmage_clip(img_arr)
        self.ui.roiframe.WriteImage(img_arr,os.path.join(self.casedir,'img_temp.nii'),norm=False,type='float')

        tfile = 'img_temp.nii'
        ofile = 'img_brain.nii'
        mfile = 'img_mask.nii'
        if self.ui.OS == 'linux':
            command = 'conda run -n brainmage brain_mage_single_run '
            command += ' -i ' + os.path.join(self.casedir,tfile)
            command += ' -o ' + os.path.join(self.casedir,mfile)
            command += ' -m ' + os.path.join(self.casedir,ofile) + ' -dev 0'
            res = os.system(command)

        elif self.ui.OS == 'win32':
            # manually escaped for shell. can also use raw string as in r"{}".format(). or subprocess.list2cmdline()
            # some problem with windows, the scrip doesn't get on PATH after env activation, so still have to specify the fullpath here
            # it is currently hard-coded to anaconda3/envs location rather than .conda/envs, but anaconda3 could be installed
            # under either ProgramFiles or Users so check both
            if os.path.isfile(os.path.expanduser('~')+'\\anaconda3\Scripts\\activate.bat'):
                activatebatch = os.path.expanduser('~')+"\\anaconda3\Scripts\\activate.bat"
            elif os.path.isfile("C:\Program Files\\anaconda3\Scripts\\activate.bat"):
                activatebatch = "C:\Program Files\\anaconda3\Scripts\\activate.bat"
            else:
                raise FileNotFoundError('anaconda3/Scripts/activate.bat')
                                
            command1 = '\"'+activatebatch+'\" \"' + os.path.expanduser('~')+'\\anaconda3\envs\\brainmage\"'
            command2 = 'python \"' + os.path.join(os.path.expanduser('~'),'anaconda3','envs','brainmage','Scripts','brain_mage_single_run')
            command2 += '\" -i   \"' + os.path.join(self.casedir,tfile)
            command2 += '\"  -o  \"' + os.path.join(os.path.expanduser('~'),'AppData','Local','Temp','foo')
            command2 += '\"   -m   \"' + os.path.join(self.casedir,ofile) + '\"'
            cstr = 'cmd /c \" ' + command1 + "&" + command2 + '\"'
            popen = subprocess.Popen(cstr,shell=True,stdout=subprocess.PIPE,universal_newlines=True)
            for stdout_line in iter(popen.stdout.readline,""):
                if stdout_line != '\n':
                    print(stdout_line)
            popen.stdout.close()
            res = popen.wait()
            if res:
                raise subprocess.CalledProcessError(res,cstr)
                print(res)

        img_nb = nb.load(os.path.join(self.casedir,'img_brain.nii'))
        img_arr = np.transpose(np.array(img_nb.dataobj),axes=(2,1,0))
        img_nb = nb.load(os.path.join(self.casedir,'img_mask.nii'))
        img_arr_mask = np.transpose(np.array(img_nb.dataobj),axes=(2,1,0))
        os.remove(os.path.join(self.casedir,'img_brain.nii'))
        os.remove(os.path.join(self.casedir,'img_temp.nii'))
        os.remove(os.path.join(self.casedir,'img_mask.nii'))
        return img_arr,img_arr_mask

                
    # if T2 matrix is different resample it to t1
    def resamplet2(self,arr_t1,arr_t2,a1,a2):
        img_arr_t1 = copy.deepcopy(arr_t1)
        img_arr_t2 = copy.deepcopy(arr_t2)
        img_t1 = nb.Nifti1Image(np.transpose(img_arr_t1,axes=(2,1,0)),affine=a1)
        img_t2 = nb.Nifti1Image(np.transpose(img_arr_t2,axes=(2,1,0)),affine=a2)
        img_t2_res = resample_from_to(img_t2,(img_t1.shape[:3],img_t1.affine))
        img_arr_t2 = np.ascontiguousarray(np.transpose(np.array(img_t2_res.dataobj),axes=(2,1,0)))
        return img_arr_t2,img_t2_res.affine
 
    def n4bias(self,img_arr,shrinkFactor=4):
        print('N4 bias correction')
        data = copy.deepcopy(img_arr)
        dataImage = ants.from_numpy(img_arr)
        # ant mask must be float. 
        mask = np.zeros_like(data,dtype=float)
        mask[np.where(data > 0)] = 1
        maskImage = ants.from_numpy(mask)
        dataImage_n4 = ants.n4_bias_field_correction(dataImage,mask=maskImage,shrink_factor=shrinkFactor)
        img_arr_n4 = dataImage_n4.numpy()
        return img_arr_n4


    def loadnifti(t1_file,casedir,type=None):
        img_arr_t1 = None
        try:
            img_nb_t1 = nb.load(os.path.join(casedir,t1_file))
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

    # use uint8 for masks 
    def writenifti(img_arr,filename,header=None,norm=False,type='float64',affine=None):
        img_arr_cp = copy.deepcopy(img_arr)
        if norm:
            img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
        # using nibabel nifti coordinates
        img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
        nb.save(img_nb,filename)
        if True:
            os.system('gzip --force {}'.format(filename))


    # register-only, for temporal parameter space
    def register(img_arr_fixed,img_arr_moving):
        print('register T1,T0')

        fixed_ants = ants.from_numpy(img_arr_fixed)
        moving_ants = ants.from_numpy(img_arr_moving)
        mytx = ants.registration(fixed=fixed_ants, moving=moving_ants, type_of_transform = 'Affine' )
        img_arr_reg = mytx['warpedmovout'].numpy()
        a=1

        return img_arr_reg
