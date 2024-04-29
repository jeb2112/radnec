# methods for pre-processing dicom input dirs

import os,sys
import numpy as np
import glob
import copy
import re
import logging
import subprocess
import pickle
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

# a collection of multiple studies
class Case():
    def __init__(self,casename,studydirs,config):

        self.config = config
        self.case = casename
        self.studydirs = studydirs
        self.studies = []


        self.load_studydirs()
        self.process_studydirs()
        self.process_timepoints()



    # load all studies of current case
    def load_studydirs(self):

        for i,sd in enumerate(self.studydirs):
            self.studies.append(Study(self.case,sd))
            self.studies[i].loaddicom()
        # sort studies by time and number of series
        self.studies = sorted(self.studies,key=lambda x:(x.studytimeattrs['StudyDate'],
                                                         len([t for t in x.dset.keys() if not x.dset[t]['ex']])))
        # combine studies with common date, for now any series in the study with fewer series, is 
        # copied to the study with more series, over-writes are not being checked yet but should be well-behaved
        # TODO: actually check the times, so a later series only overwrites an earlier series according to temporal
        # relation, ie possible rescan
        dates = sorted(list(set([d.studytimeattrs['StudyDate'] for d in self.studies])))
        studies = []
        for i,d in enumerate(dates):
            dstudies = [s for s in self.studies if s.studytimeattrs['StudyDate'] == d]
            for ds in dstudies[-1:0:-1]:
                for series in ds.dset.keys():
                    if ds.dset[series]['ex']:
                        dstudies[0].dset[series]['d'] = np.copy(ds.dset[series]['d'])
                        dstudies[0].dset[series]['affine'] = np.copy(ds.dset[series]['affine'])
                        dstudies[0].dset[series]['time'] = ds.dset[series]['time']
                        dstudies[0].dset[series]['ex'] = True
                dstudies.remove(ds)
            studies.append(dstudies[0])
        self.studies = studies
        self.p = DcmProcess(self.case,self.studies,self.config)

        return

    # resample,register,bias correction
    def process_studydirs(self):
        self.p.process_studydirs()

        return

    # register time0 to talairach, then register
    # all subsequent time points to time0
    def process_timepoints(self):
        self.p.postprocess()

        return




class Study():

    def __init__(self,case,d):
        self.studydir = d
        self.case = case
        self.date = None
        self.casedir = None
        self.dset = {'t1':{'d':None,'time':None,'affine':None,'ex':False},
                     't1+':{'d':None,'time':None,'affine':None,'ex':False,'mask':None},
                     'zt1+':{'d':None,'time':None,'affine':None,'ex':False},
                     'flair':{'d':None,'time':None,'affine':None,'ex':False},
                     'flair+':{'d':None,'time':None,'affine':None,'ex':False},
                     'zflair+':{'d':None,'time':None,'affine':None,'ex':False},
                     'cbv':{'d':None,'time':None,'affine':None,'ex':False},
                     'target':{'d':None,'affine':None,'ex':False},
                     'ET':{'d':None,'affine':None,'ex':False},
                     'WT':{'d':None,'affine':None,'ex':False}
                     }
        self.dtag = [k for k in self.dset.keys()]
        self.date = None
        return
    
    def loadnifti(self,t1_file,dir=None,type=None):
        img_arr_t1 = None
        if dir is None:
            dir = self.studydir
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


    # use uint8 for masks 
    def writenifti(self,img_arr,filename,header=None,norm=False,type='float64',affine=None):
        img_arr_cp = copy.deepcopy(img_arr)
        if norm:
            img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
        # using nibabel nifti coordinates
        img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
        nb.save(img_nb,filename)
        if True:
            os.system('gzip --force {}'.format(filename))

    def write_all(self):
        # save nifti files for future use
        for s in self.studies:
            localcasedir = os.path.join(self.config.UIlocaldir,self.case,s.studytimeattrs['StudyDate'])
            for dt in s.dtag:
                if s.dset[dt]['ex']:
                    self.writenifti(s.dset[dt]['d'],os.path.join(localcasedir,dt+'_processed.nii'),
                                                type='float',affine=self.affine)


class NiftiStudy(Study):

    def __init__(self,case,d):
        super().__init__(case,d)

    def loaddata(self):
        files = os.listdir(self.studydir)
        for dt in self.dtag:
            dt_files = [f for f in files if dt in f.lower()]
            if len(dt_files) > 0:
                if len(dt_files) > 1:
                    # by convention '_processed' is the final output from dcm preprocess()
                    dt_file = next((f for f in dt_files if re.search('_processed',f.lower())),None)
                elif len(dt_files) == 1:
                    dt_file = dt_files[0]

                self.dset[dt]['d'],self.dset[dt]['affine'] = self.loadnifti(dt_file)
                self.dset[dt]['ex'] = True
        return



class DcmStudy(Study):

    def __init__(self,case,d):
        super()._init__(case,d)
        self.localcasedir = None
        # list of time attributes to check
        self.seriestimeattrs = ['AcquisitionTime']
        self.studytimeattrs = {'StudyDate':None,'StudyTime':None}
        self.date = None
        return
    
    # load up multiple series directories in the provided study directory
    def loaddicom(self):
        d = self.studydir
        seriesdirs = os.listdir(d)

        # special case subdir for providing an externally generated mask
        # probably not needed anymore
        if False:
            if 'mask' in seriesdirs:
                seriesdirs.pop(seriesdirs.index('mask'))
                dpath = os.path.join(d,'mask')
                img_nb = nb.load(os.path.join(dpath,'img_mask.nii.gz'))
                self.dset['ref']['mask'] = np.transpose(np.array(img_nb.dataobj),axes=(2,1,0))
                img_nb = nb.load(os.path.join(dpath,'img_reference.nii.gz'))
                self.dset['ref']['d'] = np.transpose(np.array(img_nb.dataobj),axes=(2,1,0))

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
            for t in self.studytimeattrs.keys():
                if hasattr(ds0,t):
                    self.studytimeattrs[t] = getattr(ds0,t)

            if 't1' in ds0.SeriesDescription.lower():
                # so far 'pre' is sufficient for t1
                if 'pre' in ds0.SeriesDescription.lower():
                    dt = 't1'
                else:
                    dt = 't1+'

            # if flair scans aren't designated pre/post, assume post
            # this may change if a pre-contrast flair is added to protocol
            elif any([f in ds0.SeriesDescription.lower() for f in ['flair','fluid']]):
                if 'pre' in ds0.SeriesDescription.lower():
                    dt = 'flair'
                else:
                    dt = 'flair+'

            # not taking relcbv or relcbf, just relccbv
            # note this may be exported in a separate studydir, without a matching t1
            # TODO: the matching t1 has to come from another studydir, based on time tags
            elif any([f in ds0.SeriesDescription.lower() for f in ['relccbv','perf']]):
                dt = 'cbv'
            else:
                dt = None
                continue

            if dt is not None:
                self.dset[dt]['ex'] = True
                self.dset[dt]['d'] = np.zeros((len(files),ds0.Rows,ds0.Columns))
                self.dset[dt]['affine'] = self.get_affine(ds0,dslice)
                self.dset[dt]['d'][0,:,:] = ds0.pixel_array
                for i,f in enumerate(files[1:]):
                    data = pd.dcmread(os.path.join(dpath,f))
                    self.dset[dt]['d'][i+1,:,:] = data.pixel_array

                for t in self.seriestimeattrs:
                    if hasattr(ds0,t):
                        self.dset[dt]['time'] = getattr(ds0,t)
                        break

            # also create target affine with 1mm slices (not finished) 
            # t1+ will probably always have 1mm slices anyway, might not need anymore
            if False:
                if dt == 't1+':
                    if pd.tag.Tag(0x2001,0x1018) in ds0.keys() and pd.tag.Tag(0x0018,0x0088) in ds0.keys(): # philips. siemens?
                        nslice = int( ds0[(0x2001,0x1018)].value * float(ds0[(0x0018,0x0088)].value) )
                    elif pd.tag.Tag(0x0018,0x0050) in ds0.keys(): # possible alternate for siemens?
                        nslice = int(float(ds0[(0x0018,0x0050)].value) * len(files))
                    else:
                        raise Exception('number of 1mm slices cannot be parsed from header')
                    nx = int( float(ds0[(0x0028,0x0030)].value[0]) * ds0[(0x0028,0x0010)].value )
                    ny = int( float(ds0[(0x0028,0x0030)].value[1]) * ds0[(0x0028,0x0011)].value )
                    affine =  np.diag(np.ones(4),k=0)
                    # affine[:3,3] = self.dset['t1+']['affine'][:3,3]
                    # self.dset['target']['affine'] = affine
                    # self.dset['target']['d'] = np.zeros((nslice,ny,nx))

        return
    


    # load single dicom series directory 
    # from temporal regression
    # not sure if needed
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
    
    # from temporal regression
    def get_time(self):
        if all(self.dset[v]['time']['AcquisitionDate'] is not None for v in ['t0','t1r']):
            if self.dset['t0']['time']['AcquisitionDate'] > self.dset['t1r']['time']['AcquisitionDate']:
                self.dset['t1r'],self.dset['t0'] = self.dset['t0'],self.dset['t1r']
            elif self.dset['t0']['time']['AcquisitionDate'] == self.dset['t1r']['time']['AcquisitionDate']:
                if self.dset['t0']['time']['AcquisitionTime'] > self.dset['t1r']['time']['AcquisitionTime']:
                    self.dset['t1r'],self.dset['t0'] = self.dset['t0'],self.dset['t1r']
        # for a study that crosses midnight, the study date could remain the same while acquisition dates change
        elif all(self.dset[v]['time']['StudyDate'] is not None for v in ['t0','t1r']):
            if self.dset['t0']['time']['StudyDate'] > self.dset['t1r']['time']['StudyDate']:
                self.dset['t1r'],self.dset['t0'] = self.dset['t0'],self.dset['t1r']
            elif self.dset['t0']['time']['StudyDate'] == self.dset['t1r']['time']['StudyDate']:
                if self.dset['t0']['time']['AcquisitionTime'] > self.dset['t1r']['time']['AcquisitionTime']:
                    self.dset['t1r'],self.dset['t0'] = self.dset['t0'],self.dset['t1r']
        else:
            print('T0,T1 times not detected')




# process single study directory containing multiple dicom series dirs
class DcmProcess():
    def __init__(self,case,studies,config):
        # collection of scans on any given date
        self.case = case
        self.studies = studies
        self.config = config
        # talairach reference image
        self.reference,self.affine = self.loadnifti('mni_icbm152_t1_tal_nlin_sym_09a.nii',dir=os.path.join(self.config.UIdatadir,'mni152'))
        mask,_ = self.loadnifti('mni_icbm152_t1_tal_nlin_sym_09a_mask.nii',dir=os.path.join(self.config.UIdatadir,'mni152'))
        self.reference *= mask
        self.reference = self.rescale(self.reference)
        return

    # register time point0 to talairach, and all subsequent time points to time point 0
    def postprocess(self):
        s = self.studies[0]
        # ant register can't handle this flip.
        for dt in s.dtag:
            if s.dset[dt]['ex']:
                s.dset[dt]['d'] = np.flip(s.dset[dt]['d'],axis=1)
        if s.dset['t1+']['ex']:
            dref = 't1+'
        elif s.dset['t1']['ex']:
            dref = 't1'
        else:
            raise ValueError('No T1 data to register to')
        s.dset[dref]['d'],tx = self.register(self.reference,s.dset[dref]['d'],transform='Rigid')
        if True:
            self.writenifti(s.dset[dref]['d'],os.path.join(self.config.UIlocaldir,self.case,dref+'_talairach.nii'),affine=self.affine)
            # self.writenifti(self.reference,os.path.join(self.config.UIlocaldir,self.case,'talairach.nii'),affine=self.affine)

        for dt in [dt for dt in s.dtag if dt != dref]:
            if s.dset[dt]['ex']:
                s.dset[dt]['d'] = self.tx(self.reference,s.dset[dt]['d'],tx)

        # remainder of studies register to first study
        for s in self.studies[1:]:
            for dt in s.dtag:
                if s.dset[dt]['ex']:
                    s.dset[dt]['d'] = np.flip(s.dset[dt]['d'],axis=1)
            if s.dset['t1+']['ex']:
                dref = 't1+'
                s.dset[dref]['d'],tx = self.register(self.studies[0].dset[dref]['d'],s.dset[dref]['d'],transform='Rigid')
            else:
                raise ValueError('No T1+ to register to')
            for dt in [dt for dt in s.dtag if dt != dref]:
                if s.dset[dt]['ex']:
                    s.dset[dt]['d'] = self.tx(self.studies[0].dset[dref]['d'],s.dset[dt]['d'],tx)

        self.write_all()
        return



    def process_studydirs(self):
        pname = os.path.join(self.config.UIlocaldir,self.case,'studies.pkl')
        if os.path.exists(pname):
            with open(pname,'rb') as fp:
                self.studies = []
                self.studies = pickle.load(fp)
        else:
            for i,s in enumerate(self.studies):
                self.preprocess(s)
            if True:
                with open(pname,'wb') as fp:
                    pickle.dump(self.studies,fp)

    
    # resampling, registration, bias correction
    def preprocess(self,study):

        print('case = {},{}'.format(self.case,study.studydir))
        self.localcasedir = os.path.join(self.config.UIlocaldir,self.case,study.studytimeattrs['StudyDate'])
        if not os.path.exists(self.localcasedir):
            os.makedirs(self.localcasedir)

        # resample to target matrix (t1+ for now)
        if True:
            for dt in ['flair','flair+','cbv']:
                if study.dset[dt]['ex'] and study.dset['t1+']['ex']:
                    print('Resampling ' + dt + ' into target space...')
                    study.dset[dt]['d'],study.dset[dt]['affine'] = self.resamplet2(study.dset['t1+']['d'],study.dset[dt]['d'],
                                                                        study.dset['t1+']['affine'],study.dset[dt]['affine'])
                    study.dset[dt]['d']= np.clip(study.dset[dt]['d'],0,None)


        if False:
            for t in ['t1pre','t1','t2','flair']:
                if study.dset[t]['ex']:
                    self.writenifti(study.dset[t]['d'],os.path.join(d,'img_'+t+'_resampled.nii.gz'),
                                        type='float',affine=self.ui.affine['t1'])
                    

        # skull strip

        # optional. use mask. not sure if it will be needed 
        if study.dset['t1+']['mask'] is not None:

            # pre-registration. for now assuming mask is a T1post so register T1pre,T2,flair
            # reference
            fixed_image = study.dset['t1+']['d']

            for dt in ['t1pre','flair','flair+']:
                # fname = os.path.join(d,'img_'+t+'_resampled.nii.gz')
                if study.dset[dt]['ex']:
                    moving_image = study.dset[dt]['d']
                    study.dset[dt]['d'],_ = self.register(fixed_image,moving_image,transform='Rigid')
                    if False:
                        self.ui.roiframe.WriteImage(study.dset[t]['d'],os.path.join(d,'img_'+t+'_preregistered.nii.gz'),
                                                type='float',affine=self.ui.affine['t1'])

            # apply mask
            for dt in ['t1','flair','flair+']:
                if study.dset[dt]['ex']:
                    study.dset[dt]['d'] = np.where(study.dset['t1+']['mask'],study.dset[dt]['d'],0)


        # attempt model extraction
        else:    
            for dt in ['t1+','flair+','t1','flair']:
                if study.dset[dt]['ex']:
                    study.dset[dt]['d'],study.dset[dt]['mask'] = self.extractbrain2(study.dset[dt]['d'],affine=study.dset[dt]['affine'],fname=dt)

            # could maybe just post-contrast mask for speed assuming no significant change
            # in pose.
            if False:
                for dt in ['t1','flair']:
                    if study.dset[dt]['ex']:
                        study.dset[dt]['d'] *= study.dset[dt+'+']['mask']


        # registration. some RELCCBV exports may have their own study number
        # and need to be reconnected with the source T1 scan by study date tag
        if True:
            if study.dset['t1+']['ex']:
                fixed_image = study.dset['t1+']['d']
                for dt in ['t1','flair','flair+']:
                    fname = os.path.join(self.localcasedir,dt+'_resampled.nii.gz')
                    if study.dset[dt]['ex']:
                        moving_image = study.dset[dt]['d']
                        study.dset[dt]['d'],_ = self.register(fixed_image,moving_image,transform='Rigid')
                        if False:
                            self.writenifti(study.dset[t]['d'],os.path.join(d,'img_'+t+'_registered.nii.gz'),
                                                    type='float',affine=self.ui.affine['t1'])
            else:
                print('No T1+ to register on, skipping...')


        # bias correction.
        for dt in ['t1','t1+','flair','flair+']:
            if study.dset[dt]['ex']:   
                study.dset[dt]['d'] = self.n4bias(study.dset[dt]['d'])

        # rescale the data
        # if necessary clip any negative values introduced by the processing
        for dt in ['t1','t1+','flair','flair+']:
            if study.dset[dt]['ex']:
                if np.min(study.dset[dt]['d']) < 0:
                    study.dset[dt]['d'][study.dset[dt]['d'] < 0] = 0
                study.dset[dt]['d'] = self.rescale(study.dset[dt]['d'])

        # save nifti files for future use
        if False:
            for dt in ['flair+','t1','t1+','flair']:
                if study.dset[dt]['ex']:
                    self.writenifti(study.dset[dt]['d'],os.path.join(self.localcasedir,dt+'_processed.nii'),
                                                type='float',affine=study.dset['t1+']['affine'])
            for dt in ['cbv']: # pending solution for registration
                if study.dset[dt]['ex']:
                    self.writenifti(study.dset[dt]['d'],os.path.join(self.localcasedir,dt+'_processed.nii'),
                                                type='float',affine=study.dset['t1+']['affine'])

        return


    # newer from temporal regression
    # not sure if needed
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
            dset['t1r']['d'],_ = self.register(dset['t0']['d'],dset['t1r']['d'],type='ants')
            for dt in dtag:
                niftifile = os.path.join(npath,dt,dt+'.nii')
                self.writenifti(dset[dt]['d'],niftifile,type=dset[dt]['d'].dtype.name,affine=dset['t0']['affine'])

        # skull extraction
        if rc[case]['dcmtype'] == 'MR':
            for dt in dtag:
                command = 'conda run -n hdbet hd-bet '
                command += ' -i ' + os.path.join(npath,dt,dt+'.nii')
                res = os.system(command)
                dset[dt]['d'],_ = self.loadnifti(dt+'_bet.nii.gz',dir=os.path.join(npath,dt))


        # post-registration
        if rc[case]['dcmtype'] == 'MR':
            dset['t1r']['d'],_ = self.register(dset['t0']['d'],dset['t1r']['d'],type='ants')

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




    # skull extraction
    # replace with hd-bet
    def extractbrain2(self,img_arr_input,affine=None,fname=None):
        print('extract brain')
        img_arr = copy.deepcopy(img_arr_input)
        if fname is None:
            fname = 'temp'
        self.writenifti(img_arr,os.path.join(self.localcasedir,fname+'.nii'),affine=affine,norm=False,type='float')

        command = 'conda run -n hdbet hd-bet '
        command += ' -i ' + os.path.join(self.localcasedir,fname+'.nii')
        res = os.system(command)
        img_arr,_ = self.loadnifti(fname+'_bet.nii.gz',dir=self.localcasedir)
        img_arr_mask,_ = self.loadnifti(fname+'_bet_mask.nii.gz',dir=self.localcasedir)
        if fname == 'temp':
            for f in glob.glob(os.path.join(self.localcasedir,'temp*')):
                os.remove(f)
        return img_arr,img_arr_mask

    # original from brainmage
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

    # registration
    def register(self,img_arr_fixed,img_arr_moving,transform='Affine'):
        print('register fixed, moving')

        fixed_ants = ants.from_numpy(img_arr_fixed)
        moving_ants = ants.from_numpy(img_arr_moving)
        mytx = ants.registration(fixed=fixed_ants, moving=moving_ants, type_of_transform = transform )
        img_arr_reg = mytx['warpedmovout'].numpy()
        a=1

        return img_arr_reg,mytx['fwdtransforms']

    # transform
    def tx(self,img_arr_fixed,img_arr_moving,tx):
        print('transform fixed, moving')

        fixed_ants = ants.from_numpy(img_arr_fixed)
        moving_ants = ants.from_numpy(img_arr_moving)
        img_arr_tx = ants.apply_transforms(fixed_ants, moving_ants, tx).numpy()

        return img_arr_tx

    # operates on a single image channel 
    def rescale(self,img_arr,vmin=None,vmax=None):
        scaled_arr =  np.zeros(np.shape(img_arr))
        if vmin is None:
            minv = np.min(img_arr)
        else:
            minv = vmin
        if vmax is None:
            maxv = np.max(img_arr)
        else:
            maxv = vmax
        assert(maxv>minv)
        scaled_arr = (img_arr-minv) / (maxv-minv)
        scaled_arr = np.clip(scaled_arr,a_min=0,a_max=1)
        return scaled_arr

