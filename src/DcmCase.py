# methods for pre-processing dicom input dirs

import os,sys
import numpy as np
import glob
import copy
import re
import logging
import copy
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

# convenience function
def cp(item):
    return copy.deepcopy(item)


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

        if True:
            self.segment()


    # load all studies of current case
    def load_studydirs(self):

        for i,sd in enumerate(self.studydirs):
            self.studies.append(DcmStudy(self.case,sd,self.config))
            self.studies[i].loaddata()
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

        # temporary. load ET mask
        # this will be implemented later with a DL model
        if False:
            for s in self.studies:
                s.dset['ET']['d'],_ = s.loadnifti(os.path.join(self.config.UIlocaldir,self.case,'objectmask_ET.nii.gz'))
                s.dset['ET']['affine'] = s.dset['t1+']['affine']
                s.dset['ET']['ex'] = True

        return

    # resample,register,bias correction
    def process_studydirs(self):
        pname = os.path.join(self.config.UIlocaldir,self.case,'studies.pkl')
        if os.path.exists(pname):
            with open(pname,'rb') as fp:
                self.studies = []
                self.studies = pickle.load(fp)
                if False:
                    for i,s in enumerate(self.studies):
                        s.normalstats()
        else:
            for i,s in enumerate(self.studies):
                s.preprocess()
            if False:
                with open(pname,'wb') as fp:
                    pickle.dump(self.studies,fp)


    # register time point0 to talairach, and all subsequent time points to time point 0
    def process_timepoints(self):
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
        s.dset[dref]['d'],tx = s.register(s.dset['ref']['d'],s.dset[dref]['d'],transform='Rigid')
        if False:
            s.writenifti(s.dset[dref]['d'],os.path.join(self.config.UIlocaldir,self.case,dref+'_talairach.nii'),affine=s.dset['ref']['affine'])

        for dt in [dt for dt in s.dtag if dt != dref]:
            if s.dset[dt]['ex']:
                s.dset[dt]['d'] = s.tx(s.dset['ref']['d'],s.dset[dt]['d'],tx)

        # remainder of studies register to first study
        for s in self.studies[1:]:
            for dt in s.dtag:
                if s.dset[dt]['ex']:
                    s.dset[dt]['d'] = np.flip(s.dset[dt]['d'],axis=1)
            if s.dset['t1+']['ex']:
                dref = 't1+'
                s.dset[dref]['d'],tx = s.register(self.studies[0].dset[dref]['d'],s.dset[dref]['d'],transform='Rigid')
            else:
                raise ValueError('No T1+ to register to')
            for dt in [dt for dt in s.dtag if dt != dref]:
                if s.dset[dt]['ex']:
                    s.dset[dt]['d'] = s.tx(self.studies[0].dset[dref]['d'],s.dset[dt]['d'],tx)

        self.write_all()
        return


    def write_all(self):
        # save nifti files for future use
        for s in self.studies:
            localstudydir = os.path.join(self.config.UIlocaldir,self.case,s.studytimeattrs['StudyDate'])
            for dt in s.dtag:
                if s.dset[dt]['ex']:
                    s.writenifti(s.dset[dt]['d'],os.path.join(localstudydir,dt+'_processed.nii'),
                                                type='float',affine=s.dset['ref']['affine'])
                    
    def segment(self):
        for s in self.studies:
            s.localstudydir = os.path.join(self.config.UIlocaldir,self.case,s.studytimeattrs['StudyDate'])
            s.segment()


class Study():

    def __init__(self,case,d,channellist = None):
        self.studydir = d
        self.case = case
        self.date = None
        self.casedir = None
        self.localstudydir = None
        if channellist is None:
            self.channels = {0:'t1+',1:'flair'}
        else:
            self.channels = {k:v for k,v in enumerate(channellist)}

        # list of attributes for each image volume
        self.dprop = {'d':None,'time':None,'affine':None,'ex':False,'max':0,'min':0}
        self.dset = {}
        # reference image used for registrations
        self.dset['ref'] = cp(self.dprop)
        # main data 
        self.dset['raw'] = {v:cp(self.dprop) for v in self.channels.values()}
        # cbv data, if available
        self.dset['cbv'] = cp(self.dprop)
        # blast segmentation
        self.dset['seg_raw'] = {v:cp(self.dprop) for v in self.channels.values()}
        # z-scores of the raw data
        self.dset['z'] = {v:cp(self.dprop) for v in self.channels.values()}
        # color overlay of the z-scores within a masked ROI
        self.dset['zoverlay'] = {v:cp(self.dprop) for v in self.channels.values()}
        # color overlay of the CBV within a masked ROI
        self.dset['cbvoverlay'] = {v:cp(self.dprop) for v in self.channels.values()}
        # color overlay of the blast segmentation
        self.dset['seg_raw_fusion'] = {v:cp(self.dprop) for v in self.channels.values()}
        # copy for display purposes which can be scaled for colormap. maybe not needed?
        self.dset['seg_raw_fusion_d'] = {v:cp(self.dprop) for v in self.channels.values()}
        # color overlay of the final ROI selected from blast segmentation
        self.dset['seg_fusion'] = {v:cp(self.dprop) for v in self.channels.values()}
        # copy for colormap scaling
        self.dset['seg_fusion_d'] = {v:cp(self.dprop) for v in self.channels.values()}

        
        # storage for masks derived from blast segmentation or nnUNet
        self.mask =  {'ET':{'d':None,'affine':None,'ex':False},
                     'WT':{'d':None,'affine':None,'ex':False},
                     'ETblast':{'d':None,'affine':None,'ex':False},
                     'WTblast':{'d':None,'affine':None,'ex':False},
                     'ETunet':{'d':None,'affine':None,'ex':False},
                     'WTunet':{'d':None,'affine':None,'ex':False},
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
            os.system('gzip --force "{}"'.format(filename))


class NiftiStudy(Study):

    def __init__(self,case,d):    # convenience function

        super().__init__(case,d)

    def loaddata(self):
        files = os.listdir(self.studydir)
        # load channels
        for dt in self.channels.values(): 
            # by convention '_processed' is the final output from dcm preprocess()
            dt_file = dt + '_processed.nii.gz'
            if dt_file in files:
                self.dset['raw'][dt]['d'],self.dset['raw'][dt]['affine'] = self.loadnifti(dt_file)
                self.dset['raw'][dt]['max'] = np.max(self.dset['raw'][dt]['d'])
                self.dset['raw'][dt]['min'] = np.min(self.dset['raw'][dt]['d'])
                self.dset['raw'][dt]['ex'] = True
            dt_file = 'z' + dt + '_processed.nii.gz'
            if dt_file in files:
                self.dset['z'][dt]['d'],_ = self.loadnifti(dt_file)
                self.dset['z'][dt]['max'] = np.max(self.dset['z'][dt]['d'])
                self.dset['z'][dt]['min'] = np.min(self.dset['z'][dt]['d'])
                self.dset['z'][dt]['ex'] = True
                # self.dset[dt[:-1]]['max'] = self.dset[dt]['max']
                # self.dset[dt[:-1]]['min'] = self.dset[dt]['min']

        # load other
        for dt in ['cbv','ref']:
            dt_file = dt + '_processed.nii.gz'
            if dt_file in files:
                self.dset[dt]['d'],_ = self.loadnifti(dt_file)
                self.dset[dt]['ex'] = True

        # load masks
        for dt in ['ET','WT']:
            dt_file = dt + '_processed.nii.gz'
            if dt_file in files:
                self.mask[dt]['d'],_ = self.loadnifti(dt_file)
                self.mask[dt]['ex'] = True
                # for now, ET_processed is an nnunet segmentation, not a BLAST
                # so store a copy separately
                self.mask[dt+'unet']['d'] = np.copy(self.mask[dt]['d'])
                self.mask[dt+'unet']['ex'] = True
        return



class DcmStudy(Study):

    def __init__(self,case,d,config):
        super().__init__(case,d)
        self.config = config
        self.localcasedir = os.path.join(self.config.UIlocaldir,self.case)
        # list of time attributes to check
        self.seriestimeattrs = ['AcquisitionTime']
        self.studytimeattrs = {'StudyDate':None,'StudyTime':None}
        self.date = None
        # params for z-score
        self.params = {dt:{'mean':0,'std':0} for dt in ['t1','t1+','flair','flair']}
        # reference for talairach coords
        self.dset['ref']['d'],self.dset['ref']['affine'] = self.loadnifti('mni_icbm152_t1_tal_nlin_sym_09a.nii',dir=os.path.join(self.config.UIdatadir,'mni152'))
        mask,_ = self.loadnifti('mni_icbm152_t1_tal_nlin_sym_09a_mask.nii',dir=os.path.join(self.config.UIdatadir,'mni152'))
        self.dset['ref']['d'] *= mask
        # self.dset['ref']['d'] = self.rescale(self.dset['ref']['d'])

        return
    
    # load up multiple series directories in the provided study directory
    def loaddata(self):
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

            # assuming flair scans aren't designated pre/post, and will generally
            # be post, but never would be both.  
            elif any([f in ds0.SeriesDescription.lower() for f in ['flair','fluid']]):
                dt = 'flair'

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
    
    # from temporal regression. not sure if needed.
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


    # ###################
    # processing routines
    #####################

    
    # resampling, registration, bias correction
    def preprocess(self):

        print('case = {},{}'.format(self.case,self.studydir))
        self.localstudydir = os.path.join(self.config.UIlocaldir,self.case,self.studytimeattrs['StudyDate'])
        if not os.path.exists(self.localstudydir):
            os.makedirs(self.localstudydir)

        # resample to target matrix (t1+ for now)
        if True:
            for dt in ['flair','cbv']:
                if self.dset[dt]['ex'] and self.dset['t1+']['ex']:
                    print('Resampling ' + dt + ' into target space...')
                    self.dset[dt]['d'],self.dset[dt]['affine'] = self.resamplet2(self.dset['t1+']['d'],self.dset[dt]['d'],
                                                                        self.dset['t1+']['affine'],self.dset[dt]['affine'])
                    self.dset[dt]['d']= np.clip(self.dset[dt]['d'],0,None)


        if False:
            for t in ['t1pre','t1','t2','flair']:
                if self.dset[t]['ex']:
                    self.writenifti(self.dset[t]['d'],os.path.join(d,'img_'+t+'_resampled.nii.gz'),
                                        type='float',affine=self.ui.affine['t1'])
                    

        # skull strip

        # optional. use mask. not sure if it will be needed 
        if self.dset['t1+']['mask'] is not None:

            # pre-registration. for now assuming mask is a T1post so register T1pre,T2,flair
            # reference
            fixed_image = self.dset['t1+']['d']

            for dt in ['t1pre','flair','t2']:
                # fname = os.path.join(d,'img_'+t+'_resampled.nii.gz')
                if self.dset[dt]['ex']:
                    moving_image = self.dset[dt]['d']
                    self.dset[dt]['d'],_ = self.register(fixed_image,moving_image,transform='Rigid')
                    if False:
                        self.ui.roiframe.WriteImage(self.dset[t]['d'],os.path.join(d,'img_'+t+'_preregistered.nii.gz'),
                                                type='float',affine=self.ui.affine['t1'])

            # apply mask
            for dt in ['t1','flair','t2']:
                if self.dset[dt]['ex']:
                    self.dset[dt]['d'] = np.where(self.dset['t1+']['mask'],self.dset[dt]['d'],0)


        # attempt model extraction
        else:    
            for dt in ['t1+','t2','t1','flair']:
                if self.dset[dt]['ex']:
                    self.dset[dt]['d'],self.dset[dt]['mask'] = self.extractbrain2(self.dset[dt]['d'],affine=self.dset[dt]['affine'],fname=dt)

            # could maybe just post-contrast mask for speed assuming no significant change
            # in pose.
            if False:
                for dt in ['t1','flair']:
                    if self.dset[dt]['ex']:
                        self.dset[dt]['d'] *= self.dset[dt+'+']['mask']


        # registration. some RELCCBV exports may have their own study number
        # and need to be reconnected with the source T1 scan by study date tag
        if True:
            if self.dset['t1+']['ex']:
                fixed_image = self.dset['t1+']['d']
                for dt in ['t1','flair','t2']:
                    fname = os.path.join(self.localstudydir,dt+'_resampled.nii.gz')
                    if self.dset[dt]['ex']:
                        moving_image = self.dset[dt]['d']
                        self.dset[dt]['d'],_ = self.register(fixed_image,moving_image,transform='Rigid')
                        if False:
                            self.writenifti(self.dset[t]['d'],os.path.join(d,'img_'+t+'_registered.nii.gz'),
                                                    type='float',affine=self.ui.affine['t1'])
            else:
                print('No T1+ to register on, skipping...')


        # bias correction.
        self.dbias = {} # working data for calculating z-scores
        for dt in ['t1','t1+','flair','t2']:
            if self.dset[dt]['ex']:   
                self.dset['z'+dt]['d'] = np.copy(self.n4bias(self.dset[dt]['d']))
                self.dset['z'+dt]['ex'] = True

        # if necessary clip any negative values introduced by the processing
        for dt in ['t1','t1+','flair','t2']:
            if self.dset[dt]['ex']:
                if np.min(self.dset['z'+dt]['d']) < 0:
                    self.dset['z'+dt]['d'][self.dset['z'+dt]['d'] < 0] = 0
                # self.dset[dt]['d'] = self.rescale(self.dset[dt]['d'])

        # normal brain stats and z-score images
        self.normalstats()

        # save nifti files for future use
        if False:
            for dt in ['t2','t1','t1+','flair']:
                if self.dset[dt]['ex']:
                    self.writenifti(self.dset[dt]['d'],os.path.join(self.localstudydir,dt+'_processed.nii'),
                                                type='float',affine=self.dset['t1+']['affine'])
            for dt in ['cbv']: # pending solution for registration
                if self.dset[dt]['ex']:
                    self.writenifti(self.dset[dt]['d'],os.path.join(self.localstudydir,dt+'_processed.nii'),
                                                type='float',affine=self.dset['t1+']['affine'])

        return


    def normalstats(self,event=None):
        print('normal stats')
        # do kmeans
        # Creates a matrix of voxels for normal brain slice

        X={}
        vset = {}
        for dt2 in [('flair','t1'),('flair','t2')]:
            if self.dset['z'+dt2[0]]['ex'] and self.dset['z'+dt2[1]]['ex']:
                region_of_support = np.where(self.dset[dt2[0]]['d']*self.dset[dt2[1]]['d'] >0)
                background = np.where(self.dset[dt2[0]]['d']*self.dset[dt2[1]]['d'] == 0)
                # vset = np.zeros_like(region_of_support,dtype='float')
                for dt in dt2:
                    vset[dt] = np.ravel(self.dset['z'+dt]['d'][region_of_support])
                X[dt2] = np.column_stack((vset[dt2[0]],vset[dt2[1]]))

        np.random.seed(1)
        for i,layer in enumerate(X.keys()):
            kmeans = KMeans(n_clusters=2,n_init='auto').fit(X[layer])
            background_cluster = np.argmax(np.power(kmeans.cluster_centers_[:,0],2)+np.power(kmeans.cluster_centers_[:,1],2))

            # Calculate stats for brain cluster
            for ii,dt in enumerate(layer):
                self.params[dt]['std'] = np.std(X[layer][kmeans.labels_==background_cluster,ii])
                self.params[dt]['mean'] = np.mean(X[layer][kmeans.labels_==background_cluster,ii])

                plt.figure(7),plt.clf()
                ax = plt.subplot(1,2,i+1)
                plt.scatter(X[layer][kmeans.labels_==1-background_cluster,0],X[layer][kmeans.labels_==1-background_cluster,1],c='b',s=1)
                plt.scatter(X[layer][kmeans.labels_==background_cluster,0],X[layer][kmeans.labels_==background_cluster,1],c='r',s=1)
                ax.set_aspect('equal')
                # ax.set_xlim(left=0,right=1.0)
                # ax.set_ylim(bottom=0,top=1.0)
                # plt.text(0,1.02,'{:.3f},{:.3f}'.format(self.params[layer]['mean'],self.params[layer]['std']))
                if False:
                    plt.show(block=False)

                self.dset['z'+dt]['d'] = ( self.dset['z'+dt]['d'] - self.params[dt]['mean']) / self.params[dt]['std']
                self.dset['z'+dt]['d'][background] = 0
                self.writenifti(self.dset['z'+dt]['d'],os.path.join(self.localstudydir,'z'+dt+'.nii'),affine=self.dset[dt]['affine'])
        plt.savefig(os.path.join(self.localcasedir,'scatterplot_normal.png'))

        return

    # tumour segmenation nnUNet
    def segment(self,dpath=None):
        print('segment tumour')
        if dpath is None:
            dpath = os.path.join(self.localstudydir,'nnunet')
            if not os.path.exists(dpath):
                os.mkdir(dpath)
        for dt,suffix in zip(['t1+','flair'],['0000','0003']):
            l1str = 'ln -s ' + os.path.join(self.localstudydir,dt+'_processed.nii.gz') + ' '
            l1str += os.path.join(dpath,self.studytimeattrs['StudyDate']+'_'+suffix+'.nii.gz')
            os.system(l1str)
            # self.writenifti(self.dset['z'+dt]['d'],os.path.join(dpath,self.studytimeattrs['StudyDate']+'_'+suffix+'.nii'))         


        if os.name == 'posix':
            command = 'conda run -n ptorch nnUNetv2_predict '
            command += ' -i ' + dpath
            command += ' -o ' + dpath
            command += ' -d137 -c 3d_fullres'
            res = os.system(command)
        elif os.name == 'nt':
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
            command1 = '\"'+activatebatch+'\" \"' + os.path.expanduser('~')+'\\anaconda3\envs\\hdbet\"'
            command2 = 'python \"' + os.path.join(self.config.HDBETPath,'HD_BET','hd-bet')
            command2 += '\" -i   \"' + tfile
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
        sfile = self.studytimeattrs['StudyDate'] + '.nii.gz'
        segmentation,affine = self.loadnifti(sfile,dpath)
        ET = np.zeros_like(segmentation)
        ET[segmentation == 3] = 1
        WT = np.zeros_like(segmentation)
        WT[segmentation > 0] = 1
        self.writenifti(ET,os.path.join(self.localstudydir,'ET_processed.nii'),affine=affine)
        self.writenifti(WT,os.path.join(self.localstudydir,'WT_processed.nii'),affine=affine)
        if False:
            os.remove(os.path.join(dpath,sfile))

        return 



    # brain extraction
    # now using hd-bet
    def extractbrain2(self,img_arr_input,affine=None,fname=None):
        print('extract brain')
        img_arr = copy.deepcopy(img_arr_input)
        if fname is None:
            fname = 'temp'
        tfile = os.path.join(self.localstudydir,fname+'.nii')
        self.writenifti(img_arr,tfile,affine=affine,norm=False,type='float')

        if os.name == 'posix':
            command = 'conda run -n hdbet hd-bet '
            command += ' -i ' + tfile
            res = os.system(command)
        elif os.name == 'nt':
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
            command1 = '\"'+activatebatch+'\" \"' + os.path.expanduser('~')+'\\anaconda3\envs\\hdbet\"'
            command2 = 'python \"' + os.path.join(self.config.HDBETPath,'HD_BET','hd-bet')
            command2 += '\" -i   \"' + tfile
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

        img_arr,_ = self.loadnifti(fname+'_bet.nii.gz',dir=self.localstudydir)
        img_arr_mask,_ = self.loadnifti(fname+'_bet_mask.nii.gz',dir=self.localstudydir)
        if fname == 'temp':
            for f in glob.glob(os.path.join(self.localstudydir,'temp*')):
                os.remove(f)
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

