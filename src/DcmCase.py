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
from matplotlib.path import Path
from cProfile import Profile
from pstats import SortKey,Stats
from enum import Enum
import ants

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import scipy
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice
from scipy.interpolate import splev, splrep
from sklearn.linear_model import LinearRegression,RANSACRegressor


# convenience function
def cp(item):
    return copy.deepcopy(item)


# Classes and methods for loading a collection of multiple dicom studies as a case
# and pre-processing to produce nifti output files which are then loaded into the
# viewer.

# a collection of dicom studies at several time points
# datadir - root directory containing one or mulitple cases
# casename - main directory of the current case, naming convention is 'M' + 5 digits for now
# studydirs - list of study directories in the current case
class Case():
    def __init__(self,casename,studydirs,datadir,config):

        self.config = config
        self.case = casename
        self.datadir = datadir
        self.casedir = os.path.join(self.datadir,self.case)
        self.studydirs = studydirs
        self.studies = []
        # convention for two displayed images: the left image is more recent/index1, and right image is earlier/index0
        # in future when more than two study/timepoints are available, self.timepoints will select for display
        self.timepoints = [0,1]


        self.load_studydirs()
        self.process_studydirs()
        self.process_timepoints()
        if False:
            self.regression()
            self.segment()


    # load all studies of current case
    def load_studydirs(self):

        for i,sd in enumerate(self.studydirs):
            print('loading {}\n'.format(sd))
            self.studies.append(DcmStudy(self.case,sd,self.config))
            self.studies[i].loaddata()
        # sort studies by time and number of series
        # self.studies = sorted(self.studies,key=lambda x:(x.studytimeattrs['StudyDate'],
        #                                                  len([t for t in x.dset.keys() if not x.dset[t]['ex']])))
        self.studies = sorted(self.studies,key=lambda x:(x.studytimeattrs['StudyDate'],
                                                         len([t for t in x.dset['raw'].keys() if not x.dset['raw'][t]['ex']])))
        # combine studies with common date, for now any series in the study with fewer series, is 
        # copied to the study with more series, over-writes are not being checked yet but should be well-behaved
        # some RELCCBV exports may have their own study number
        # and need to be reconnected with the source T1 scan by study date tag??
        # TODO: actually check the times, so a later series only overwrites an earlier series according to temporal
        # relation, ie possible rescan
        dates = sorted(list(set([d.studytimeattrs['StudyDate'] for d in self.studies])))
        studies = []
        for i,d in enumerate(dates):
            dstudies = [s for s in self.studies if s.studytimeattrs['StudyDate'] == d]
            for ds in dstudies[-1:0:-1]:
                for dc in ['raw']:
                    for series in list(ds.channels.values()):
                        if ds.dset[dc][series]['ex']:
                            dstudies[0].dset[dc][series]['d'] = np.copy(ds.dset[dc][series]['d'])
                            dstudies[0].dset[dc][series]['affine'] = np.copy(ds.dset[dc][series]['affine'])
                            dstudies[0].dset[dc][series]['time'] = np.copy(ds.dset[dc][series]['time'])
                            dstudies[0].dset[dc][series]['ex'] = True
                # for series in ['cbv']:
                #     if ds.dset[series]['ex']:
                #         dstudies[0].dset[series]['d'] = np.copy(ds.dset[series]['d'])
                #         dstudies[0].dset[series]['affine'] = np.copy(ds.dset[series]['affine'])
                #         dstudies[0].dset[series]['time'] = ds.dset[series]['time']
                #         dstudies[0].dset[series]['ex'] = True

                dstudies.remove(ds)
            studies.append(dstudies[0])
        self.studies = studies

        return

    # resample,register,bias correction
    def process_studydirs(self):
        pname = os.path.join(self.datadir,self.case,'studies.pkl')
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
            if True:
                with open(pname,'wb') as fp:
                    pickle.dump(self.studies,fp)


    # register time point0 to talairach, and all subsequent time points to time point 0
    def process_timepoints(self,flip=False):
        s = self.studies[0]
        # TODO. there is a possible flip in axis=1 that comes down from 
        # something that is missed/undone in the prior dicom processing.
        # ant register can't handle an exact flip in this dimension.
        # until it is figured out properly, optionally flip the data here 
        # if needed to facilitate ants
        if flip:
            for dc in ['raw','z','cbv','adc']:
                for dt in list(s.channels.values()):
                    if s.dset[dc][dt]['ex']:
                        s.dset[dc][dt]['d'] = np.flip(s.dset[dc][dt]['d'],axis=1)
        if s.dset['raw']['t1+']['ex']:
            dref = 't1+'
        elif s.dset['raw']['t1']['ex']:
            dref = 't1'
        else:
            raise ValueError('No T1 data to register to')
        # register the designated reference image to the talairach coords
        s.dset['raw'][dref]['d'],tx = s.register(s.dset['ref']['d'],s.dset['raw'][dref]['d'],transform='Rigid')
        if False:
            s.writenifti(s.dset['raw'][dref]['d'],os.path.join(self.datadir,self.case,dref+'_talairach.nii'),affine=s.dset['ref']['affine'])
        # apply that same registration transform to all remaining images in this study
        for dc in ['raw','z','cbv','adc']:
            for dt in list(s.channels.values()):
                if dt == dref and dc == 'raw':
                    continue
                else:
                    if s.dset[dc][dt]['ex']:
                        s.dset[dc][dt]['d'] = s.tx(s.dset['ref']['d'],s.dset[dc][dt]['d'],tx)

        # repeat process for remainder of studies
        for s in self.studies[1:]:
            if flip:
                for dc in ['raw','z','cbv','adc']:
                    for dt in list(s.channels.values()):
                        if s.dset[dc][dt]['ex']:
                            s.dset[dc][dt]['d'] = np.flip(s.dset[dc][dt]['d'],axis=1)
            if s.dset['raw']['t1+']['ex']:
                dref = 't1+'
                s.dset['raw'][dref]['d'],tx = s.register(self.studies[0].dset['raw'][dref]['d'],s.dset['raw'][dref]['d'],transform='Rigid')
            else:
                raise ValueError('No T1+ to register to')
            for dc in ['raw','z','cbv','adc']:
                for dt in list(s.channels.values()) :
                    if dt == dref and dc == 'raw':
                        continue
                    else:
                        if s.dset[dc][dt]['ex']:
                            s.dset[dc][dt]['d'] = s.tx(self.studies[0].dset['raw'][dref]['d'],s.dset[dc][dt]['d'],tx)

        self.write_all()
        return

    # save all data to nifti files for future use
    def write_all(self):
        for s in self.studies:
            localstudydir = os.path.join(self.datadir,self.case,s.studytimeattrs['StudyDate'])
            localniftidir = os.path.join(self.datadir,'dicom2nifti',self.case,s.studytimeattrs['StudyDate'])
            if not os.path.exists(localniftidir):
                os.makedirs(localniftidir,exist_ok=True)
            for dc in ['raw','z','cbv','adc']:
                for dt in list(s.channels.values()):
                    if s.dset[dc][dt]['ex']:
                        if dc in ['adc','cbv']:
                            dstr = dc + '_processed.nii'
                        elif dc == 'z':
                            dstr = 'z' + dt + '_processed.nii'
                        else:
                            dstr = dt+'_processed.nii'
                        s.writenifti(s.dset[dc][dt]['d'],os.path.join(localniftidir,dstr),
                                                    type='float',affine=s.dset['ref']['affine'])

    # run nnunet segmentation                
    def segment(self):
        for s in self.studies:
            s.localstudydir = os.path.join(self.datadir,self.case,s.studytimeattrs['StudyDate'])
            s.segment()

    # assumes there are just two time points for now
    # TODO: use self.ui.timepoints from gui to select time points
    def regression(self):
        for s in self.studies:
            s.normalize(['t1','t2','t1+','flair'])

        _,plotdata = self.get_data()

        for dt in ['t1','t2','t1+','flair']:
            # two studies hard-coded
            if self.studies[0].dset['raw'][dt]['ex'] and self.studies[1].dset['raw'][dt]['ex']:
                region_of_support = np.where((self.studies[0].dset['raw'][dt]['d'] > 0) & (self.studies[1].dset['raw'][dt]['d'] > 0))
                xdata = np.ravel(self.studies[self.timepoints[0]].dset['raw'][dt]['d_norm'][region_of_support]).reshape(-1,1)
                ydata = np.ravel(self.studies[self.timepoints[1]].dset['raw'][dt]['d_norm'][region_of_support])

                xfit_range = plotdata['MR']['d_norm']
                xfit = np.arange(xfit_range[0],xfit_range[1],np.diff(xfit_range)[0]/100).reshape(-1,1)

                mdl = LinearRegression()
                mdl_ransac = RANSACRegressor(random_state=0,max_trials=1000)

                mdls = [mdl_ransac]
                mdl_tags = ['_ransac']
                reg_image = {}

                for i,m in enumerate(mdls):

                    m.fit(xdata,ydata)

                    yfit = m.predict(xfit)
                    yhat = m.predict(xdata)
                    resid = ydata - yhat
                    reg_image[mdl_tags[i]] = np.zeros_like(self.studies[0].dset['raw'][dt]['d'])
                    reg_image[mdl_tags[i]][region_of_support] = resid
                    nsample = ydata.size
                    dof = nsample - m.n_features_in_
                    # TODO: one-side or two-side. 
                    alpha = 0.05
                    t = scipy.stats.t.ppf(1-alpha, dof)
                    s_err = np.sqrt(np.sum(resid**2) / dof)                    # standard deviation of the error
                    ci = np.ravel(t * s_err * np.sqrt(1+1/nsample + (xfit - np.mean(xfit))**2 / np.sum((xfit - np.mean(xfit))**2)))

                    # plot regression
                    if True:
                        self.plot_regression(xdata,ydata,resid,xfit,yfit,ci,self.casedir,
                                            tag='_'+dt+mdl_tags[i],save=True)

                    # create mask images for voxels beyond prediction interval
                    pts = np.vstack((xdata.flatten(),resid.flatten())).T
                    # form ci polygon
                    xpts_ci = np.concatenate((xfit,np.flip(xfit)),axis=0)
                    ypts_ci = np.concatenate((-ci,np.flip(ci)),axis=0).reshape(-1,1) 
                    pts_ci = np.hstack((xpts_ci,ypts_ci))
                    pts_ci = np.concatenate((pts_ci,np.atleast_2d(pts_ci[0,:])),axis=0) # close path

                    reg_set = np.zeros_like(resid)
                    if False:
                        # too slow for large dataset.
                        ci_path = Path(pts_ci,closed=True)
                        reg_set = ~(ci_path.contains_points(pts)).flatten()
                    else:
                        # just use the residuals for scalar comparison
                        reg_set[(resid > np.mean(ci)) | (resid < -np.mean(ci))] = True
                    reg_set_above = copy.deepcopy(reg_set)
                    reg_set_above[resid < 0] = False
                    reg_set_below = copy.deepcopy(reg_set)
                    reg_set_below[resid > 0] = False

                    # indivdual change masks
                    if False:
                        mask_below = np.zeros_like(self.studies[0].dset[dt]['d'],dtype='uint8')
                        mask_below[region_of_support] = reg_set_below
                        self.studies[0].writenifti(mask_below,os.path.join(self.casedir,'regression_mask_below_{}.nii'.format(mdl_tags[i])),
                                                affine=self.studies[0].dset['ref']['affine'])

                        mask_above = np.zeros_like(mask_below)
                        mask_above[region_of_support] = reg_set_above
                        self.studies[0].writenifti(mask_above,os.path.join(self.casedir,'regression_mask_above_{}.nii'.format(mdl_tags[i])),
                                                affine=self.studies[0].dset['ref']['affine'])

                    # for the combined mask, will have negative pixel changes indicated by a value less than background, so it will 
                    # map intuitively to a colorbar scale. but still using uint8, so make an offset of +2
                    # which will then be removed at display
                    mask = np.ones_like(self.studies[0].dset['raw'][dt]['d'],dtype='uint8')*2
                    mask[region_of_support] = reg_set_below * 1 + reg_set_above * 3 
                    mask[mask==0] = 2
                    self.studies[0].writenifti(mask,os.path.join(self.studies[self.timepoints[1]].localstudydir,'tempo'+dt+'.nii'),
                                            affine=self.studies[0].dset['ref']['affine'])


        return
    
    # regression scatter plots
    def plot_regression(self,xdata,ydata,resid,xfit,yfit,ci,datadir,block=False,save=False,tag=''):
        _,plotdata = self.get_data()
        fig = plt.figure(11,figsize=(5,3))
        fig.clf()
        ax2 = plt.subplot(1,2,1)
        ax2.cla()
        ax2.plot(xfit,yfit*0,'b')
        # this tuple is a string in json
        plt.xlim(plotdata['MR']['d_norm'])
        plt.xlabel('time 0')
        plt.ylabel('residual')
        ax2.fill_between(np.ravel(xfit), ci, -ci, color="#0277bd", edgecolor=None,alpha=0.5,zorder=2)
        ax2.set_aspect('equal')
        plt.scatter(xdata[::1000],resid[::1000],s=1,c='r',zorder=1)

        ax3 = plt.subplot(1,2,2)
        ax3.cla()
        ax3.plot(xfit,yfit,'b')
        plt.xlabel('time 0')
        plt.ylabel('time 1')
        ax3.fill_between(np.ravel(xfit), yfit + ci, yfit - ci, color="#0277bd", edgecolor=None,alpha=0.5,zorder=2)
        ax3.set_aspect('equal')
        plt.scatter(xdata[::1000],ydata[::1000],s=1,c='r',zorder=1)
        plt.xlim(plotdata['MR']['d_norm'])
        plt.ylim(plotdata['MR']['d_norm'])
        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(datadir,'regression{:s}.png'.format(tag)))
        else:
            plt.show(block=block)



    # temporary arrangement for some hard-coded range values. move to config probably
    def get_data(self):

        # default colors for plots
        defcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        # some hard-coded limits for plots
        plotdata  = {
            "CT":
                {
                    "d":(0,200),
                    "d_norm":(-2,2),
                    "d_diff":(-50,50),
                    "d_norm_diff":(-1,1)
                },
            "MR":
                {
                    "d":(0,500),
                    "d_norm":(-2,2),
                    "d_diff":(-100,100),
                    "d_norm_diff":(-1,1)
                }
            }

        return defcolors,plotdata

# set of dicom series at one time point of a Case
# base class for input dicom images or output nifti images
class Study():

    def __init__(self,case,d,channellist = None):
        self.studydir = d
        self.case = case
        self.date = None
        self.localstudydir = None
        if channellist is None:
            self.channels = {0:'t1',1:'t1+',2:'t2',3:'flair',4:'dwi'}
        else:
            self.channels = {k:v for k,v in enumerate(channellist)}

        ####################################
        # main data structure for the viewer
        ####################################

        # list of attributes for each image volume. 'd' is the main data array,
        # 'ex' is existence as a convenience for checking whether populated. 'dref' is an optional image volume
        # that may be associated with 'd' for any reason
        self.dprop = {'d':None,'time':None,'affine':None,'ex':False,'max':0,'min':0,'w':0,'l':0,'mask':None,'dref':None}
        # special case for blast overlay layers
        self.dprop_layer = {'dET':None,'dT2 hyper':None,'ex':False}

        # the dataset structure could be its own class, but for now is just a dict
        self.dset = {}
        # reference image used only for registration purposes
        self.dset['ref'] = cp(self.dprop)
        # main raw imaging data 
        self.dset['raw'] = {v:cp(self.dprop) for v in self.channels.values()}
        # cbv data, if available. this is not channel-specific so just create references for dummy channels
        self.dset['cbv'] = cp(self.dprop)
        for v in self.channels.values():
            self.dset['cbv'][v] = self.dset['cbv'] 
        # adc data, if available. this is not channel-specific so just create references for dummy channels
        self.dset['adc'] = cp(self.dprop)
        for v in self.channels.values():
            self.dset['adc'][v] = self.dset['adc'] 
        # raw blast segmentation
        self.dset['seg_raw'] = {v:cp(self.dprop) for v in self.channels.values()}
        # z-score image
        self.dset['z'] = {v:cp(self.dprop) for v in self.channels.values()}
        # SAM segmentation
        self.dset['seg_sam'] = {v:cp(self.dprop) for v in self.channels.values()}
        # tempo regression differences of the 'raw' data at two time points
        self.dset['tempo'] = {v:cp(self.dprop) for v in self.channels.values()}
        # color overlay of the z-scores
        self.dset['zoverlay'] = {v:cp(self.dprop) for v in self.channels.values()}
        # color overlay of the CBV
        self.dset['cbvoverlay'] = {v:cp(self.dprop) for v in self.channels.values()}
        # color overlay of the regression.
        self.dset['tempooverlay'] = {v:cp(self.dprop) for v in self.channels.values()}
        # color overlay of the raw blast segmentation. has different keys for separate layers
        self.dset['seg_raw_fusion'] = {v:cp(self.dprop_layer) for v in self.channels.values()}
        # a copy for display purposes which can be scaled for colormap. maybe not needed?
        self.dset['seg_raw_fusion_d'] = {v:cp(self.dprop_layer) for v in self.channels.values()}
        # color overlay of the final smoothed ROI created from raw blast segmentation
        self.dset['seg_fusion'] = {v:cp(self.dprop) for v in self.channels.values()}
        # color overlay of the SAM of the final smoothed ROI 
        self.dset['sam_fusion'] = {v:cp(self.dprop) for v in self.channels.values()}
        # copy for colormap scaling
        self.dset['seg_fusion_d'] = {v:cp(self.dprop) for v in self.channels.values()}

        
        # storage for masks derived from blast segmentation or nnUNet
        mask_layer = {'ET':{'d':None,'ex':False},'TC':{'d':None,'ex':False},'WT':{'d':None,'ex':False}}
        self.mask =  copy.deepcopy(mask_layer)
        for m in ['blast','sam','unet','gt']:
            self.mask[m] = copy.deepcopy(mask_layer)

        self.dtag = [k for k in self.dset.keys()]
        self.date = None
        return
    
    # load a single nifti file
    def loadnifti(self,t1_file,dir=None,type='uint8'):
        img_arr_t1 = None
        if dir is None:
            dir = self.studydir
        try:
            img_nb_t1 = nb.load(os.path.join(dir,t1_file))
        except IOError as e:
            print('Can\'t import {}'.format(t1_file))
            return None,None
        affine = copy.copy(img_nb_t1.affine)
        img_arr_t1 = np.array(img_nb_t1.dataobj)
        # modify the affine to match itksnap convention
        for i in range(2):
            if affine[i,i] > 0:
                affine[i,3] += (img_nb_t1.shape[i]-1) * affine[i,i]
                affine[i,i] = -1*(affine[i,i])
                # will use flips for now for speed
                img_arr_t1 = np.flip(img_arr_t1,axis=i)
        # this takes too long and requires re-masking
        if False:
            img_nb_t1 = nb.processing.resample_from_to(img_nb_t1,(img_nb_t1.shape,affine))

        nb_header = img_nb_t1.header.copy()
        # nibabel convention will be transposed to sitk convention
        img_arr_t1 = np.transpose(img_arr_t1,axes=(2,1,0))
        if type is not None:
            img_arr_t1 = img_arr_t1.astype(type)

        return img_arr_t1,affine


    # write a single nifti file. use uint8 for masks 
    def writenifti(self,img_arr,filename,header=None,norm=False,type='float64',affine=None):
        img_arr_cp = copy.deepcopy(img_arr)
        if norm:
            img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
        # using nibabel nifti coordinates
        img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
        nb.save(img_nb,filename)
        if True:
            os.system('gzip --force "{}"'.format(filename))


    # normalize histograms for regression
    def normalize(self,dtag):

        # hard-coded points on the cumulative density
        slims = (.2,.8)

        for dt in dtag:
            if self.dset['raw'][dt]['ex']:
                norm_vals = {dt:{}}
                region_of_support = np.where((self.dset['raw'][dt]['d'] > 0))
                background = np.where((self.dset['raw'][dt]['d'] == 0))
                # 'counts' is (uniquevals, counts)
                norm_vals[dt]['counts'] = np.unique(np.round(self.dset['raw'][dt]['d'][region_of_support]).reshape(-1),
                                                                return_counts=True)
                # calculate normalized quantiles for each array
                norm_vals[dt]['q'] = np.cumsum(norm_vals[dt]['counts'][1]) / len(region_of_support[0])
                # smoothing spline
                spl = splrep(norm_vals[dt]['counts'][0],norm_vals[dt]['q'],s=0.01)
                norm_vals[dt]['spl_q'] = splev(norm_vals[dt]['counts'][0],spl)
                # take 20th,80th quantiles for normalization
                norm_vals[dt]['slim'] = np.array([np.argmin(norm_vals[dt]['spl_q'] < slims[0]),np.argmin(norm_vals[dt]['spl_q'] < slims[1])])
                self.dset['raw'][dt]['d_norm'] = np.copy(self.dset['raw'][dt]['d'])
                self.dset['raw'][dt]['d_norm'] -= norm_vals[dt]['counts'][0][norm_vals[dt]['slim'][0]]
                self.dset['raw'][dt]['d_norm'] /= (norm_vals[dt]['counts'][0][norm_vals[dt]['slim'][1]] - norm_vals[dt]['counts'][0][norm_vals[dt]['slim'][0]])
                if False:
                    self.dset[dt]['d_norm'] *= self.dset[dt]['mask']
                else:
                    self.dset['raw'][dt]['d_norm'][background] = 0
                if False:
                    writenifti(dset['raw'][dt][d+'_norm'],os.path.join(datadir,'t0',dt+'_bet_norm.nii'),affine=dset['t0']['affine'],type=float)

        return

# sub-class for handling the output nifti images and loading into viewer
class NiftiStudy(Study):

    def __init__(self,case,d,groundtruth=None):

        super().__init__(case,d)
        self.gt = groundtruth

    def loaddata(self):
        files = os.listdir(self.studydir)
        # load channels
        for dt in self.channels.values(): 
            # by convention '_processed' is the final output from dcm preprocess()
            dt_file = dt + '_processed.nii.gz'
            if dt_file in files:
                self.dset['raw'][dt]['d'],self.dset['raw'][dt]['affine'] = self.loadnifti(dt_file,type=np.float32)
                self.dset['raw'][dt]['max'] = np.max(self.dset['raw'][dt]['d'])
                self.dset['raw'][dt]['min'] = np.min(self.dset['raw'][dt]['d'])
                self.dset['raw'][dt]['ex'] = True
                self.dset['raw'][dt]['w'] = self.dset['raw'][dt]['max']/2
                self.dset['raw'][dt]['l'] = self.dset['raw'][dt]['max']/4
            # z-scores
            dt_file = 'z' + dt + '_processed.nii.gz'
            if dt_file in files:
                self.dset['z'][dt]['d'],_ = self.loadnifti(dt_file,type=np.float32)
                self.dset['z'][dt]['max'] = np.max(self.dset['z'][dt]['d'])
                self.dset['z'][dt]['min'] = np.min(self.dset['z'][dt]['d'])
                self.dset['z'][dt]['ex'] = True
                # self.dset[dt[:-1]]['max'] = self.dset[dt]['max']
                # self.dset[dt[:-1]]['min'] = self.dset[dt]['min']
            # tempo subtractions
            dt_file = 'tempo' + dt + '_processed.nii.gz'
            if dt_file in files:
                self.dset['tempo'][dt]['d'],_ = self.loadnifti(dt_file)
                # awkward special case for tempo. the mask is logically -1,0,1
                # for areas of reduction, neutral or enhancement. 
                # in generating an overlay, need to have neutral==0 to overlay only non-zero pixels
                # but for colormap, this has to be mapped into [0,0.5,1]
                # but for uint8 file, it is being stored as [1,2,3]
                # so the values have to be juggled a couple different ways
                # here subtract offset of +2 to place the original uint8 in a [-1,0,1] range
                self.dset['tempo'][dt]['d'] -= 2
                self.dset['tempo'][dt]['max'] = np.max(self.dset['tempo'][dt]['d'])
                self.dset['tempo'][dt]['min'] = np.min(self.dset['tempo'][dt]['d'])
                self.dset['tempo'][dt]['ex'] = True

        # load other
        for dt in ['cbv','ref','adc']:
            dt_file = dt + '_processed.nii.gz'
            if dt_file in files:
                self.dset[dt]['d'],_ = self.loadnifti(dt_file)                    
                self.dset[dt]['ex'] = True

        # load masks
        for dt in ['ET','WT']:
            dt_file = dt + '.nii.gz'
            if dt_file in files:
                self.mask[dt]['d'],_ = self.loadnifti(dt_file)
                self.mask[dt]['ex'] = True
                # for now, ET is an nnunet segmentation, not a BLAST
                # so store a copy separately
                self.mask['unet'][dt]['d'] = np.copy(self.mask[dt]['d'])
                self.mask['unet'][dt]['ex'] = True
            # check additionally for a blast mask
            dt_file = dt+'blast.nii.gz'
            if dt_file in files:
                self.mask['blast'][dt]['d'],_ = self.loadnifti(dt_file)
                self.mask['blast'][dt]['ex'] = True
            else:
                self.mask['blast'][dt]['d'] = np.ones_like(self.mask['ET']['d'])
                self.mask['blast'][dt]['ex'] = False

        # ground truth mask. self.gt is a compiled regex for filename matching
        if self.gt is not None:
            gt_files = [f for f in files if re.match(self.gt,f)]
            if len(gt_files) == 1:
                gt_file = gt_files[0]
                gtmask, _ = self.loadnifti(gt_file)
                # these values hardcoded for BraTS 2024 MET
                for dt,val in zip(['ET','TC','WT'],[3,1,2]):
                    self.mask['gt'][dt]['d'] = gtmask == val
                    self.mask['gt'][dt]['ex'] = True
                # to be confirmed? sam project defines TC as ET+necrosis, in BraTS TC is just necrosis
                self.mask['gt']['TC']['d'] = self.mask['gt']['TC']['d'] | self.mask['gt']['ET']['d']
            else:
                print('No or multiple files match ground truth regex')

        return


# other sub-class for the preprocessing pipeline
class DcmStudy(Study):

    def __init__(self,case,d,config,**kwargs):
        super().__init__(case,d,**kwargs)

        # override data structure in the case of derived datasets
        # cbv data. for purposes of dicom processing, this will be treated as a 'flair' channel. 
        self.dset['cbv'] = {v:cp(self.dprop) for v in self.channels.values()}
        # adc data. for purposes of dicom processing, this will be treated as a 'flair' channel. 
        self.dset['adc'] = {v:cp(self.dprop) for v in self.channels.values()}

        self.config = config
        # re-generating this path for output plots but awkward, needs better arrangement
        self.localcasedir = self.studydir.split(case)[0]+case
        # list of time attributes to check
        self.seriestimeattrs = ['AcquisitionTime']
        self.studytimeattrs = {'StudyDate':None,'StudyTime':None}
        self.date = None
        # params for z-score
        self.params = {dt:{'mean':0,'std':0} for dt in ['t1','t1+','flair','flair']}
        # reference for talairach coords
        self.dset['ref']['d'],self.dset['ref']['affine'] = self.loadnifti('mni_icbm152_t1_tal_nlin_sym_09a.nii',
                                                                          dir=os.path.join(self.config.UIdatadir,'mni152'),
                                                                          type='uint16')
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
            if (0x0020,0x0032) in ds0.keys():
                slice0 = ds0[(0x0020,0x0032)].value[2]
                ds = pd.dcmread(os.path.join(dpath,files[-1]))
                dslice = ds[(0x0020,0x0032)].value[2] - slice0
                if dslice < 0:
                    files = sorted(files,reverse=True)
            elif hasattr(ds0,'SliceThickness'):
                dslice = float(ds0['SliceThickness'].value)
            else:
                print('No slice thickness found, skipping...')
                continue
            print(ds0.SeriesDescription)
            for t in self.studytimeattrs.keys():
                if hasattr(ds0,t):
                    self.studytimeattrs[t] = getattr(ds0,t)

            dc = 'raw'
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

            #   
            elif any([f in ds0.SeriesDescription.lower() for f in ['tracew']]):
                dt = 'dwi'

            # not taking relcbv or relcbf, just relccbv. could use 'perf' as well.
            # note this may be exported in a separate studydir, without a matching t1
            # TODO: the matching t1 has to come from another studydir, based on time tags
            elif any([f in ds0.SeriesDescription.lower() for f in ['relccbv']]):
                dt = 'flair' # cbv will be stored arbitrarily as a flair channel for processing purposes
                dc = 'cbv'
            # likewise adc will be stored in the 'dwi' channel for processing purposes 
            elif any([f in ds0.SeriesDescription.lower() for f in ['adc']]):
                dt = 'dwi'
                dc = 'adc'
            else:
                dt = None
                continue

            if dt is not None:
                if dt in list(self.channels.values()):
                    dref = self.dset[dc][dt]
                    dref['ex'] = True
                    dref['d'] = np.zeros((len(files),ds0.Rows,ds0.Columns),dtype='uint16')
                    dref['affine'] = self.get_affine(ds0,dslice)
                    dref['d'][0,:,:] = ds0.pixel_array
                    for i,f in enumerate(files[1:]):
                        data = pd.dcmread(os.path.join(dpath,f))
                        dref['d'][i+1,:,:] = data.pixel_array

                    for t in self.seriestimeattrs:
                        if hasattr(ds0,t):
                            dref['time'] = float(getattr(ds0,t))
                            break

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

    # main routine for the preprocessing pipeline
    # eg resampling, registration, bias correction
    def preprocess(self,extract=False):

        print('case = {},{}'.format(self.case,self.studydir))
        # TODO. don't have a great arrangement for parallel dicom and nifti directories 
        # currently localstudydir just creates an output directory for nifti files based on the 
        # StudyDate attribute.
        self.localstudydir = os.path.join(self.localcasedir,self.studytimeattrs['StudyDate'])
        # self.localstudydir = os.path.join(self.ui.caseframe.datadir.get(),self.case,self.studytimeattrs['StudyDate'])
        if not os.path.exists(self.localstudydir):
            os.makedirs(self.localstudydir)

        # resample to target matrix (t1+ for now)
        if True:
            for dc in ['raw','cbv','adc']:
                for dt in ['flair','t1','t2','dwi']:
                    if self.dset[dc][dt]['ex'] and self.dset['raw']['t1+']['ex']:
                        print('Resampling ' + dc+','+dt + ' into target space...')
                        self.dset[dc][dt]['d'],self.dset[dc][dt]['affine'] = self.resamplet2(self.dset['raw']['t1+']['d'],self.dset[dc][dt]['d'],
                                                                            self.dset['raw']['t1+']['affine'],self.dset[dc][dt]['affine'])
                        self.dset[dc][dt]['d']= np.clip(self.dset[dc][dt]['d'],0,None)


        if False:
            for dt in ['t1pre','t1','t2','flair']:
                if self.dset['raw'][dt]['ex']:
                    self.writenifti(self.dset['raw'][dt]['d'],os.path.join(d,'img_'+t+'_resampled.nii.gz'),
                                        type='float',affine=self.ui.affine['t1'])
                    

        # skull strip
        # hd-bet model extraction
        if True:
            for dt in ['t1+','t2','t1','flair','dwi']:
                if self.dset['raw'][dt]['ex']:
                    if extract:
                        self.dset['raw'][dt]['d'],self.dset['raw'][dt]['mask'] = self.extractbrain2(self.dset['raw'][dt]['d'],
                                                                                                    affine=self.dset['raw'][dt]['affine'],fname=dt)
                    else:
                        self.dset['raw'][dt]['mask'] = np.ones_like(self.dset['raw'][dt]['d'])
                                                                                                
        # For ADC can just use the DWI mask
        if self.dset['adc']['dwi']['ex']:
            self.dset['adc']['dwi']['d'] *= self.dset['raw']['dwi']['mask']

        # preliminary registration, within the study to t1+ image. probably this is minimal
        # and skipping it shouldn't matter too much.
        # TODO. cbv should be handled here as well.
        if True:
            if self.dset['raw']['t1+']['ex']:
                fixed_image = self.dset['raw']['t1+']['d']
                for dt in ['t1','flair','t2']:
                    if self.dset['raw'][dt]['ex']:
                        moving_image = self.dset['raw'][dt]['d']
                        self.dset['raw'][dt]['d'],tx = self.register(fixed_image,moving_image,transform='Rigid')

                # if t1+ image took place immediately after cbv it can be assumed no registration is 
                # needed. 
                if self.dset['cbv']['flair']['ex']:
                    # flair_cbv_time = self.dset['cbv']['flair']['time'] - self.dset['raw']['flair']['time']
                    cbv_t1post_time = self.dset['raw']['t1+']['time'] - self.dset['cbv']['flair']['time']
                    if cbv_t1post_time > 0 and cbv_t1post_time < 600:
                        print('CBV-t1post time: {:.0f}'.format(cbv_t1post_time))
                        print('Not attempting pre-registration')
                    else:
                        print('CBV reference time uncertain, pre-registration is required')

                # for adc, just use dwi registration
                for dt in ['dwi']:
                    if self.dset['raw'][dt]['ex']:
                        moving_image = self.dset['raw'][dt]['d']
                        self.dset['raw'][dt]['d'],tx = self.register(fixed_image,moving_image,transform='Rigid')
                    if self.dset['adc']['dwi']['ex']:
                        self.dset['adc']['dwi']['d'] = self.tx(fixed_image,self.dset['adc']['dwi']['d'],tx)



            else:
                print('No T1+ to register on, skipping...')


        # bias correction.
        # self.dbias = {} # working data for calculating z-scores
        # TODO: use viewer mode designation here
        if False:
            for dt in ['t1','t1+','flair','t2','dwi']:
                if self.dset['raw'][dt]['ex']:   
                    self.dset['z'][dt]['d'] = np.copy(self.n4bias(self.dset['raw'][dt]['d']))
                    self.dset['z'][dt]['ex'] = True

            # if necessary clip any negative values introduced by the processing
            for dt in ['t1','t1+','flair','t2','dwi']:
                if self.dset['z'][dt]['ex']:
                    if np.min(self.dset['z'][dt]['d']) < 0:
                        self.dset['z'][dt]['d'][self.dset['z'][dt]['d'] < 0] = 0
                    # self.dset[dt]['d'] = self.rescale(self.dset[dt]['d'])

            # normal brain stats and z-score images
            self.normalstats()

        return

    # calculate stats to create z-score images
    # duplicates normalslice_callback code in main viewer, should be combined
    def normalstats(self,event=None):
        print('normal stats')
        # do kmeans
        # Creates a matrix of voxels for normal brain slice

        X={}
        vset = {}
        for dt2 in [('flair','t1+'),('flair','t2')]:
            if self.dset['z'][dt2[0]]['ex'] and self.dset['z'][dt2[1]]['ex']:
                region_of_support = np.where(self.dset['raw'][dt2[0]]['d']*self.dset['raw'][dt2[1]]['d'] >0)
                background = np.where(self.dset['raw'][dt2[0]]['d']*self.dset['raw'][dt2[1]]['d'] == 0)
                # vset = np.zeros_like(region_of_support,dtype='float')
                for dt in dt2:
                    vset[dt] = np.ravel(self.dset['z'][dt]['d'][region_of_support])
                # note hard-coded indexing here to match the ('flair',t12') tuple above
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

                self.dset['z'][dt]['d'] = ( self.dset['z'][dt]['d'] - self.params[dt]['mean']) / self.params[dt]['std']
                self.dset['z'][dt]['d'][background] = 0
                if False:
                    self.writenifti(self.dset['z'][dt]['d'],os.path.join(self.localstudydir,'z'+dt+'.nii'),affine=self.dset['raw'][dt]['affine'])
        plt.savefig(os.path.join(self.localcasedir,'scatterplot_normal.png'))

        return

    # tumour segmenation by nnUNet
    def segment(self,dpath=None):
        print('segment tumour')
        if dpath is None:
            dpath = os.path.join(self.localstudydir,'nnunet')
            if not os.path.exists(dpath):
                os.mkdir(dpath)
        for dt,suffix in zip(['t1+','flair'],['0000','0003']):
            if os.name == 'posix':
                l1str = 'ln -s ' + os.path.join(self.localstudydir,dt+'_processed.nii.gz') + ' '
                l1str += os.path.join(dpath,self.studytimeattrs['StudyDate']+'_'+suffix+'.nii.gz')
            elif os.name == 'nt':
                l1str = 'copy  \"' + os.path.join(self.localstudydir,dt+'_processed.nii.gz') + '\" \"'
                l1str += os.path.join(dpath,os.path.join(dpath,self.studytimeattrs['StudyDate']+'_'+suffix+'.nii.gz')) + '\"'
            os.system(l1str)


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
            if os.path.isdir(os.path.expanduser('~')+'\\anaconda3\envs\\' + self.ui.config.UIpytorch):
                envpath = os.path.expanduser('~')+'\\anaconda3\envs\\' + self.ui.config.UIpytorch
            elif os.path.isdir(os.path.expanduser('~')+'\\.conda\envs\\' + self.ui.config.UIpytorch):
                envpath = os.path.expanduser('~')+'\\.conda\envs\\' + self.ui.config.UIpytorch
            else:
                raise FileNotFoundError(self.ui.config.UIpytorch)

            command1 = '\"'+activatebatch+'\" \"' + envpath + '\"'
            command2 = 'nnUNetv2_predict -i \"' + dpath + '\" -o \"' + dpath + '\" -d137 -c 3d_fullres'
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
        self.writenifti(ET,os.path.join(self.localstudydir,'ET.nii'),affine=affine)
        self.writenifti(WT,os.path.join(self.localstudydir,'WT.nii'),affine=affine)
        if False:
            os.remove(os.path.join(dpath,sfile))

        return 


    # brain extraction from skull, currently using hd-bet
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
 
    # ants N4 bias correction
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

    # ants registration
    def register(self,img_arr_fixed,img_arr_moving,transform='Affine'):
        print('register fixed, moving')

        fixed_ants = ants.from_numpy(img_arr_fixed)
        moving_ants = ants.from_numpy(img_arr_moving)
        mytx = ants.registration(fixed=fixed_ants, moving=moving_ants, type_of_transform = transform )
        img_arr_reg = mytx['warpedmovout'].numpy()
        a=1

        return img_arr_reg,mytx['fwdtransforms']

    # apply registration transform to another volume
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

