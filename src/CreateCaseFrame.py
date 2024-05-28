import os,sys
import numpy as np
import glob
import copy
import re
import logging
import pickle
import tkinter as tk
import nibabel as nb
from nibabel.processing import resample_from_to
import pydicom as pd
from tkinter import ttk,StringVar,DoubleVar,PhotoImage
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import matplotlib
import matplotlib.pyplot as plt
from cProfile import Profile
from pstats import SortKey,Stats
from enum import Enum

matplotlib.use('TkAgg')
import SimpleITK as sitk
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice

from src.NavigationBar import NavigationBar
from src.FileDialog import FileDialog
from src.CreateFrame import *
from src.DcmCase import Case,NiftiStudy


################
# Case Selection
################

class CreateCaseFrame(CreateFrame):
    def __init__(self,parent,ui=None):
        super().__init__(parent,ui=ui)

        self.fd = FileDialog(initdir=self.config.UIdatadir)
        self.datadir = StringVar() # parent of a directory containing case directories
        self.datadir.set(self.fd.dir)
        self.filenames = None
        self.casename = StringVar()
        self.casefile_prefix = None
        self.casedir_prefix = 'M' # simple convention to identify root dir of a case
        self.caselist = {'casetags':[],'casedirs':[]} # list of case directories in self.datadir, and short-form tags
        self.processed = False
        self.pp = None

        # case selection
        self.frame.grid(row=0,column=0,columnspan=3,sticky='ew')

        # select directory
        self.fdbicon = PhotoImage(file=os.path.join(self.config.UIResourcesPath,'folder_icon_16.png'))
        # select a parent dir for a group of case sub-dirs
        self.fdbutton = ttk.Button(self.frame,image=self.fdbicon, command=self.select_dir)
        self.fdbutton.grid(row=0,column=2)
        # select file
        self.fdbicon_file = PhotoImage(file=os.path.join(self.config.UIResourcesPath,'file_icon_16.png'))
        self.fdbutton_file = ttk.Button(self.frame,image=self.fdbicon_file,command = self.datafileentry_callback)
        self.fdbutton_file.grid(row=0,column=3)

        self.datadirentry = ttk.Entry(self.frame,width=40,textvariable=self.datadir)
        # event currently a dummy arg since not being used in datadirentry_callback
        self.datadirentry.bind('<Return>',lambda event=None:self.datadirentry_callback(event=event))
        self.datadirentry.grid(row=0,column=4,columnspan=5)
        caselabel = ttk.Label(self.frame, text='Case: ')
        caselabel.grid(column=0,row=0,sticky='we')
        # self.casename.trace_add('write',self.case_callback)
        self.w = ttk.Combobox(self.frame,width=8,textvariable=self.casename,values=self.caselist['casetags'])
        self.w.grid(column=1,row=0)
        self.w.bind("<<ComboboxSelected>>",self.case_callback)


    # callback for file dialog 
    def select_dir(self):
        self.resetCase()
        self.fd.select_dir()
        self.datadir.set(self.fd.dir)
        self.datadirentry.update()
        self.datadirentry_callback()

    # callback for loading by individual files
    def datafileentry_callback(self):
    # def select_file(self):
        self.resetCase()
        self.fd.select_file()
        if len(self.fd.filenames) != 3:
            self.ui.set_message('Three files must be selected')
            return
        self.ui.set_message('')
        self.casedir = os.path.split(self.fd.filenames[0])[0]
        self.caselist['casedirs'] = [os.path.split(self.casedir)[1]]
        self.datadir.set(os.path.split(self.casedir)[0])
        self.filenames = [os.path.split(f)[1] for f in self.fd.filenames]
        # sort assumes minimal tags t1,t2 are present flair image is the 3rd.
        self.filenames = sorted(self.filenames,key=lambda x:(x.lower().find('t1'),x.lower().find('t2')),reverse=True)
        # self.datafileentry_callback()
        # for indiviudally selected files, datadir is the parent and intermediate 'casedirs' not used
        self.caselist['casetags'] = os.path.split(self.datadir.get())[1]
        self.casefile_prefix = ''
        self.config.UIdataroot = self.casefile_prefix
        self.w['values'] = self.caselist['casetags']
        self.w.current(0)
        self.w.config(width=min(20,len(self.caselist['casetags'])))
        self.casename.set(self.caselist['casetags'])
        # self.datadir.set(os.path.split(dir)[0])
        self.casetype = 0
        self.case_callback(files=self.filenames)
        return

    def case_callback(self,casevar=None,val=None,event=None,files=None):
        case = self.casename.get()
        self.ui.set_casename(val=case)
        print('Loading case {}'.format(case))
        self.loadCase(files=files)
        self.ui.chselection = 't1+'
        self.ui.dataselection = 'raw'
        self.ui.sliceviewerframe.tbar.home()
        self.ui.updateslice()

    def loadCase(self,case=None,files=None):

        # reset and reinitialize
        self.ui.resetUI()
        if case is not None:
            self.casename.set(case)
            self.ui.set_casename()
        caseindex = self.caselist['casetags'].index(self.casename.get())
        # self.casedir = os.path.join(self.datadir.get(),self.config.UIdataroot+self.casename.get())
        if len(self.caselist['casedirs']):
            self.casedir = os.path.join(self.datadir.get(),self.caselist['casedirs'][caseindex])
        else:
            raise ValueError('No cases to load')

        # if three image files are given load them directly
        # mignt not need this anymore
        if files is not None:
            if len(files) != 3:
                self.ui.set_message('Select three image files')
                return
            t1ce_file,t2_file,flair_file = self.filenames
            dset = {'t1pre':{'d':None,'ex':False},'t1':{'d':None,'ex':False},'t2':{'d':None,'ex':False},
                    'flair':{'d':None,'ex':False},'ref':{'d':None,'mask':None,'ex':False}}
            dset['t1']['d'],dset['t2']['d'],dset['flair']['d'],affine = self.loadData(t1ce_file,t2_file,flair_file)
            # might not need this flag anymore
            self.processed = True
            self.ui.affine['t1'] = affine

        # check for nifti image files with matching filenames
        elif self.casetype <= 1:
            dset = {'t1pre':{'d':None,'ex':False},'t1':{'d':None,'ex':False},'t2':{'d':None,'ex':False},
                    'flair':{'d':None,'ex':False},'ref':{'d':None,'mask':None,'ex':False}}
            studies = [f for f in os.listdir(self.casedir) if os.path.isdir(os.path.join(self.casedir,f)) ]
            # by convention, study dir name is the date of the study
            for i,sname in enumerate(studies):
                # won't use date string as a key
                # self.ui.timepoints = studies
                self.ui.data[i] = NiftiStudy(self.casename.get(),os.path.join(self.casedir,sname))
                self.ui.data[i].loaddata()
                self.ui.data[i].date = sname

        # update sliceviewers according to data loaded
        for s in self.ui.sliceviewerframes.values():
            for dt in self.ui.data[0].channels.values():
                if (self.ui.data[0].dset['raw'][dt]['ex'] and self.ui.data[1].dset['raw'][dt]['ex']):
                    s.chdisplay_button[dt]['state'] = 'normal'
                # else:
                #     s.chdisplay_button[dt]['state'] = 'normal'
            s.dim = np.shape(self.ui.data[0].dset['raw']['t1+']['d'])
            s.level = np.array([self.ui.data[0].dset['raw']['t1+']['max']/4]*2)
            s.window = np.array([self.ui.data[0].dset['raw']['t1+']['max']/2]*2)
            s.create_canvas()
        if False: # cbv will have to display just one overlay if necessary
            # update roiframe according to data loaded
            for dt in ['cbv']:
                if not(self.ui.data[0].dset[dt]['ex'] and self.ui.data[1].dset[dt]['ex']):
                    self.ui.roiframe.overlay_type_button[dt]['state'] = 'disabled'
                else:
                    self.ui.roiframe.overlay_type_button[dt]['state'] = 'normal'
    

        # create the label. 'seg' picks up the BraTS convention but may need to be more specific
        if False:
            if self.casetype <= 1:
                seg_file = next((f for f in files if 'seg' in f),None)
                if seg_file is not None and 'blast' not in seg_file:
                    label = sitk.ReadImage(os.path.join(self.casedir,seg_file))
                    img_arr = sitk.GetArrayFromImage(label)
                    self.ui.data['label'] = img_arr
                else:
                    self.ui.data['label'] = None
            else:
                self.ui.data['label'] = None

            # supplementary labels. brats and nnunet conventions are differnt.
            if self.ui.data['label'] is not None:
                if False: # nnunet
                    self.ui.data['manual_ET'] = (self.ui.data['label'] == 3).astype('int') #enhancing tumor 
                    self.ui.data['manual_TC'] = (self.ui.data['label'] >= 2).astype('int') #tumour core
                    self.ui.data['manual_WT'] = (self.ui.data['label'] >= 1).astype('int') #whole tumour
                else: # brats
                    self.ui.data['manual_ET'] = (self.ui.data['label'] == 4).astype('int') #enhancing tumor 
                    self.ui.data['manual_TC'] = ((self.ui.data['label'] == 1) | (self.ui.data['label'] == 4)).astype('int') #tumour core
                    self.ui.data['manual_WT'] = (self.ui.data['label'] >= 1).astype('int') #whole tumour

    # probably don't need this anymore
    def loadData(self,dt_file,type=None):
        img_arr = None
        if 'nii' in dt_file:
            try:
                img_nb = nb.load(os.path.join(self.casedir,dt_file))
            except IOError as e:
                self.ui.set_message('Can\'t import {} or {}'.format(dt_file))
            nb_header = img_nb.header.copy()
            # nibabel convention will be transposed to sitk convention
            img_arr = np.transpose(np.array(img_nb.dataobj),axes=(2,1,0))
            affine = img_nb.affine

        return img_arr,affine
    

    #################
    # process methods
    #################


   
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
    


    # main callback for selecting a data directory either by file dialog or text entry
    # find the list of cases in the current directory, set the combobox, and optionally load a case
    def datadirentry_callback(self,event=None):
        dir = self.datadir.get().strip()
        if os.path.exists(dir):
            self.w.config(state='normal')            
            files = os.listdir(dir)
            casefiles = []
            if len(files):

                niftidirs,dcmdirs = self.get_imagedirs(dir)
                # niftidirs option could be a single case, or a directory of multiple cases 
                if len(niftidirs):
                    self.datadir.set(dir)

                    self.casefile_prefix = ''
                    # for niftidirs that are processed dicomdirs, there may be
                    # multiple empty subdirectories. for now assume that the 
                    # immediate subdir of the datadir is the best tag for the casefile
                    # if there are multiple niftidirs, and the selected datadir itself is the
                    # tag for a single nifti file
                    # the casedir is a sub-directory path between the upper datadir,
                    # and the parent of the dicom series dirs where the nifti's get stored
                    if len(niftidirs) > 1:
                        niftidirs = self.group_dcmdirs(niftidirs)
                        # casefiles = [re.split(r'/|\\\\',d[len(self.datadir.get())+1:])[0] for d in niftidirs]
                        casedirs = [k for k in niftidirs.keys()]
                        # if a single nifti case dir at the level  of the case dir and not the parent dir,
                        # need to adjust datadir to be the parent directory
                        if len(casedirs) == 1:
                            datadir = self.datadir.get()
                            if casedirs[0] in datadir:
                                self.datadir.set(os.path.split(datadir)[0])
                        doload = self.config.AutoLoad
                    elif len(niftidirs) == 1:
                        raise ValueError('Only one image directory found for this case')
                    # may need a future sort
                    if False:
                        casefiles,casedirs = (list(t) for t in zip(*sorted(zip(casefiles,casedirs))))
                    self.caselist['casetags'] = casedirs
                    self.caselist['casedirs'] = casedirs
                    self.w.config(width=max(20,len(casedirs[0])))
                    self.casetype = 1

                # if dicom dirs, then preprocess only. could be a single case or multiple cases
                elif len(dcmdirs):
                    dcmdirs = self.group_dcmdirs(dcmdirs)
                    for c in dcmdirs.keys():
                        case = Case(c,dcmdirs[c],self.config)
                    return

            if len(self.caselist['casetags']):
                self.config.UIdataroot = self.casefile_prefix
                # TODO: will need a better sort here
                self.caselist['casetags'] = sorted(self.caselist['casetags'])
                self.w['values'] = self.caselist['casetags']
                self.w.current(0)
                # current(0) should do this too, but sometimes it does not
                self.casename.set(self.caselist['casetags'][0])
                # autoload first case
                if doload:
                    self.case_callback()
            else:
                print('No cases found in directory {}'.format(dir))
                self.ui.set_message('No cases found in directory {}'.format(dir))
        else:
            print('Directory {} not found.'.format(dir))
            self.ui.set_message('Directory {} not found.'.format(dir))
            self.w.config(state='disable')
            self.datadirentry.update()
        return

    def get_imagefiles(self,files):
        imagefiles = [re.match('(^.*(t1|t2|flair).*\.(nii|nii\.gz|dcm)$)',f.lower()) for f in files]
        imagefiles = list(filter(lambda item: item is not None,imagefiles))
        if len(imagefiles):
            self.ui.set_message('')
        return imagefiles
    
    # get list of all image directories under the selected directory
    # in the case of dcmdirs, it could be one or more cases
    # in the case of niftidirs, it should just be one case
    def get_imagedirs(self,dir):
        # dir = self.datadir.get()
        dcmdirs = []
        niftidirs = []
        for root,dirs,files in os.walk(dir,topdown=True):
            if len(files):
                dcmfiles = [f for f in files if re.match('.*\.dcm',f.lower())]
                niftifiles = [f for f in files if re.match('.*(t1|t2|flair).*\.(nii|nii\.gz)',f.lower())]
                if len(dcmfiles):
                    # for now assume that the parent of this dir is a series dir, and will take 
                    # the dicomdir as the parent of the series dir
                    # but for exported sunnybrook dicoms at least the more recognizeable dir might 
                    # be two levels above at the date.
                    dcmdirs.append(os.path.split(root)[0])
                if len(niftifiles):
                    niftidirs.append(os.path.join(root))
        if len(niftidirs+dcmdirs):
            self.ui.set_message('')
            # self.datadir.set(dir)
        # due to the intermediate seriesdirs, the above walk generates duplicates
        dcmdirs = list(set(dcmdirs))
        # for nifti dirs, need to set the casefiles for the pulldown at one of the more
        # recognizeable subdirs of the datadir. 

        return niftidirs,dcmdirs
    
    # further group dcmdirs into separate cases
    # the download directory structure isn't specified
    # for now, assume that if multiple dcmdirs are present
    # in selected dir, and the root dir of those dmcdirs is prefixed with a certain string,
    # then they can be further grouped as case dirs
    def group_dcmdirs(self,dcmdirs):
        dcm_casedirs = {}
        casedirs = []
        for i,d in enumerate(dcmdirs):
            casedirs.append([s for s in re.split('\/|\\\\',d) if s.startswith(self.casedir_prefix)][0])
        if len(casedirs) == len(dcmdirs):
            casedir_keys = set(casedirs)
            dcm_casedirs = {c:[] for c in casedir_keys}
            for r,d in zip(casedirs,dcmdirs):
                dcm_casedirs[r].append(d)
            return dcm_casedirs

        else:
            raise ValueError('Not all directories match a case prefix')



    

    def resetCase(self):
        self.filenames = None
        self.casename = StringVar()
        self.casefile_prefix = None
        self.caselist['casetags'] = []
        self.caselist['casedirs'] = []
