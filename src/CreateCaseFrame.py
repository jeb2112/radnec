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
from tkinter import ttk,StringVar,DoubleVar,PhotoImage
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from cProfile import Profile
from pstats import SortKey,Stats
from enum import Enum

matplotlib.use('TkAgg')
import SimpleITK as sitk
import itk
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice

from src.NavigationBar import NavigationBar
from src.FileDialog import FileDialog
from src.CreateFrame import *


################
# Case Selection
################

class CreateCaseFrame(CreateFrame):
    def __init__(self,parent,ui=None):
        super().__init__(parent,ui=ui)

        self.fd = FileDialog(initdir=self.config.UIdatadir)
        self.datadir = StringVar()
        self.datadir.set(self.fd.dir)
        self.filenames = None
        self.casename = StringVar()
        self.casefile_prefix = None
        self.caselist = {'casetags':[],'casedirs':[]}
        self.processed = False

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
        # if normal stats is 3d then seg runs automatically, can show 'seg_raw' directly
        if self.ui.sliceviewerframe.slicevolume_norm.get() == 0:
            self.ui.dataselection = 'raw'
        else:
            self.ui.roiframe.enhancingROI_overlay_value.set(True)
            # self.ui.roiframe.enhancingROI_overlay_callback()
        self.ui.sliceviewerframe.tbar.home()
        self.ui.updateslice()
        self.ui.starttime()

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
        # 'processed' refers to earlier output and is loaded preferentially.
        # for now assuming files are either all or none processed
        elif self.casetype <= 1:
            dset = {'t1pre':{'d':None,'ex':False},'t1':{'d':None,'ex':False},'t2':{'d':None,'ex':False},
                    'flair':{'d':None,'ex':False},'ref':{'d':None,'mask':None,'ex':False}}
            files = os.listdir(self.casedir)
            # files = self.get_imagefiles(files)
            t1_files = [f for f in files if 't1' in f.lower()]
            if len(t1_files) > 0:
                if len(t1_files) > 1:
                    t1ce_file = next((f for f in t1_files if re.search('(processed)',f.lower())),None)
                    if t1ce_file is None:
                        t1ce_file = next((f for f in t1_files if re.search('(ce|gad|gd|post)',f.lower())),t1_files[0])
                elif len(t1_files) == 1:
                    t1ce_file = t1_files[0]
            flair_files = [f for f in files if 'flair' in f.lower()]
            if len(flair_files) > 0:
                if len(flair_files) > 1:
                    flair_file = next((f for f in flair_files if re.search('(processed)',f.lower())),None)
                    if flair_file is None:
                        flair_file = next((f for f in flair_files if re.search('(ce|gad|gd|post)',f.lower())),flair_files[0])
                elif len(flair_files) == 1:
                    flair_file = flair_files[0]
            t2_files = [f for f in files if 't2' in f.lower()]
            if len(t2_files) > 0:
                if len(t2_files) > 1:
                    t2_file = next((f for f in t2_files if re.search('(processed)',f.lower())),None)
                    if t2_file is None:
                        t2_file = next((f for f in t2_files if re.search('(ce|gad|gd|post)',f.lower())),t2_files[0])
                elif len(t2_files) == 1:
                    t2_file = t2_files[0]
            else:
                t2_file = None
            dset['t1']['d'],dset['t2']['d'],dset['flair']['d'],affine = self.loadData(t1ce_file,t2_file,flair_file)
            # might not need this flag anymore
            self.processed = True
            self.ui.affine['t1'] = affine

        # convenience test for presence of dataset
        for t in ['t1','t2','flair']:
            if dset[t]['d'] is not None:
                dset[t]['ex'] = True

        self.ui.sliceviewerframe.dim = np.shape(dset['t1']['d'])
        self.ui.sliceviewerframe.create_canvas()

        # 3 channels hard-coded. not currently loading t1 pre-contrast
        self.ui.data['raw'] = np.zeros((3,)+self.ui.sliceviewerframe.dim,dtype='float32')
        self.ui.data['raw'][0] = dset['t1']['d']
        self.ui.data['raw'][1] = dset['flair']['d']
        # if a t2 image is not available, fall back to using the t1 post image.
        if dset['t2']['ex']:
            self.ui.data['raw'][2] = dset['t2']['d']
        else:
            self.ui.data['raw'][2] = dset['t1']['d']

            
        # save copy of the raw data
        self.ui.data['raw_copy'] = copy.deepcopy(self.ui.data['raw'])

        # automatically run normal stats if volume selected
        if self.ui.sliceviewerframe.slicevolume_norm.get() == 1:
            self.ui.sliceviewerframe.normalslice_callback()

        # create the label. 'seg' picks up the BraTS convention but may need to be more specific
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

    def loadData(self,t1_file,t2_file,flair_file,type=None):
        img_arr_t1 = None
        img_arr_t2 = None
        img_arr_flair = None
        if 'nii' in t1_file:
            try:
                img_nb_t1 = nb.load(os.path.join(self.casedir,t1_file))
                img_nb_flair = nb.load(os.path.join(self.casedir,flair_file))
            except IOError as e:
                self.ui.set_message('Can\'t import {} or {}'.format(t1_file,flair_file))
            self.ui.nb_header = img_nb_t1.header.copy()
            # nibabel convention will be transposed to sitk convention
            img_arr_t1 = np.transpose(np.array(img_nb_t1.dataobj),axes=(2,1,0))
            img_arr_flair = np.transpose(np.array(img_nb_flair.dataobj),axes=(2,1,0))
            affine = img_nb_t1.affine
            if t2_file is not None:
                try:
                    img_nb_t2 = nb.load(os.path.join(self.casedir,t2_file))
                except IOError as e:
                    self.ui.set_message('Can\'t import {} or {}'.format(t2_file))
                img_arr_t2 = np.transpose(np.array(img_nb_t2.dataobj),axes=(2,1,0))
            else:
                img_arr_t2 = None

        elif 'dcm' in t1_file: # not finished yet
            try:
                img_dcm_t1 = pd.dcmread(os.path.join(self.casedir,t1ce_file))
                img_dcm_t2 = pd.dcmread(os.path.join(self.casedir,t2flair_file))
            except IOError as e:
                self.ui.set_message('Can\'t import {} or {}'.format(t1ce_file,t2flair_file))
            self.ui.dcm_header = None
            img_arr_t1 = np.transpose(np.array(img_dcm_t1.dataobj),axes=(2,1,0))
            img_arr_t2 = np.transpose(np.array(img_dcm_t2.dataobj),axes=(2,1,0))
            affine = None
        return img_arr_t1,img_arr_t2,img_arr_flair,affine
    

    #################
    # process methods
    #################


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
            if False:   
                info = subprocess.STARTUPINFO()
                info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                info.wShowWindow = subprocess.SW_HIDE
                res = subprocess.run(cstr,shell=True,startupinfo=info,creationflags=subprocess.CREATE_NO_WINDOW)
                # res = subprocess.run(cstr,shell=True)
            else:
                popen = subprocess.Popen(cstr,shell=True,stdout=subprocess.PIPE,universal_newlines=True)
                for stdout_line in iter(popen.stdout.readline,""):
                    if stdout_line != '\n':
                        print(stdout_line)
                popen.stdout.close()
                res = popen.wait()
                if res:
                    raise subprocess.CalledProcessError(res,cstr)
                    print(res)
        # print(res)

        img_nb = nb.load(os.path.join(self.casedir,'img_brain.nii'))
        img_arr = np.transpose(np.array(img_nb.dataobj),axes=(2,1,0))
        img_nb = nb.load(os.path.join(self.casedir,'img_mask.nii'))
        img_arr_mask = np.transpose(np.array(img_nb.dataobj),axes=(2,1,0))
        os.remove(os.path.join(self.casedir,'img_brain.nii'))
        os.remove(os.path.join(self.casedir,'img_temp.nii'))
        os.remove(os.path.join(self.casedir,'img_mask.nii'))
        return img_arr,img_arr_mask

    # clip outliers as in brainmage code
    def brainmage_clip(self,img):
        img_temp = img[np.where(img>0)]
        img_temp = img[img >= img_temp.mean()]
        p1 = np.percentile(img_temp, 1)
        p2 = np.percentile(img_temp, 99)
        img[img > p2] = p2
        img = (img - p1) / p2
        return img.astype(np.float32)

                
    # if T2 matrix is different resample it to t1
    def resamplet2(self,arr_t1,arr_t2,a1,a2):
        img_arr_t1 = copy.deepcopy(arr_t1)
        img_arr_t2 = copy.deepcopy(arr_t2)
        img_t1 = nb.Nifti1Image(np.transpose(img_arr_t1,axes=(2,1,0)),affine=a1)
        img_t2 = nb.Nifti1Image(np.transpose(img_arr_t2,axes=(2,1,0)),affine=a2)
        img_t2_res = resample_from_to(img_t2,(img_t1.shape[:3],img_t1.affine))
        img_arr_t2 = np.ascontiguousarray(np.transpose(np.array(img_t2_res.dataobj),axes=(2,1,0)))
        return img_arr_t2,img_t2_res.affine
    
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
    
    def n4bias(self,img_arr,shrinkFactor=4,nFittingLevels=4):
        print('N4 bias correction')
        data = copy.deepcopy(img_arr)
        dataImage = sitk.Cast(sitk.GetImageFromArray(data),sitk.sitkFloat32)
        sdataImage = sitk.Shrink(dataImage,[shrinkFactor]*dataImage.GetDimension())
        maskImage = sitk.Cast(sitk.GetImageFromArray(np.where(data,True,False).astype('uint8')),sitk.sitkUInt8)
        maskImage = sitk.Shrink(maskImage,[shrinkFactor]*maskImage.GetDimension())
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        lowres_img = corrector.Execute(sdataImage,maskImage)
        log_bias_field = corrector.GetLogBiasFieldAsImage(dataImage)
        log_bias_field_arr = sitk.GetArrayFromImage(log_bias_field)
        corrected_img = dataImage / sitk.Exp(log_bias_field)
        corrected_img_arr = sitk.GetArrayFromImage(corrected_img)
        img_arr = copy.deepcopy(corrected_img_arr)
        return img_arr


    # main callback for selecting a data directory either by file dialog or text entry
    # find the list of cases in the current directory, set the combobox, and optionally load a case
    def datadirentry_callback(self,event=None):
        dir = self.datadir.get().strip()
        if os.path.exists(dir):
            self.w.config(state='normal')            
            files = os.listdir(dir)
            casefiles = []
            if len(files):

                imagefiles = self.get_imagefiles(files)

                # single case directory with image files
                if len(imagefiles) > 1:
                    imagefiles = [i.group(1) for i in imagefiles]
                    self.casefile_prefix = ''
                    casefiles = [os.path.split(dir)[1]]
                    self.caselist['casetags'] = casefiles
                    self.caselist['casedirs'] = [os.path.split(dir)[1]]
                    self.ui.set_message('')
                    self.w.config(width=min(20,len(casefiles[0])))
                    self.datadir.set(os.path.split(dir)[0])
                    self.casetype = 0
                    doload = True

                # one or more case subdirectories
                else:
                    niftidirs,dcmdirs = self.get_imagedirs(dir)
                    # niftidirs option is intended for processing cases from a parent directory. such as BraTS.
                    # imagefiles and dcmdirs intended for processing at the level of the individual case directory.
                    if len(niftidirs):
                        self.datadir.set(dir)

                        # check for BraTS format first
                        brats = re.match('(^.*brats.*)0[0-9]{4}',niftidirs[0],flags=re.I)
                        if brats:
                            self.casefile_prefix = brats.group(1)
                            casefiles = [re.match('.*(0[0-9]{4})',f).group(1) for f in files if re.search('_0[0-9]{4}$',f)]
                            self.caselist['casetags'] = [re.match('.*(0[0-9]{4})',f).group(1) for f in files if re.search('_0[0-9]{4}$',f)]
                            self.caselist['casedirs'] = files
                            self.w.config(width=6)
                            # brats data already have this processing
                            self.register_check_value.set(0)
                            self.skullstrip_check_value.set(0)
                        else:
                            self.casefile_prefix = ''
                            # for niftidirs that are processed dicomdirs, there may be
                            # multiple empty subdirectories. for now assume that the 
                            # immediate subdir of the datadir is the best tag for the casefile
                            # if there are multiple niftidirs, and the selected datadir itself is the
                            # tag for a single nifti file
                            # the casedir is a sub-directory path between the upper datadir,
                            # and the parent of the dicom series dirs where the nifti's get stored
                            if len(niftidirs) > 1:
                                casefiles = [re.split(r'/|\\',d[len(self.datadir.get())+1:])[0] for d in niftidirs]
                                casedirs = [d[len(self.datadir.get())+1:] for d in niftidirs]
                                doload = self.config.AutoLoad
                            elif len(niftidirs) == 1:
                                self.casedir = niftidirs[0]
                                self.datadir.set(os.path.split(dir)[0])
                                casefiles = [re.split(r'/|\\',d[len(self.datadir.get())+1:])[0] for d in niftidirs]
                                casedirs = [d[len(self.datadir.get())+1:] for d in niftidirs]
                                doload = True
                            # may need a future sort
                            if False:
                                casefiles,casedirs = (list(t) for t in zip(*sorted(zip(casefiles,casedirs))))
                            self.caselist['casetags'] = casefiles
                            self.caselist['casedirs'] = casedirs
                            self.w.config(width=max(20,len(casefiles[0])))
                        self.casetype = 1

                    # assumes all nifti dirs or all dicom dirs.
                    # if only a single dicom case directory continue directly to blast
                    elif len(dcmdirs)==1:
                        # self.datadir.set(dir)
                        self.casefile_prefix = ''
                        self.casedir = dcmdirs[0]
                        self.datadir.set(os.path.split(dir)[0])
                        self.caselist['casetags'] = [re.split(r'/|\\',d[len(self.datadir.get())+1:])[0] for d in dcmdirs]
                        # self.caselist['casetags'] = [os.path.split(self.casedir)[1]]
                        self.caselist['casedirs'] = [d[len(self.datadir.get())+1:] for d in dcmdirs]
                        self.w.config(width=max(20,len(self.caselist['casetags'][0])))
                        self.casetype = 2
                        doload = True
                    elif len(dcmdirs) > 1:
                    # if multiple dicom dirs, preprocess only
                        self.datadir.set(dir)
                        self.preprocess(dcmdirs)
                        casefiles = []
                        doload = False
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
    

    def resetCase(self):
        self.filenames = None
        self.casename = StringVar()
        self.casefile_prefix = None
        self.caselist['casetags'] = []
        self.caselist['casedirs'] = []
