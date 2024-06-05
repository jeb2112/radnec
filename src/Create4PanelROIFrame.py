import os,sys
import re
import numpy as np
import pickle
import copy
import logging
import time
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import _Mode
matplotlib.use('TkAgg')
import SimpleITK as sitk
import nibabel as nb
from skimage.morphology import disk,square,binary_dilation,binary_closing,flood_fill,ball,cube,reconstruction
from skimage.measure import find_contours
from scipy.spatial.distance import dice
from scipy.ndimage import binary_closing as scipy_binary_closing
from scipy.io import savemat

from src.OverlayPlots import *
from src.CreateFrame import CreateFrame,Command
from src.ROI import ROI

# contains various ROI methods and variables for 'overlay' mode
class Create4PanelROIFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)

        ########################
        # layout for the buttons
        ########################

        # dummy frame to hide
        self.dummy_frame = ttk.Frame(self.parentframe,padding='0')
        self.dummy_frame.grid(row=2,column=4,rowspan=5,sticky='news')

        # actual frame
        self.frame.grid(row=2,column=4,rowspan=5,sticky='ne')

        return


    #############
    # ROI methods
    ############# 
        
    # main callback for handling the overlays
    def overlay_callback(self,updateslice=True,wl=False,redo=False):

        if self.overlay_value.get() == True:
            ovly = self.overlay_type.get()
            self.sliderframe[ovly].lift()
            self.sliderframe['dummy'].lower(self.sliderframe[ovly])
            ovly_str = ovly + 'overlay'
            ch = self.ui.sliceviewerframe.chdisplay.get()
            usemask = self.mask_value.get()
            if ovly == 'tempo': # probably want an attribute for this
                colormap = 'tempo'
            else:
                colormap = 'viridis'

            # needed here?
            # self.ui.sliceviewerframe.updatewl_fusion()

            # generate a new overlay
            for s in self.ui.data:
                if not self.ui.data[s].dset[ovly_str][ch]['ex'] or redo:

                    # eg for cbv there might only be one DSC study, so just
                    # use the base grayscale for a dummy overlay image
                    # in the other study, and set the 'ex' False so the 
                    # colormap can later be assigned 'gray'
                    if not self.ui.data[s].dset[ovly][ch]['ex']:
                        self.ui.data[s].dset[ovly_str][ch]['d'] = np.copy(self.ui.data[s].dset['raw'][ch]['d'])
                        self.ui.data[s].dset[ovly_str][ch]['ex'] = False
                    else:
                        if usemask:
                            mask = self.ui.data[s].mask[self.mask_type.get()]['d']
                        else:
                            mask = None

                        self.ui.data[s].dset[ovly_str][ch]['d'] = generate_overlay(
                            self.ui.data[s].dset['raw'][ch]['d'],
                            self.ui.data[s].dset[ovly][ch]['d'],
                            mask,
                            image_wl = [self.ui.sliceviewerframe.window[0],self.ui.sliceviewerframe.level[0]],
                            overlay_wl = self.ui.sliceviewerframe.wl[ovly],
                            overlay_intensity=self.config.OverlayIntensity,
                            colormap = colormap)
                        self.ui.data[s].dset[ovly_str][ch]['ex'] = True

            self.ui.dataselection = ovly_str

        else:
            self.sliderframe['dummy'].lift()
            self.ui.set_dataselection('raw')

        if updateslice:
            self.ui.updateslice()


    def updateslider(self,layer,slider):
        self.overlay_callback(wl=True,redo=True)

    def updatesliderlabel(self,layer,slider):
        if layer == 'cbv':
            pfstr = '{:.0f}'
        else:
            pfstr = '{:.1f}'
        sval_min = self.sliders[layer]['min'].get()
        sval_max = self.sliders[layer]['max'].get()
        sval = np.round(self.sliders[layer][slider].get() / self.thresholds[layer]['inc']) * self.thresholds[layer]['inc']
        try:
            self.sliderlabels[layer][slider]['text'] = pfstr.format(sval)
            self.ui.sliceviewerframe.wl[layer] = [sval_max-sval_min,(sval_max-sval_min)/2+sval_min]
        except KeyError as e:
            print(e)
       
    
    def resetROI(self):
        return
    
    def resetCursor(self,event=None):
        self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='watch')
        self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()
    
    # not recently updated.
    # for now output only segmentations so uint8
    def WriteImage(self,img_arr,filename,header=None,norm=False,type='uint8',affine=None):
        img_arr_cp = copy.deepcopy(img_arr)
        if norm:
            img_arr_cp = (img_arr_cp -np.min(img_arr_cp)) / (np.max(img_arr_cp)-np.min(img_arr_cp)) * norm
        # using nibabel nifti coordinates
        if True:
            img_nb = nb.Nifti1Image(np.transpose(img_arr_cp.astype(type),(2,1,0)),affine,header=header)
            nb.save(img_nb,filename)
        # couldn't get sitk nifti coordinates to work in mricron viewer
        else:
            img = sitk.GetImageFromArray(img_arr_cp.astype('uint8'))
            # this fails to copy origin, so do it manually
            # img.CopyInformation(self.ui.t1ce)
            img.SetDirection(self.ui.direction)
            img.SetSpacing(self.ui.spacing)
            img.SetOrigin(self.ui.origin)
            writer = sitk.ImageFileWriter()
            writer.SetImageIO('NiftiImageIO')
            writer.SetFileName(filename)
            writer.Execute(img)
        return
        

