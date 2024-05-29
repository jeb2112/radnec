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

# contains various ROI methods and variables
class CreateOverlayFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)

        self.buttonpress_id = None # temp var for keeping track of button press event
        self.finalROI_overlay_value = tk.BooleanVar(value=False)
        self.overlay_value = tk.BooleanVar(value=False)
        self.mask_value = tk.BooleanVar(value=False)
        roidict = {'ET':{'t12':None,'flair':None,'bc':None},'T2 hyper':{'t12':None,'flair':None,'bc':None}}
        self.overlay_type = tk.StringVar(value=self.config.OverlayType)
        self.mask_type = tk.StringVar(value=self.config.MaskType)
        self.layerlist = {'blast':['ET','T2 hyper'],'seg':['ET','TC','WT','all']}
        self.layer = tk.StringVar(value='ET')
        self.layerROI = tk.StringVar(value='ET')
        self.layertype = tk.StringVar(value='blast')
        self.currentroi = tk.IntVar(value=0)
        self.roilist = []

        roidict = {'z':{'min':None,'max':None,'minmax':None},'cbv':{'min':None,'max':None,'minmax':None}}
        self.thresholds = copy.deepcopy(roidict)
        self.sliders = copy.deepcopy(roidict)
        self.sliderlabels = copy.deepcopy(roidict)
        self.sliderframe = {}
        self.thresholds['z']['min'] = tk.DoubleVar(value=self.ui.config.zmin)
        self.thresholds['z']['max'] = tk.DoubleVar(value=self.ui.config.zmax)
        self.thresholds['z']['inc'] = self.ui.config.zinc
        self.thresholds['cbv']['min'] = tk.IntVar(value=self.ui.config.cbvmin)
        self.thresholds['cbv']['max'] = tk.IntVar(value=self.ui.config.cbvmax)
        self.thresholds['cbv']['inc'] = self.ui.config.cbvinc


        ########################
        # layout for the buttons
        ########################

        # dummy frame to hide
        self.dummy_frame = ttk.Frame(self.parentframe,padding='0')
        self.dummy_frame.grid(row=2,column=4,rowspan=5,sticky='news')

        # actual frame
        self.frame.grid(row=2,column=4,rowspan=5,sticky='ne')

        # overlay type
        overlay_type_label = ttk.Label(self.frame, text='overlay type: ')
        overlay_type_label.grid(row=0,column=2,padx=(50,0),sticky='e')
        self.overlay_type_button = {}
        self.overlay_type_button['z'] = ttk.Radiobutton(self.frame,text='z-score',variable=self.overlay_type,value='z',
                                                    command=Command(self.overlay_callback))
        self.overlay_type_button['z'].grid(row=0,column=3,sticky='w')
        self.overlay_type_button['cbv'] = ttk.Radiobutton(self.frame,text='CBV',variable=self.overlay_type,value='cbv',
                                                    command=Command(self.overlay_callback))
        self.overlay_type_button['cbv'].grid(row=0,column=4,sticky='w')
        self.overlay_type_button['tempo'] = ttk.Radiobutton(self.frame,text='TEMPO',variable=self.overlay_type,value='tempo',
                                                    command=Command(self.overlay_callback))
        self.overlay_type_button['tempo'].grid(row=0,column=5,sticky='w')

        # on/off button
        overlay_label = ttk.Label(self.frame,text='overlay on/off')
        overlay_label.grid(row=0,column=0,sticky='e')
        overlay_button = ttk.Checkbutton(self.frame,text='',
                                               variable=self.overlay_value,
                                               command=self.overlay_callback)
        overlay_button.grid(row=0,column=1,sticky='w')

        # layer mask
        mask_type_label = ttk.Label(self.frame, text='mask: ')
        mask_type_label.grid(row=1,column=2,padx=(50,0),sticky='e')
        self.mask_type_button = {}
        self.mask_type_button['ET'] = ttk.Radiobutton(self.frame,text='ET',variable=self.mask_type,value='ET',
                                                    command=Command(self.overlay_callback))
        self.mask_type_button['ET'].grid(row=1,column=3,sticky='w')
        self.mask_type_button['WT'] = ttk.Radiobutton(self.frame,text='WT',variable=self.mask_type,value='WT',
                                                    command=Command(self.overlay_callback))
        self.mask_type_button['WT'].grid(row=1,column=4,sticky='w')

        # on/off button
        mask_label = ttk.Label(self.frame,text='mask on/off')
        mask_label.grid(row=1,column=0,sticky='e')
        mask_button = ttk.Checkbutton(self.frame,text='',
                                               variable=self.mask_value,
                                               command=self.overlay_callback)
        mask_button.grid(row=1,column=1,sticky='w')


        ########################
        # layout for the sliders
        ########################
        self.sliderframe['dummy'] = ttk.Frame(self.frame,padding=0)
        self.sliderframe['dummy'].grid(column=0,row=2,columnspan=6,sticky='news')
        self.sliderframe['tempo'] = ttk.Frame(self.frame,padding=0)
        self.sliderframe['tempo'].grid(column=0,row=2,columnspan=6,sticky='news')

        self.sliderframe['z'] = ttk.Frame(self.frame,padding='0')
        self.sliderframe['z'].grid(column=0,row=2,columnspan=6,sticky='e')
        self.sliderframe['z'].lower()

        # z-score sliders
        zlabel = ttk.Label(self.sliderframe['z'], text='z-score (min,max)')
        zlabel.grid(column=0,row=0,sticky='e')

        self.sliders['z']['min'] = ttk.Scale(self.sliderframe['z'],from_=self.ui.config.zmin,to=self.ui.config.zmax/2,variable=self.thresholds['z']['min'],state='normal',
                                  length='1i',command=Command(self.updatesliderlabel,'z','min'),orient='horizontal')
        self.sliders['z']['min'].grid(row=0,column=1,sticky='e')
        self.sliderlabels['z']['min'] = ttk.Label(self.sliderframe['z'],text=self.thresholds['z']['min'].get())
        self.sliderlabels['z']['min'].grid(row=0,column=2,sticky='e')

        self.sliders['z']['max'] = ttk.Scale(self.sliderframe['z'],from_=self.ui.config.zmax/2,to=self.ui.config.zmax,variable=self.thresholds['z']['max'],state='normal',
                                  length='1i',command=Command(self.updatesliderlabel,'z','max'),orient='horizontal')
        self.sliders['z']['max'].grid(row=0,column=3,sticky='e')
        self.sliderlabels['z']['max'] = ttk.Label(self.sliderframe['z'],text=self.thresholds['z']['max'].get())
        self.sliderlabels['z']['max'].grid(row=0,column=4,sticky='e')

        # combo slider not available in tkinter, this one breaks tkinter look
        # self.sliders['z']['minmax'] = RangeSliderH(self.zsliderframe,[self.thresholds['z']['min'],self.thresholds['z']['max']],
        #                                            max_val = self.thresholds['z']['max'].get(),padX=50)
        # self.sliders['z']['minmax'].grid(row=0,column=1,sticky='e')


        # cbv sliders. resolution hard-coded
        self.sliderframe['cbv'] = ttk.Frame(self.frame,padding='0')
        self.sliderframe['cbv'].grid(column=0,row=2,columnspan=6,sticky='e')
        self.sliderframe['cbv'].lower()

        cbvlabel = ttk.Label(self.sliderframe['cbv'], text='CBV (min,max)')
        cbvlabel.grid(column=0,row=0,sticky='e')

        self.sliders['cbv']['min'] = ttk.Scale(self.sliderframe['cbv'],from_=self.ui.config.cbvmin,to=self.ui.config.cbvmax/2,variable=self.thresholds['cbv']['min'],state='normal',
                                  length='1i',command=Command(self.updatesliderlabel,'cbv','min'),orient='horizontal')
        self.sliders['cbv']['min'].grid(row=0,column=1,sticky='e')
        self.sliderlabels['cbv']['min'] = ttk.Label(self.sliderframe['cbv'],text=self.thresholds['cbv']['min'].get())
        self.sliderlabels['cbv']['min'].grid(row=0,column=2,sticky='e')

        self.sliders['cbv']['max'] = ttk.Scale(self.sliderframe['cbv'],from_=self.ui.config.cbvmax/2,to=self.ui.config.cbvmax,variable=self.thresholds['cbv']['max'],state='normal',
                                  length='1i',command=Command(self.updatesliderlabel,'cbv','max'),orient='horizontal')
        self.sliders['cbv']['max'].grid(row=0,column=3,sticky='e')
        self.sliderlabels['cbv']['max'] = ttk.Label(self.sliderframe['cbv'],text=self.thresholds['cbv']['max'].get())
        self.sliderlabels['cbv']['max'].grid(row=0,column=4,sticky='e')
        self.sliderframe['dummy'].lift()

        for k in self.sliders.keys():
            for m in ['min','max']:
                self.sliders[k][m].bind("<ButtonRelease-1>",Command(self.updateslider,k,m))

    # use lift/lower instead
    def slider_state(self,s=None):
        # if s is None:
        s = self.overlay_type.get()
        for m in ['min','max']:
            for k in self.sliders.keys():
                if k == s:
                    self.sliders[k][m]['state']='normal'
                else:
                    self.sliders[k][m]['state'] = 'disabled'

        return


    #############
    # ROI methods
    ############# 
        
    def overlay_callback(self,updateslice=True,wl=False):

        if self.overlay_value.get() == True:
            ovly = self.overlay_type.get()
            self.sliderframe[ovly].lift()
            self.sliderframe['dummy'].lower(self.sliderframe[ovly])
            ovly_str = ovly + 'overlay'
            ch = self.ui.sliceviewerframe.chdisplay.get()
            mask = self.ui.sliceviewerframe.maskdisplay.get()
            if ovly == 'tempo':
                colormap = 'tempo'
            else:
                colormap = 'viridis'

            # self.ui.sliceviewerframe.updatewl_fusion()

            # generate a new overlay
            for s in self.ui.data:
                if not self.ui.data[s].dset[ovly_str][ch]['ex']:
                    # self.ui.data[s].dset[ovly_str][ch]['base'] != ch or \
                    # self.ui.data[s].dset[ovly_str][ch]['mask'] != mask:

                    # additional check if box not previously grayed out.
                    # eg for cbv there might only be one DSC study, so just
                    # use the base grayscale for the dummy overlay image
                    if not self.ui.data[s].dset[ovly][ch]['ex']:
                        self.ui.data[s].dset[ovly_str][ch]['d'] = np.copy(self.ui.data[s].dset['raw'][ch]['d'])
                        self.ui.data[s].dset[ovly_str][ch]['ex'] = False
                        # self.ui.data[s].dset[ovly_str][ch]['base'] = ch
                        # self.ui.data[s].dset[ovly_str][ch]['mask'] = mask
                    else:

                        self.ui.data[s].dset[ovly_str][ch]['d'] = generate_overlay(
                            self.ui.data[s].dset['raw'][ch]['d'],
                            self.ui.data[s].dset[ovly][ch]['d'],
                            self.ui.data[s].mask['ET']['d'],
                            image_wl = [self.ui.sliceviewerframe.window[0],self.ui.sliceviewerframe.level[0]],
                            overlay_wl = self.ui.sliceviewerframe.wl[ovly],
                            overlay_intensity=self.config.OverlayIntensity,
                            colormap = colormap)
                        self.ui.data[s].dset[ovly_str][ch]['ex'] = True
                        # these may be redundant now
                        if False:
                            self.ui.data[s].dset[ovly_str][ch]['base'] = ch
                            self.ui.data[s].dset[ovly_str][ch]['mask'] = mask

                    # self.ui.data['overlay_d'] = copy.deepcopy(self.ui.data['overlay']),
            self.ui.dataselection = ovly_str

        else:
            self.sliderframe['dummy'].lift()
            # self.ui.set_dataselection()

        if updateslice:
            self.ui.updateslice()


    def updateslider(self,layer,slider):
        self.overlay_callback(wl=True)

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
        

