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
class CreateROIFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)

        self.buttonpress_id = None # temp var for keeping track of button press event
        self.finalROI_overlay_value = tk.BooleanVar(value=False)
        self.enhancingROI_overlay_value = tk.BooleanVar(value=False)
        roidict = {'ET':{'t12':None,'flair':None,'bc':None},'T2 hyper':{'t12':None,'flair':None,'bc':None}}
        self.overlaytype = tk.IntVar(value=self.config.OverlayType)
        self.layerlist = {'blast':['ET','T2 hyper'],'seg':['ET','TC','WT','all']}
        self.layer = tk.StringVar(value='ET')
        self.layerROI = tk.StringVar(value='ET')
        self.layertype = tk.StringVar(value='blast')
        self.currentroi = tk.IntVar(value=0)
        self.roilist = []

        ########################
        # layout for the buttons
        ########################

        self.frame.grid(row=2,column=1,rowspan=3,sticky='NEW')

        # overlay type
        overlaytype_label = ttk.Label(self.frame, text='overlay type: ')
        overlaytype_label.grid(row=0,column=0,padx=(50,0),sticky='e')
        self.overlaytype_button = ttk.Radiobutton(self.frame,text='z-score',variable=self.overlaytype,value=0,
                                                    command=Command(self.ui.updateslice,wl=True))
        self.overlaytype_button.grid(row=0,column=1,sticky='w')
        self.overlaytype_button = ttk.Radiobutton(self.frame,text='CBV',variable=self.overlaytype,value=1,
                                                    command=Command(self.ui.updateslice,wl=True))
        self.overlaytype_button.grid(row=0,column=2,sticky='w')

        # ROI buttons
        enhancingROI_label = ttk.Label(self.frame,text='overlay on/off')
        enhancingROI_label.grid(row=1,column=0,sticky='e')
        enhancingROI_overlay = ttk.Checkbutton(self.frame,text='',
                                               variable=self.enhancingROI_overlay_value,
                                               command=self.enhancingROI_overlay_callback)
        enhancingROI_overlay.grid(row=1,column=1,sticky='w')

        # enhancing layer choice
        if False:
            layerlabel = ttk.Label(self.frame,text='BLAST layer:')
            layerlabel.grid(row=0,column=0,sticky='w')
            self.layer.trace_add('write',lambda *args: self.layer.get())
            self.layermenu = ttk.OptionMenu(self.frame,self.layer,self.layerlist['blast'][0],
                                            *self.layerlist['blast'],command=self.layer_callback)
            self.layermenu.config(width=7)
            self.layermenu.grid(row=0,column=1,sticky='w')

        # ROI segmentation layer choice
            finalROI_overlay = ttk.Checkbutton(self.frame,text='',
                                            variable=self.finalROI_overlay_value,
                                            command=self.finalROI_overlay_callback)
            finalROI_overlay.grid(row=1,column=3,sticky='w')
            layerlabel = ttk.Label(self.frame,text='ROI layer:')
            layerlabel.grid(row=0,column=2,sticky='w')
            self.layerROI.trace_add('write',lambda *args: self.layerROI.get())
            self.layerROImenu = ttk.OptionMenu(self.frame,self.layerROI,self.layerlist['blast'][0],
                                            *self.layerlist['seg'],command=self.layerROI_callback)
            self.layerROImenu.config(width=8)
            self.layerROImenu.grid(row=0,column=3,sticky='w')


    #############
    # ROI methods
    ############# 

    # methods for BLAST layer options menu
    def layer_callback(self,layer=None,updateslice=True,updatedata=True,overlay=True):

        # if in the opposite mode, then switch same as if the checkbutton was used. 
        # but don't run the checkbutton callback because
        # don't yet have logic to check if the existing overlay is correct or
        # needs to be redone.
        # also if in ROI mode, then copy the relevant data back for BLAST mode.
        if self.finalROI_overlay_value.get() == True:
            self.updateData()
            self.finalROI_overlay_value.set(False)
        self.enhancingROI_overlay_value.set(True)
        self.ui.dataselection = 'seg_raw_fusion_d'

        self.ui.sliceviewerframe.updatewl_fusion()

        if layer is None:
            layer = self.layer.get()
        else:
            self.layer.set(layer)
        self.ui.currentlayer = layer
        roi = self.ui.get_currentroi()

        # when switching layers, raise/lower the corresponding sliders
        # slider values switch but no need to run re-blast immediately. 
        self.updatesliders()
        for s in ['t12','flair','bc']:
            if layer == 'T2 hyper':
                self.sliders[layer][s].lift()
                self.sliders['ET'][s].lower()
            # self.t2slider.configure(state='active')
            # self.updatebcsize(self.bct2size.get(),blast=False)
            else:
                self.sliders[layer][s].lift()
                self.sliders['T2 hyper'][s].lower()
                # self.t2slider.configure(state='disabled')
                # self.t1slider.configure(state='active')

        # generate a new overlay
        # TODO: check for existing instead of automatically re-generating
        # in blast mode, overlays are stored in main ui data, and are not associated with a ROI yet ( ie until create or update ROI event)
        if overlay:
            self.ui.data['seg_raw_fusion'] = generate_overlay(self.ui.data['raw'],self.ui.data['seg_raw'],layer=layer,
                                                                    overlay_intensity=self.config.OverlayIntensity)
            self.ui.data['seg_raw_fusion_d'] = copy.deepcopy(self.ui.data['seg_raw_fusion'])

        if updateslice:
            self.ui.updateslice()

    # and ROI layer options menu
    def layerROI_callback(self,layer=None,updateslice=True,updatedata=True):

        roi = self.ui.get_currentroi()
        if roi == 0:
            return
        # if in the opposite mode, then switch
        self.enhancingROI_overlay_value.set(False)
        self.finalROI_overlay_value.set(True)
        self.ui.dataselection = 'seg_fusion_d'

        self.ui.sliceviewerframe.updatewl_fusion()

        if layer is None:
            layer = self.layerROI.get()
        else:
            self.layerROI.set(layer)
        self.ui.currentROIlayer = self.layerROI.get()
        
        self.updatesliders()

        # a convenience reference
        data = self.ui.roi[roi].data
        # in seg mode, the context is an existing ROI, so the overlays are first stored directly in the ROI dict
        # then also copied back to main ui data
        # TODO: check mouse event, versus layer_callback called by statement
        if self.ui.sliceviewerframe.overlaytype.get() == 0:
            data['seg_fusion'] = generate_overlay(self.ui.data['raw'],data['seg'],contour=data['contour'],layer=layer,
                                                        overlay_intensity=self.config.OverlayIntensity)
        else:
            data['seg_fusion'] = generate_overlay(self.ui.data['raw'],data['seg'],layer=layer,
                                                        overlay_intensity=self.config.OverlayIntensity)

        data['seg_fusion_d'] = copy.deepcopy(data['seg_fusion'])

        if updatedata:
            self.updateData()

        if updateslice:
            self.ui.updateslice()

        return

    def update_layermenu_options(self,roi):
        roi = self.ui.get_currentroi()
        if self.ui.roi[roi].data['WT'] is None:
            layerlist = ['ET','TC']
        elif self.ui.roi[roi].data['ET'] is None:
            layerlist = ['WT']
        else:
            layerlist = self.layerlist['seg']
        menu = self.layerROImenu['menu']
        menu.delete(0,'end')
        for s in layerlist:
            menu.add_command(label=s,command = tk._setit(self.layerROI,s,self.layerROI_callback))
        self.layerROI.set(layerlist[0])
    
    def set_currentroi(self,var,index,mode):
        if mode == 'write':
            self.ui.set_currentroi()    
       
    def finalROI_overlay_callback(self,event=None):
        if self.finalROI_overlay_value.get() == False:
            self.ui.dataselection = 'raw'
            self.ui.data['raw'] = copy.deepcopy(self.ui.data['raw_copy'])
            self.ui.updateslice()
        else:
            self.enhancingROI_overlay_value.set(False)
            self.ui.dataselection = 'seg_fusion_d'
            # handle the case of switching manually to ROI mode with only one of ET T2 hyper selected.
            # eg the INDIGO case there won't be any ET. for now just a temp workaround.
            # but this might need to become the default behaviour for all cases, and if it's automatic
            # it won't pass through this callback but will be handled elsewhere.
            roi = self.ui.get_currentroi()
            if self.ui.roi[roi].status is False:
                if self.ui.roi[roi].data['WT'] is not None:
                    self.layerROI_callback(layer='WT')
                elif self.ui.roi[roi].data['ET'] is not None:
                    self.layerROI_callback(layer='ET')
            self.ui.updateslice(wl=True)

    def enhancingROI_overlay_callback(self,event=None):
        # if currently in roi mode, copy relevant data back to blast mode
        if self.finalROI_overlay_value.get() == True:
            self.updateData()

        if self.enhancingROI_overlay_value.get() == False:
            self.ui.dataselection = 'raw'
            self.ui.data['raw'] = copy.deepcopy(self.ui.data['raw_copy'])
            self.ui.updateslice()

        else:
            self.finalROI_overlay_value.set(False)
            self.ui.dataselection = 'seg_raw_fusion_d'
            self.ui.updateslice(wl=True)

    def enhancingROI_callback(self,event=None):
        self.finalROI_overlay_value.set(False)
        self.enhancingROI_overlay_value.set(True)
        self.ui.runblast()

    def selectROI(self,event=None):
        if self.enhancingROI_overlay_value.get(): # only activate cursor in BLAST mode
            self.buttonpress_id = self.ui.sliceviewerframe.canvas.callbacks.connect('button_press_event',self.ROIclick)
            self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='crosshair')
            # lock pan and zoom
            # self.ui.sliceviewerframe.canvas.widgetlock(self.ui.sliceviewerframe)
            # also if zoom or pan currently active, turn off
            # if self.ui.sliceviewerframe.tbar.mode == _Mode.PAN or self.ui.sliceviewerframe.tbar.mode == _Mode.ZOOM:
            #     self.ui.sliceviewerframe.tbar.mode = _Mode.NONE
                # self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()

        return None
    
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
        

    def updateROIData(self):
        # save current dataset into the current roi. 
        for k,v in self.ui.data.items():
            if k != 'raw':
                self.ui.roi[self.ui.currentroi].data[k] = copy.deepcopy(self.ui.data[k])
            else: # reference only
                self.ui.roi[self.ui.currentroi].data[k] = self.ui.data[k]

    # calculate the combined mask from separate layers
    def updateBLAST(self,layer=None):
        # record slider values
        if layer is None:
            layer = self.layer.get()

        if all(self.ui.data['blast'][x] is not None for x in ['ET','T2 hyper']):
            # self.ui.data['seg_raw'] = self.ui.data['blast']['ET'].astype('int')*2 + (self.ui.data['blast']['T2 hyper'].astype('int'))
            self.ui.data['seg_raw'] = (self.ui.data['blast']['T2 hyper'].astype('int'))
            et = np.where(self.ui.data['blast']['ET'])
            self.ui.data['seg_raw'][et] += 4
        elif self.ui.data['blast']['ET'] is not None:
            self.ui.data['seg_raw'] = self.ui.data['blast']['ET'].astype('int')*4
        elif self.ui.data['blast']['T2 hyper'] is not None:
            self.ui.data['seg_raw'] = self.ui.data['blast']['T2 hyper'].astype('int')

    def updateData(self):
        for k in ['seg_fusion_d','seg_fusion','seg_raw_fusion','seg_raw_fusion_d','seg_raw','blast']:
            self.ui.data[k] = copy.deepcopy(self.ui.roi[self.ui.currentroi].data[k])
        self.updatesliders()


    # eliminate all ROIs, ie for loading another case
    def resetROI(self):
        self.currentroi.set(0)
        self.ui.roi = [0]
        self.ui.roiframe.finalROI_overlay_value.set(False)
        self.ui.roiframe.enhancingROI_overlay_value.set(False)
        self.ui.roiframe.layertype.set('blast')
        self.ui.roiframe.layer.set('ET')
        self.ui.dataselection='t1+'

    def append_roi(self,d):
        for k,v in d.items():
            if isinstance(v,dict):
                self.append_roi(d)
            else:
                v.append(0)

    def ROIstats(self):
        
        roi = self.ui.get_currentroi()
        data = self.ui.roi[roi].data
        for t in ['ET','TC','WT']:
            # check for a complete segmentation
            if t not in data.keys():
                continue
            elif data[t] is None:
                continue
            self.ui.roi[roi].stats['vol'][t] = len(np.where(data[t])[0])

            if self.ui.data['label'] is not None:
                sums = data['manual_'+t] + data[t]
                subs = data['manual_'+t] - data[t]
                        
                TP = len(np.where(sums == 2)[0])
                FP = len(np.where(subs == -1)[0])
                TN = len(np.where(sums == 0)[0])
                FN = len(np.where(subs == 1)[0])

                self.ui.roi[roi].stats['spec'][t] = TN/(TN+FP)
                self.ui.roi[roi].stats['sens'][t] = TP/(TP+FN)
                self.ui.roi[roi].stats['dsc'][t] = 1-dice(data['manual_'+t].flatten(),data[t].flatten()) 

                # Calculate volumes
                self.ui.roi[roi].stats['vol']['manual_'+t] = len(np.where(data['manual_'+t])[0])

