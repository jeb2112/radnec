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
from RangeSlider.RangeSlider import RangeSliderH

from src.OverlayPlots import *
from src.CreateFrame import CreateFrame,Command
from src.ROI import ROI

# contains various ROI methods and variables
class CreateROIFrame(CreateFrame):
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

        self.frame.grid(row=2,column=1,rowspan=5,sticky='ne')

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
            ovly_str = 'overlay_' + ovly
            base = self.ui.sliceviewerframe.basedisplay.get()
            if ovly == 'cbv':
                ovly_data = ovly
            else:
                # check for available data. or implement by deactivating button.
                if False:
                    if not self.ui.data[s].dset[base]['ex']:
                        print('{} data not loaded'.format(base))
                        return
                ovly_data = ovly + base

            # self.ui.sliceviewerframe.updatewl_fusion()

            # generate a new overlay
            for s in self.ui.data:
                if not self.ui.data[s].dset[ovly_str]['ex'] or self.ui.data[s].dset[ovly_str]['base'] != base or wl:

                    # additional check if box not previously grayed out.
                    # eg for cbv there might only be one DSC study, so just
                    # use the base grayscale for the dummy overlay image
                    if not self.ui.data[s].dset[ovly_data]['ex']:
                        self.ui.data[s].dset[ovly_str]['d'] = np.copy(self.ui.data[s].dset[base]['d'])
                        self.ui.data[s].dset[ovly_str]['ex'] = False
                        self.ui.data[s].dset[ovly_str]['base'] = base
                    else:
                        if self.mask_value.get():
                            mask = self.ui.data[s].dset[self.mask_type.get()]['d']
                        elif self.overlay_type.get() == 'tempo': # special case for tempo
                            mask = np.where(self.ui.data[s].dset[ovly_data]['d'] != 2)
                        else:
                            mask = None

                        self.ui.data[s].dset[ovly_str]['d'] = generate_overlay(
                            self.ui.data[s].dset[base]['d'],
                            self.ui.data[s].dset[ovly_data]['d'],
                            mask,
                            image_wl = [self.ui.sliceviewerframe.window[0],self.ui.sliceviewerframe.level[0]],
                            overlay_wl = self.ui.sliceviewerframe.wl[ovly],
                            overlay_intensity=self.config.OverlayIntensity,
                            colormap = self.config.OverlayCmap[ovly])
                        self.ui.data[s].dset[ovly_str]['ex'] = True
                        self.ui.data[s].dset[ovly_str]['base'] = base

            self.ui.dataselection = ovly_str

        else:
            self.sliderframe['dummy'].lift()
            self.ui.set_dataselection()

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

    # and ROI layer options menu
    def layerROI_callback(self,layer=None,updateslice=True,updatedata=True):

        roi = self.ui.get_currentroi()
        if roi == 0:
            return
        # if in the opposite mode, then switch
        self.overlay_value.set(False)
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
        if self.ui.sliceviewerframe.overlay_type.get() == 0:
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
            self.overlay_value.set(False)
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

    def selectROI(self,event=None):
        if self.overlay_value.get(): # only activate cursor in BLAST mode
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
        for k in ['overlay_fusion','overlay_fusion_d','seg_raw']:
            self.ui.data[k] = copy.deepcopy(self.ui.roi[self.ui.currentroi].data[k])
        self.updatesliders()


    # eliminate all ROIs, ie for loading another case
    def resetROI(self):
        self.currentroi.set(0)
        self.ui.roi = [0]
        self.ui.roiframe.finalROI_overlay_value.set(False)
        self.ui.roiframe.overlay_value.set(False)
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

