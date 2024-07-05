import os,sys
import re
import numpy as np
import pickle
import copy
import logging
import time
import shutil
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import _Mode
matplotlib.use('TkAgg')
import SimpleITK as sitk
import nibabel as nb
import PIL
from skimage.morphology import disk,square,binary_dilation,binary_closing,flood_fill,ball,cube,reconstruction
from skimage.measure import find_contours
from scipy.spatial.distance import dice
from scipy.ndimage import binary_closing as scipy_binary_closing
from scipy.io import savemat
if os.name == 'posix':
    from cucim.skimage.morphology import binary_closing as cucim_binary_closing
elif os.name == 'nt':
    from cupyx.scipy.ndimage import binary_closing as cupy_binary_closing
import cupy as cp
import cc3d

from src.OverlayPlots import *
from src.CreateFrame import CreateFrame,Command
from src.ROI import ROI

# contains various ROI methods and variables for 'BLAST' mode
class CreateROIFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)

        self.buttonpress_id = None # temp var for keeping track of button press event
        self.finalROI_overlay_value = tk.BooleanVar(value=False)
        self.enhancingROI_overlay_value = tk.BooleanVar(value=False)
        self.SAM_overlay_value = tk.BooleanVar(value=False)
        roidict = {'ET':{'t12':None,'flair':None,'bc':None},'T2 hyper':{'t12':None,'flair':None,'bc':None}}
        self.thresholds = copy.deepcopy(roidict)
        self.sliders = copy.deepcopy(roidict)
        self.sliderlabels = copy.deepcopy(roidict)
        self.thresholds['ET']['t12'] = tk.DoubleVar(value=self.ui.config.T1default)
        self.thresholds['T2 hyper']['t12'] = tk.DoubleVar(value=self.ui.config.T2default)
        self.thresholds['ET']['flair'] = tk.DoubleVar(value=self.ui.config.T2default)
        self.thresholds['T2 hyper']['flair'] = tk.DoubleVar(value=self.ui.config.T2default)
        self.thresholds['ET']['bc'] = tk.DoubleVar(value=self.ui.config.BCdefault[0])
        self.thresholds['T2 hyper']['bc'] = tk.DoubleVar(value=self.ui.config.BCdefault[1])
        self.overlay_type = tk.IntVar(value=0)
        self.layerlist = {'blast':['ET','T2 hyper'],'seg':['ET','TC','WT','all'],'sam':['ET','TC','WT','all']}
        self.layer = tk.StringVar(value='ET')
        self.layerROI = tk.StringVar(value='ET')
        self.layerSAM = tk.StringVar(value='ET')
        self.layertype = tk.StringVar(value='blast')
        self.currentroi = tk.IntVar(value=0)
        self.roilist = []

        ########################
        # layout for the buttons
        ########################

        # dummy frame to hide
        self.dummy_frame = ttk.Frame(self.parentframe,padding='0')
        self.dummy_frame.grid(row=2,column=4,sticky='news')

        # actual frame
        self.frame.grid(row=2,column=4,rowspan=3,sticky='NE')

        # ROI buttons for raw BLAST segmentation
        enhancingROI_label = ttk.Label(self.frame,text='overlay on/off')
        enhancingROI_label.grid(row=1,column=0,sticky='e')
        enhancingROI_overlay = ttk.Checkbutton(self.frame,text='',
                                               variable=self.enhancingROI_overlay_value,
                                               command=self.enhancingROI_overlay_callback)
        enhancingROI_overlay.grid(row=1,column=1,sticky='w')

        layerlabel = ttk.Label(self.frame,text='BLAST layer:')
        layerlabel.grid(row=0,column=0,sticky='w')
        self.layer.trace_add('write',lambda *args: self.layer.get())
        self.layermenu = ttk.OptionMenu(self.frame,self.layer,self.layerlist['blast'][0],
                                        *self.layerlist['blast'],command=self.layer_callback)
        self.layermenu.config(width=7)
        self.layermenu.grid(row=0,column=1,sticky='w')

        # ROI buttons for final smoothed segmentation
        finalROI_overlay = ttk.Checkbutton(self.frame,text='',
                                           variable=self.finalROI_overlay_value,
                                           command=self.finalROI_overlay_callback)
        finalROI_overlay.grid(row=1,column=3,sticky='w')
        layerlabel = ttk.Label(self.frame,text='ROI layer:')
        layerlabel.grid(row=0,column=2,sticky='w')
        self.layerROI.trace_add('write',lambda *args: self.layerROI.get())
        self.layerROImenu = ttk.OptionMenu(self.frame,self.layerROI,self.layerlist['blast'][0],
                                           *self.layerlist['seg'],command=self.layerROI_callback)
        self.layerROImenu.config(width=4)
        self.layerROImenu.grid(row=0,column=3,sticky='w')

        # ROI button for SAM segmentation
        SAM_overlay = ttk.Checkbutton(self.frame,text='',
                                           variable=self.SAM_overlay_value,
                                           command=self.SAM_overlay_callback)
        SAM_overlay.grid(row=1,column=5,sticky='w')
        layerlabel = ttk.Label(self.frame,text='SAM layer:')
        layerlabel.grid(row=0,column=4,sticky='w')
        self.layerSAM.trace_add('write',lambda *args: self.layerSAM.get())
        self.layerSAMmenu = ttk.OptionMenu(self.frame,self.layerSAM,self.layerlist['blast'][0],
                                           *self.layerlist['seg'],command=self.layerSAM_callback)
        self.layerSAMmenu.config(width=4)
        self.layerSAMmenu.grid(row=0,column=5,sticky='w')

        # for multiple roi's, n'th roi number choice
        roinumberlabel = ttk.Label(self.frame,text='ROI number:')
        roinumberlabel.grid(row=0,column=6,sticky='w')
        self.currentroi.trace_add('write',self.set_currentroi)
        self.roinumbermenu = ttk.OptionMenu(self.frame,self.currentroi,*self.roilist,command=self.roinumber_callback)
        self.roinumbermenu.config(width=2)
        self.roinumbermenu.grid(row=0,column=7,sticky='w')
        self.roinumbermenu.configure(state='disabled')

        # select ROI button
        selectROI = ttk.Button(self.frame,text='select ROI',command = self.selectROI)
        selectROI.grid(row=1,column=6,sticky='w')

        # save ROI button
        saveROI = ttk.Button(self.frame,text='save ROI',command = self.saveROI)
        saveROI.grid(row=1,column=8,sticky='w')

        # clear ROI button
        clearROI = ttk.Button(self.frame,text='clear ROI',command = self.clearROI)
        clearROI.grid(row=1,column=7,sticky='w')
        self.frame.update()

        ########################
        # layout for the sliders
        ########################

        # frames for sliders
        self.sliderframe = {}
        self.sliderframe['ET'] = ttk.Frame(self.frame,padding='0')
        self.sliderframe['ET'].grid(row=3,column=2,columnspan=7,sticky='e')
        self.sliderframe['T2 hyper'] = ttk.Frame(self.frame,padding='0')
        self.sliderframe['T2 hyper'].grid(row=3,column=2,columnspan=7,sticky='e')

        # ET sliders
        # t1 slider
        t1label = ttk.Label(self.sliderframe['ET'], text='T1/T2')
        t1label.grid(column=0,row=0,sticky='w')

        self.sliders['ET']['t12'] = ttk.Scale(self.sliderframe['ET'],from_=-4,to=4,variable=self.thresholds['ET']['t12'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'ET','t12'),orient='horizontal')
        self.sliders['ET']['t12'].grid(row=0,column=1,sticky='e')
        self.sliderlabels['ET']['t12'] = ttk.Label(self.sliderframe['ET'],text=self.thresholds['ET']['t12'].get())
        self.sliderlabels['ET']['t12'].grid(row=0,column=2,sticky='e')

        #flairt1 slider
        flairt1label = ttk.Label(self.sliderframe['ET'], text='flair')
        flairt1label.grid(row=1,column=0,sticky='w')
        self.sliders['ET']['flair'] = ttk.Scale(self.sliderframe['ET'],from_=-4,to=4,variable=self.thresholds['ET']['flair'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'ET','flair'),orient='horizontal')
        self.sliders['ET']['flair'].grid(row=1,column=1,sticky='e')
        self.sliderlabels['ET']['flair'] = ttk.Label(self.sliderframe['ET'],text=self.thresholds['ET']['flair'].get())
        self.sliderlabels['ET']['flair'].grid(row=1,column=2,sticky='e')

        #braint1 cluster slider
        bclabel = ttk.Label(self.sliderframe['ET'],text='b.c.')
        bclabel.grid(row=2,column=0,sticky='w')
        self.sliders['ET']['bc'] = ttk.Scale(self.sliderframe['ET'],from_=0,to=4,variable=self.thresholds['ET']['bc'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'ET','bc'),orient='horizontal')
        self.sliders['ET']['bc'].grid(row=2,column=1,sticky='e')
        self.sliderlabels['ET']['bc'] = ttk.Label(self.sliderframe['ET'],text=self.thresholds['ET']['bc'].get())
        self.sliderlabels['ET']['bc'].grid(row=2,column=2,sticky='e')


        # T2 hyper sliders
        # t2 slider
        t2label = ttk.Label(self.sliderframe['T2 hyper'], text='T1/T2')
        t2label.grid(row=0,column=0,sticky='w')
        self.sliders['T2 hyper']['t12'] = ttk.Scale(self.sliderframe['T2 hyper'],from_=-4,to=4,variable=self.thresholds['T2 hyper']['t12'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'T2 hyper','t12'),orient='horizontal')
        self.sliders['T2 hyper']['t12'].grid(row=0,column=1,sticky='e')
        self.sliderlabels['T2 hyper']['t12'] = ttk.Label(self.sliderframe['T2 hyper'],text=self.thresholds['T2 hyper']['t12'].get())
        self.sliderlabels['T2 hyper']['t12'].grid(row=0,column=2,sticky='e')

        #flairt2 slider
        flairt2label = ttk.Label(self.sliderframe['T2 hyper'], text='flair')
        flairt2label.grid(row=1,column=0,sticky='w')
        self.sliders['T2 hyper']['flair'] = ttk.Scale(self.sliderframe['T2 hyper'],from_=-4,to=4,variable=self.thresholds['T2 hyper']['flair'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'T2 hyper','flair'),orient='horizontal')
        self.sliders['T2 hyper']['flair'].grid(row=1,column=1,sticky='e')
        self.sliderlabels['T2 hyper']['flair'] = ttk.Label(self.sliderframe['T2 hyper'],text=self.thresholds['T2 hyper']['flair'].get())
        self.sliderlabels['T2 hyper']['flair'].grid(row=1,column=2,sticky='e')

        #braint2 cluster slider
        bclabel = ttk.Label(self.sliderframe['T2 hyper'],text='b.c.')
        bclabel.grid(row=2,column=0,sticky='w')
        self.sliders['T2 hyper']['bc'] = ttk.Scale(self.sliderframe['T2 hyper'],from_=0,to=4,variable=self.thresholds['T2 hyper']['bc'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'T2 hyper','bc'),orient='horizontal')
        self.sliders['T2 hyper']['bc'].grid(row=2,column=1,sticky='e')
        self.sliderlabels['T2 hyper']['bc'] = ttk.Label(self.sliderframe['T2 hyper'],text=self.thresholds['T2 hyper']['bc'].get())
        self.sliderlabels['T2 hyper']['bc'].grid(row=2,column=2,sticky='e')


    #############
    # ROI methods
    ############# 

    # main method for handling ET versus WT selection in BLAST raw segmentation
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
        self.ui.dataselection = 'seg_raw_fusion'

        self.ui.sliceviewerframe.updatewl_fusion()

        if layer is None:
            layer = self.layer.get()
        else:
            self.layer.set(layer)
        roi = self.ui.get_currentroi()

        # when switching layers, raise/lower the corresponding sliders
        # slider values switch but no need to run re-blast immediately. 
        self.updatesliders()
        if layer == 'T2 hyper':
            self.sliderframe[layer].lift()
            # self.sliders['ET'].lower()
        else:
            self.sliderframe[layer].lift()
            # self.sliders['T2 hyper'].lower()

        # generate a new overlay
        # in blast mode, overlays are stored in main ui data, and are not associated with a ROI yet ( ie until create or update ROI event)
        # logic to use existing overlay or force a new one. might need fixing.
        if overlay:
            s = self.ui.s
            chlist = [self.ui.chselection , 'flair']
            for ch in chlist:
                # note that this check does not cover both layers 'ET' and 'T2 Hyper' separately.
                # they are assumed either both or neither to exist. probably needs to be fixed.
                if not self.ui.data[s].dset['seg_raw_fusion'][ch]['ex'] or False:
                    self.ui.data[s].dset['seg_raw_fusion'][ch]['d'+layer] = \
                        generate_blast_overlay(self.ui.data[s].dset['raw'][ch]['d'],
                                                self.ui.data[s].dset['seg_raw'][self.ui.chselection]['d'],
                                                layer=layer,overlay_intensity=self.config.OverlayIntensity)
                    self.ui.data[s].dset['seg_raw_fusion'][ch]['ex'] = True
                    # self.ui.data[s].dset['seg_raw_fusion_d'][ch]['d'+layer] = copy.deepcopy(self.ui.data[s].dset['seg_raw_fusion'][self.ui.chselection]['d'+layer])

        if updateslice:
            self.ui.updateslice()

    # main method for handling ET,TC,WT selection in final ROI smoothed segmentation
    def layerROI_callback(self,layer=None,updateslice=True,updatedata=True):

        roi = self.ui.get_currentroi()
        if roi == 0:
            return
        # if in the opposite mode, then switch
        self.enhancingROI_overlay_value.set(False)
        self.finalROI_overlay_value.set(True)
        self.ui.dataselection = 'seg_fusion'

        self.ui.sliceviewerframe.updatewl_fusion()

        if layer is None:
            layer = self.layerROI.get()
        else:
            self.layerROI.set(layer)
        self.ui.currentROIlayer = self.layerROI.get()
        
        self.updatesliders()

        # a convenience reference
        data = self.ui.roi[self.ui.s][roi].data
        # in seg mode, the context is an existing ROI, so the overlays are first stored directly in the ROI dict
        # then also copied back to main ui data
        # TODO: check mouse event, versus layer_callback called by statement
        if self.ui.sliceviewerframe.overlay_type.get() == 0: # contour not updated lately
            data['seg_fusion'] = generate_blast_overlay(self.ui.data[self.ui.s].dset['raw'][self.ui.chselection]['d'],
                                                        data['seg'],contour=data['contour'],layer=layer,
                                                        overlay_intensity=self.config.OverlayIntensity)
        else:
            for ch in [self.ui.chselection,'flair']:
                data['seg_fusion'][ch] = generate_blast_overlay(self.ui.data[self.ui.s].dset['raw'][ch]['d'],
                                                                data['seg'],layer=layer,
                                                            overlay_intensity=self.config.OverlayIntensity)

        # data['seg_fusion_d'] = copy.deepcopy(data['seg_fusion'])

        if updatedata:
            self.updateData()

        if updateslice:
            self.ui.updateslice()

        return
    
    def layerSAM_callback(self):
        return

    # update ROI layers that can be displayed according to availability
    def update_layermenu_options(self,roi):
        roi = self.ui.get_currentroi()
        if self.ui.roi[self.ui.s][roi].data['WT'] is None:
            layerlist = ['ET','TC']
        elif self.ui.roi[self.ui.s][roi].data['ET'] is None:
            layerlist = ['WT']
        else:
            layerlist = self.layerlist['seg']
        menu = self.layerROImenu['menu']
        menu.delete(0,'end')
        for s in layerlist:
            menu.add_command(label=s,command = tk._setit(self.layerROI,s,self.layerROI_callback))
        self.layerROI.set(layerlist[0])

    # methods for roi number choice menu
    def roinumber_callback(self,item=None):
        if self.enhancingROI_overlay_value.get() == True:
            self.enhancingROI_overlay_value.set(False)
            self.finalROI_overlay_value.set(True)
            self.finalROI_overlay_callback()

        self.ui.set_currentroi()
        # reference or copy
        self.layerROI_callback(updatedata=True)
        self.ui.updateslice()
        return
    
    def update_roinumber_options(self,n=None):
        if n is None:
            n = len(self.ui.roi[self.ui.s])
        menu = self.roinumbermenu['menu']
        menu.delete(0,'end')
        # 1-based indexing
        for s in [str(i) for i in range(1,n)]:
            menu.add_command(label=s,command = tk._setit(self.currentroi,s,self.roinumber_callback))
        self.roilist = [str(i) for i in range(1,n)]
        if n>1:
            self.roinumbermenu.configure(state='active')
        else:
            self.roinumbermenu.configure(state='disabled')
            self.finalROI_overlay_value.set(False)

    def set_currentroi(self,var,index,mode):
        if mode == 'write':
            self.ui.set_currentroi()    

    # callbacks for the BLAST threshold slider bars

    # def updatet1threshold(self,event=None,currentslice=True):
    #     self.enhancingROI_overlay_value.set(True)
    #     # for now, this event reverts to BLAST preview mode and will not directly reprocess the final segmentation
    #     if self.finalROI_overlay_value.get() == True:
    #         self.finalROI_overlay_value.set(False)
    #         self.enhancingROI_overlay_callback()

    #     # force recalc of gates
    #     layer = self.layer.get()
    #     self.ui.blastdata[self.ui.s]['blast']['gates'][layer] = None
    #     self.ui.runblast(currentslice=currentslice)
    #     self.t1sliderlabel['text'] = '{:.1f}'.format(self.t1threshold.get())

    # updates the text field showing the value during slider drag
    # def updatet1label(self,event=None):
    #     self.t1sliderlabel['text'] = '{:.1f}'.format(self.t1threshold.get())

    # def updatet2threshold(self,event=None,currentslice=True):
    #     self.enhancingROI_overlay_value.set(True)
    #     if self.finalROI_overlay_value.get() == True:
    #         self.finalROI_overlay_value.set(False)
    #         self.enhancingROI_overlay_callback()
    #     # force recalc of gates
    #     layer = self.layer.get()
    #     self.ui.blastdata[self.ui.s]['blast']['gates'][layer] = None
    #     self.ui.runblast(currentslice=currentslice)
    #     self.t2sliderlabel['text'] = '{:.1f}'.format(self.t2threshold.get())
    #     # ie not using this workflow presently
    #     if self.finalROI_overlay_value.get() == True:
    #         self.ROIclick(do3d=True)
    #     return
    
    # def updatet2label(self,event=None):
    #     self.t2sliderlabel['text'] = '{:.1f}'.format(self.t2threshold.get())

    # def updateflairt1threshold(self,event=None,currentslice=True):
    #     self.enhancingROI_overlay_value.set(True)
    #     if self.finalROI_overlay_value.get() == True:
    #         self.finalROI_overlay_value.set(False)
    #         self.enhancingROI_overlay_callback()
    #     # force recalc of gates
    #     layer = self.layer.get()
    #     self.ui.blastdata[self.ui.s]['blast']['gates'][layer] = None
    #     self.ui.runblast(currentslice=currentslice)
    #     self.flairsliderlabel['text'] = '{:.1f}'.format(self.flairt1threshold.get())
    #     # ie not using this workflow presently
    #     if self.finalROI_overlay_value.get() == True:
    #         self.ROIclick(do3d=True)
    #     return
    
    # def updateflairt1label(self,event=None):
    #     self.flairsliderlabel['text'] = '{:.1f}'.format(self.flairt1threshold.get())

    # def updateflairt2threshold(self,event=None,currentslice=True):
    #     self.enhancingROI_overlay_value.set(True)
    #     if self.finalROI_overlay_value.get() == True:
    #         self.finalROI_overlay_value.set(False)
    #         self.enhancingROI_overlay_callback()
    #     # force recalc of gates
    #     layer = self.layer.get()
    #     self.ui.blastdata[self.ui.s]['blast']['gates'][layer] = None
    #     self.ui.runblast(currentslice=currentslice)
    #     self.flairsliderlabel['text'] = '{:.1f}'.format(self.flairt2threshold.get())
    #     # ie not using this workflow presently
    #     if self.finalROI_overlay_value.get() == True:
    #         self.ROIclick(do3d=True)
    #     return
    
    # def updateflairt1label(self,event=None):
    #     self.flairsliderlabel['text'] = '{:.1f}'.format(self.flairt1threshold.get())

    def updateslider(self,layer,slider,event=None):
        self.enhancingROI_overlay_value.set(True)
        if self.finalROI_overlay_value.get() == True:
            self.finalROI_overlay_value.set(False)
            self.enhancingROI_overlay_callback()
        # layer = self.layer.get()
        if slider == 'bc':
            self.ui.blastdata[self.ui.s]['blast']['gates']['brain '+layer] = None
        self.ui.blastdata[self.ui.s]['blast']['gates'][layer] = None
        self.ui.runblast(currentslice=True)
        self.updatesliderlabel(layer,slider)

    def updatesliderlabel(self,layer,slider):
        # if 'T2 hyper' in self.sliderlabels.keys() and 'T2 hyper' in self.sliders.keys():
        try:
            self.sliderlabels[layer][slider]['text'] = '{:.1f}'.format(self.sliders[layer][slider].get())
        except KeyError as e:
            print(e)
        # else:
        #     a=1

    # def updatebct1size(self,event=None):
    #     self.enhancingROI_overlay_value.set(True)
    #     if self.finalROI_overlay_value.get() == True:
    #         self.finalROI_overlay_value.set(False)
    #         self.enhancingROI_overlay_callback()
    #     layer = self.layer.get()
    #     self.ui.blastdata[self.ui.s]['blast']['gates']['brain '+layer] = None
    #     self.ui.runblast(currentslice=True)
    #     self.updatebct1label()
    #     # self.updatesliderlabel(layer,slider)
    #     return

    # def updatebct1label(self,event=None):
    #     bct1size = self.ui.get_bct1size()
    #     self.bct1sliderlabel['text'] = '{:.1f}'.format(bct1size)

    # def updatebct2size(self,event=None):
    #     self.enhancingROI_overlay_value.set(True)
    #     if self.finalROI_overlay_value.get() == True:
    #         self.finalROI_overlay_value.set(False)
    #         self.enhancingROI_overlay_callback()
    #     layer = self.layer.get()
    #     self.ui.blastdata[self.ui.s]['blast']['gates']['brain '+layer] = None
    #     self.ui.runblast(currentslice=True)
    #     self.updatebct2label()
    #     return

    # def updatebct2label(self,event=None):
    #     bct2size = self.ui.get_bct2size()
    #     self.bct2sliderlabel['text'] = '{:.1f}'.format(bct2size)

    # switch to show sliders and values according to current layer being displayed
    def updatesliders(self):
        if self.enhancingROI_overlay_value.get() == True:
            layer = self.layer.get()
        elif self.finalROI_overlay_value.get() == True:
            # ie display slider values that were used for current ROI
            layer = self.layerROI.get()
            if layer == 'WT':
                layer = 'T2 hyper'
            else:
                layer = 'ET'
        for sl in ['t12','flair','bc']:
            self.thresholds[layer][sl].set(self.ui.blastdata[self.ui.s]['blast']['params'][layer][sl])
            self.updatesliderlabel(layer,sl)
       
    # callback for final smoothed ROI on/off selection
    def finalROI_overlay_callback(self,event=None):
        if self.finalROI_overlay_value.get() == False:
            # base display, not data selection
            self.ui.dataselection = 'raw'
            if False: # no longer needed?
                self.ui.data[self.ui.dataselection][self.ui.chselection]['d'] = copy.deepcopy(self.ui.data[self.ui.chselection+'_copy']['d'])
            self.ui.updateslice()
        else:
            self.enhancingROI_overlay_value.set(False)
            self.ui.dataselection = 'seg_fusion'
            # handle the case of switching manually to ROI mode with only one of ET T2 hyper selected.
            # eg the INDIGO case there won't be any ET. for now just a temp workaround.
            # but this might need to become the default behaviour for all cases, and if it's automatic
            # it won't pass through this callback but will be handled elsewhere.
            roi = self.ui.get_currentroi()
            if self.ui.roi[self.ui.s][roi].status is False:
                if self.ui.roi[self.ui.s][roi].data['WT'] is not None:
                    self.layerROI_callback(layer='WT')
                elif self.ui.roi[self.ui.s][roi].data['ET'] is not None:
                    self.layerROI_callback(layer='ET')
            self.ui.updateslice(wl=True)

    # callback for raw BLAST segmentation on/off selection
    def enhancingROI_overlay_callback(self,event=None):
        # if currently in roi mode, copy relevant data back to blast mode
        if self.finalROI_overlay_value.get() == True:
            self.updateData()

        if self.enhancingROI_overlay_value.get() == False:
            # base display, not data selection
            self.ui.dataselection = 'raw'
            if False:
                self.ui.data['raw'][self.ui.chselection]['d'] = copy.deepcopy(self.ui.data['t1+_copy']['d'])
            self.ui.updateslice()

        else:
            self.finalROI_overlay_value.set(False)
            self.ui.dataselection = 'seg_raw_fusion'
            self.ui.updateslice(wl=True)

    def SAM_overlay_callback(self):
        return

    # creates a ROI selection button press event
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
        
    # processes a ROI selection button press event
    def ROIclick(self,event=None,do3d=True):
        if event:
            if event.button > 1: # ROI selection on left mouse only
                return
            if self.enhancingROI_overlay_value.get() == False: # no selection if BLAST mode not active
                return
            
        # self.ui.sliceviewerframe.canvas.widgetlock.release(self.ui.sliceviewerframe)
        self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='watch')
        self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()
        if event:
            # print(event.xdata,event.ydata)
            # need check for inbounds
            # convert coords from dummy label axis
            s = event.inaxes.format_coord(event.xdata,event.ydata)
            event.xdata,event.ydata = map(float,re.findall(r"(?:\d*\.\d+)",s))
            if event.xdata < 0 or event.ydata < 0:
                return None
            # elif self.ui.data['raw'][0,self.ui.get_currentslice(),int(event.x),int(event.y)] == 0:
            #     print('Clicked in background')
            #     return
            else:
                roi = self.ui.get_currentroi()
                # if current roi has segmentations for both compartments, start a new ROI
                # otherwise if only 1 segmentation is present and the mouse click is for that
                # same compartment, it will be updated.
                if roi > 0:
                    if self.ui.roi[self.ui.s][roi].data['ET'] is not None and self.ui.roi[self.ui.s][roi].data['WT'] is not None:
                        self.createROI(int(event.xdata),int(event.ydata),self.ui.get_currentslice())
                    else:
                        self.updateROI(event)
                else:
                    self.createROI(int(event.xdata),int(event.ydata),self.ui.get_currentslice())
            
        roi = self.ui.get_currentroi()
        try:
            self.closeROI(self.ui.data[self.ui.s].dset['seg_raw'][self.ui.chselection]['d'],self.ui.get_currentslice(),do3d=do3d)
        except Exception as e:
            print(e)
            print('ROI not completed')
            return
        # update layer menu
        self.update_layermenu_options(self.ui.roi[self.ui.s][roi])

        if False: # not ported yet
            self.ROIstats()
        # fusionstack = np.zeros((2,155,240,240))
        # note some duplicate calls to generate_overlay should be removed
        for ch in [self.ui.chselection,'flair']:
            fusionstack = generate_blast_overlay(self.ui.data[self.ui.s].dset['raw'][ch]['d'],
                                             self.ui.roi[self.ui.s][roi].data['seg'],
                                            layer=self.ui.roiframe.layer.get(),
                                            overlay_intensity=self.config.OverlayIntensity)
            self.ui.roi[self.ui.s][roi].data['seg_fusion'][ch] = fusionstack
            if False:
                fusionstack = generate_blast_overlay(self.ui.data[self.ui.s].dset['raw'][ch]['d'],
                                                self.ui.roi[self.ui.s][roi].data['sam'],
                                                layer=self.ui.roiframe.layer.get(),
                                                overlay_intensity=self.config.OverlayIntensity)
                self.ui.roi[self.ui.s][roi].data['sam_fusion'][ch] = fusionstack
        # self.ui.roi[self.ui.s][roi].data['seg_fusion_d'] = copy.deepcopy(self.ui.roi[self.ui.s][roi].data['seg_fusion'])
        # need to update ui data here?? or just let it run from layerROI_callback below
        if False:
            self.updateData()

        # if triggered by a button event
        if self.buttonpress_id:
            self.ui.sliceviewerframe.canvas.callbacks.disconnect(self.buttonpress_id)
            self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='')
            self.buttonpress_id = None
        self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='arrow')
        self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()

        # currently this logic stays in BLAST mode if both ET and t2 hyper are not yet selected.
        # however, the preferred workflow may be to switch to ROI mode automatically after 1st BLAST
        # segmentation is selected, then manually jump back to BLAST mode for the 2nd selection
        roi = self.ui.get_currentroi()
        if self.ui.roi[self.ui.s][roi].status:
            self.finalROI_overlay_value.set(True)
            self.enhancingROI_overlay_value.set(False)
            self.ui.dataselection = 'seg_fusion'
            self.layerROI_callback(layer='ET')

            # output final ROI to files for follow-on SAM processing
            self.saveROI(roi,sam=True)
            self.segment_sam()
            # TODO. reload segmentations to viewer
        else:
            self.finalROI_overlay_value.set(False)
            self.enhancingROI_overlay_value.set(True)
            self.ui.dataselection = 'seg_raw_fusion'
            self.ui.sliceviewerframe.updateslice()
            if self.ui.roi[self.ui.s][roi].data['WT'] is None:
                self.layer_callback(layer='T2 hyper')
            else:
                self.layer_callback(layer='ET')


        return None
    
    # records button press coords in a new ROI object
    def createROI(self,x,y,slice):
        compartment = self.layer.get()
        roi = ROI(x,y,slice,compartment=compartment)
        self.ui.roi[self.ui.s].append(roi)
        self.currentroi.set(self.currentroi.get() + 1)
        self.updateROIData()
        self.update_roinumber_options()

    # adds button press coords for second layer (ie 'WT' after 'ET')
    def updateROI(self,event):
        compartment = self.layer.get()
        roi = self.ui.roi[self.ui.s][self.ui.get_currentroi()]
        roi.coords[compartment]['x'] = int(event.xdata)
        roi.coords[compartment]['y'] = int(event.ydata)
        roi.coords[compartment]['slice'] = self.ui.get_currentslice()
        self.updateROIData()

    # main method for creating a smoothed final ROI from a raw BLAST segmentation
    # needs tidyup
    def closeROI(self,metmaskstack,currentslice,do3d=True):

        # process matching ROI to selected BLAST layer
        m = self.layer.get()
        s = self.ui.s
        roi = self.ui.get_currentroi()
        xpos = self.ui.roi[s][roi].coords[m]['x']
        ypos = self.ui.roi[s][roi].coords[m]['y']
        roislice = self.ui.roi[s][roi].coords[m]['slice']

        if m == 'T2 hyper': # difference in naming convention between BLAST and final segmentation
            m = 'WT'
        # a quick config for ET, WT smoothing
        mlist = {'ET':{'threshold':4,'dball':10,'dcube':2},
                    'WT':{'threshold':1,'dball':10,'dcube':2}}
        mask = (metmaskstack & mlist[m]['threshold'] == mlist[m]['threshold']).astype('double')

        # dust operation gets rid of a lot of small noise elements in the raw BLAST segmentation
        # connectivity = 6 is the minimum, again to filter out noise and tenuously connected regions
        CC_labeled = cc3d.dust(mask,connectivity=6,threshold=100,in_place=False)
        CC_labeled = cc3d.connected_components(CC_labeled,connectivity=6)
        stats = cc3d.statistics(CC_labeled)

        objectnumber = CC_labeled[roislice,ypos,xpos]
        objectmask = (CC_labeled == objectnumber).astype('double')
        # other than cc3d, currently there is no additional processing on ET or WT
        self.ui.roi[s][roi].data[m] = objectmask.astype('uint8')

        
        if m == 'ET': # calculate TC, a smoothed/filled version of ET
            objectmask_closed = np.zeros(np.shape(self.ui.data[self.ui.s].dset[self.ui.dataselection][self.ui.chselection]['d'+m])[1:])
            objectmask_final = np.zeros(np.shape(self.ui.data[self.ui.s].dset[self.ui.dataselection][self.ui.chselection]['d'+m])[1:])

            # thisBB = BB.BoundingBox[objectnumber,:,:]
            thisBB = stats['bounding_boxes'][objectnumber]

            # step 1. binary closing
            # for 3d closing, gpu is faster. use cucim library
            if do3d:
                se = ball(mlist[m]['dball'])
                start = time.time()
                objectmask_cp = cp.array(objectmask)
                se_cp = cp.array(se)
                # TODO: iterate binary_closing?
                if os.name == 'posix':
                    close_object_cucim = cucim_binary_closing(objectmask_cp,footprint=se_cp)
                    objectmask_closed = np.array(close_object_cucim.get())
                elif os.name == 'nt':
                    close_object_cupy = cupy_binary_closing(objectmask_cp,se_cp)
                    objectmask_closed = np.array(close_object_cupy.get())            
                end = time.time()
                print('binary closing time = {:.2f} sec'.format(end-start))
                # use cupy library.
                if False:
                    start = time.time()
                    objectmask_cp = cp.array(objectmask)
                    se_cp = cp.array(se)
                    close_object_cupy = cupy_binary_closing(objectmask_cp,se_cp)
                    close_object = np.array(close_object_cupy.get())
                    end = time.time()
                    print('time = {}'.format(end-start))
            else:
                # 2d preview mode. not sure if this is going to be useful might phase it out
                # cpu only. use skimage library
                # if a final 3d seg already exists, use it?
                # if self.ui.data[m] is not None:
                #     objectmask_final = self.ui.data[m]
                # else:
                se = disk(mlist[m]['dball'])
                start = time.time()
                close_object = binary_closing(objectmask[currentslice,:,:],se)
                end = time.time()
                print('binary closing time = {}'.format(end-start))
                # use scipy
                if False:
                    start = time.time()
                    close_object_scipy = scipy_binary_closing(objectmask,structure=se)
                    end = time.time()
                    print('binary closing time = {}'.format(end-start))
                objectmask_closed[currentslice,:,:] = close_object

            # step 2. flood fill
            # for small kernel, don't need gpu
            # TODO: does ET need anything further. otherwise, tc has dilation that et does not
            if do3d:
                se2 = cube(mlist[m]['dcube'])
                objectmask_filled = binary_dilation(objectmask_closed,se2)
                seed = np.ones_like(objectmask_filled)*255
                seed[:,:,0] = 0
                seed[:,:,-1] = 0
                seed[:,0,:] = 0
                seed[:,-1,:] = 0
                seed[0,:,:] = 0
                seed[-1,:,:] = 0
                try:
                    objectmask_filled = reconstruction(seed,objectmask_filled,method='erosion')
                except ValueError as e:
                    print(e)
                    print('Try increasing the threshold to have fewer pixels in the BLAST mask')
                objectmask_final = objectmask_filled.astype('int')
                self.ui.roi[s][roi].data['TC'] = objectmask_final.astype('uint8')
            else: # 2d mode probably not used anymore
                se2 = square(mlist[m]['dcube'])
                objectmask_filled = binary_dilation(objectmask_closed[currentslice,:,:],se2)
                objectmask_filled = flood_fill(objectmask_filled,(ypos,xpos),True)
                objectmask_final[currentslice,:,:] = objectmask_filled.astype('int')     
                self.ui.roi[s][roi].data['TC'] = objectmask_final.astype('uint8')

            # step 3. TC contouring option. not recently updated.
            if do3d:
                objectmask_contoured = {}
                for sl in range(self.config.ImageDim[0]):
                    objectmask_contoured[sl] = find_contours(objectmask_final[sl,:,:])
                self.ui.roi[s][roi].data['contour']['TC'] = objectmask_contoured
 
            # create a combined seg mask from the three layers
            # using nnunet convention for labels
            if self.ui.roi[s][roi].data['WT'] is None:
                self.ui.roi[s][roi].data['seg'] = 4*self.ui.roi[s][roi].data['ET'] + \
                                                    2*self.ui.roi[s][roi].data['TC']
            else:
                self.ui.roi[s][roi].data['seg'] = 4*self.ui.roi[s][roi].data['ET'] + \
                                                    2*self.ui.roi[s][roi].data['TC'] + \
                                                    1*self.ui.roi[s][roi].data['WT']
                self.ui.roi[s][roi].status = True # ie ROI has both compartments selected                                                    

        elif m == 'WT': # WT gets no additional processing. just create a combined seg mask from the three layers
                        # nnunet convention for labels
            if self.ui.roi[s][roi].data['ET'] is None:
                self.ui.roi[s][roi].data['seg'] = 1*self.ui.roi[s][roi].data['WT']
            else:
                # update WT based on smoothing for TC
                self.ui.roi[s][roi].data['WT'] = self.ui.roi[s][roi].data['WT'] | self.ui.roi[s][roi].data['TC']
                # if 'ET' exists but 'WT' didn't, then have to rebuild the combined mask because of the dummy +1
                self.ui.roi[s][roi].data['seg'] = 4*self.ui.roi[s][roi].data['ET'] + \
                                                    2*self.ui.roi[s][roi].data['TC'] + \
                                                    1*self.ui.roi[s][roi].data['WT']
                self.ui.roi[s][roi].status = True # ROI has both compartments selected
            # WT contouring. not updated lately.
            objectmask_contoured = {}
            for sl in range(self.config.ImageDim[0]):
                objectmask_contoured[sl] = find_contours(self.ui.roi[s][roi].data['WT'][sl,:,:])
            self.ui.roi[s][roi].data['contour']['WT'] = objectmask_contoured

        return None

    # for exporting BLAST segmentations.
    def saveROI(self,roi=None,outputpath=None,sam=False):
        if outputpath is None:
            outputpath = self.ui.caseframe.casedir
        if roi is None:
            roilist = list(map(int,self.roilist))
        else:
            if type(roi) == list:
                roilist = roi
            else:
                roilist = [roi]

        # temp output just for sam segmentation
        # using tmpfiles so ptorch not needed in the blast env
        if sam:
            fileroot = os.path.join(self.ui.data[self.ui.s].studydir,'sam')
            if not os.path.exists(fileroot):
                os.mkdir(fileroot)
            else:
                shutil.rmtree(fileroot)
                os.mkdir(fileroot)
            roisuffix = ''
            for ch,img in zip(['t1+','flair'],['ET','WT']):
                for roi in roilist: # assuming just one roi for sam
                    if len(roilist) > 1:
                        roisuffix = '_roi'+roi
                    rref = self.ui.roi[self.ui.s][roi].data[img]
                    dref = self.ui.data[self.ui.s].dset['raw'][ch]['d']
                    affine_bytes = self.ui.data[self.ui.s].dset['raw'][ch]['affine'].tobytes()
                    affine_bytes_str = str(affine_bytes)
                    if dref is not None:
                        for slice in range(self.ui.sliceviewerframe.dim[0]):
                            if len(np.where(rref[slice])[0]):
                                outputfilename = os.path.join(fileroot,'mask_' + str(slice) + '_' + ch + '.png')
                                plt.imsave(outputfilename,rref[slice],cmap='gray',)
                                outputfilename = os.path.join(fileroot,'slice_' + str(slice) + '_' + ch +'.png')
                                meta = PIL.PngImagePlugin.PngInfo()
                                meta.add_text('slicedim',str(self.ui.sliceviewerframe.dim[0]))
                                meta.add_text('affine',affine_bytes_str)
                                plt.imsave(outputfilename,dref[slice],cmap='gray',pil_kwargs={'pnginfo':meta})

        # BLAST image file outputs.
        fileroot = self.ui.data[self.ui.s].studydir
        for img in ['seg','ET','TC','WT']:
            for roi in roilist:
                if len(roilist) > 1:
                    roisuffix = '_roi'+roi
                    outputfilename = os.path.join(fileroot,'{}_{}_blast_processed.nii'.format(img,roisuffix))
                else:
                    outputfilename = os.path.join(fileroot,'{}_blast_processed.nii'.format(img))
                if self.ui.roi[self.ui.s][int(roi)].data[img] is not None:
                    self.ui.data[self.ui.s].writenifti(self.ui.roi[self.ui.s][int(roi)].data[img],
                                                       outputfilename,
                                                       affine=self.ui.data[self.ui.s].dset['raw']['t1+']['affine'])

        # also output to pickle
        if False:
            sdict = {}
            bdict = {}
            for i,r in enumerate(self.ui.roi[self.ui.s][1:]): # skip dummy 
                sdict['roi'+str(i)] = r.stats
                bdict['roi'+str(i)] = dict((k,r.data[k]) for k in ('ET','TC','WT','blast','raw'))

            filename = fileroot+'_stats.pkl'
            with open(filename,'ab') as fp:
                pickle.dump((sdict,bdict),fp)

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
        

    # back-copy an existing ROI and overlay from current dataset back into the current roi. 
    # not sure if still needed though
    def updateROIData(self):
        for k in ['raw','seg_fusion']:
            if k == 'raw': # refernece only
                self.ui.roi[self.ui.s][self.ui.currentroi].data[k] = self.ui.data[self.ui.s].dset[k]
            else: 
                self.ui.roi[self.ui.s][self.ui.currentroi].data[k] = copy.deepcopy(self.ui.data[self.ui.s].dset[k])

    # calculate the combined mask from separate layers
    def updateBLAST(self,layer=None):
        s = self.ui.s
        # record slider values
        if layer is None:
            layer = self.layer.get()
        for sl in ['t12','flair','bc']:
            self.ui.blastdata[s]['blast']['params'][layer][sl] = self.thresholds[layer][sl].get()

        if all(self.ui.blastdata[s]['blast'][x] is not None for x in ['ET','T2 hyper']):
            # self.ui.data['seg_raw'] = self.ui.blastdata['blast']['ET'].astype('int')*2 + (self.ui.blastdata['blast']['T2 hyper'].astype('int'))
            self.ui.data[s].dset['seg_raw'][self.ui.chselection]['d'] = (self.ui.blastdata[s]['blast']['T2 hyper'].astype('uint8'))
            et = np.where(self.ui.blastdata[s]['blast']['ET'])
            self.ui.data[s].dset['seg_raw'][self.ui.chselection]['d'][et] += 4
        elif self.ui.blastdata[s]['blast']['ET'] is not None:
            self.ui.data[s].dset['seg_raw'][self.ui.chselection]['d'] = self.ui.blastdata[s]['blast']['ET'].astype('uint8')*4
        elif self.ui.blastdata[s]['blast']['T2 hyper'] is not None:
            self.ui.data[s].dset['seg_raw'][self.ui.chselection]['d'] = self.ui.blastdata[s]['blast']['T2 hyper'].astype('uint8')

    # forward-copy certain results from the BLAST ROI to the main dataset
    # may need further work
    def updateData(self,updatemask=False):
        s = self.ui.s
        # anything else to copy??  'seg_raw_fusion_d','seg_raw','blast','seg_raw_fusion'
        layer = self.layer.get()
        for dt in ['seg_fusion']:
            for ch in [self.ui.chselection,'flair']:
                self.ui.data[s].dset[dt][ch]['d'] = copy.deepcopy(self.ui.roi[s][self.ui.currentroi].data[dt][ch])
        for dt in ['ET','WT']:
            self.ui.data[s].mask[dt+'blast']['d'] = copy.deepcopy(self.ui.roi[s][self.ui.currentroi].data[dt])
            self.ui.data[s].mask[dt+'blast']['ex'] = True
            if self.ui.sliceviewerframes['overlay'] is not None:
                self.ui.sliceviewerframes['overlay'].maskdisplay_button['blast'].configure(state='active')
            if updatemask and False:
                # by this option, a BLAST segmentation could overwrite the current UI mask directly as a convenience.
                # otherwise, it will be done in separate step from the Overlay sliceviewer. 
                # would better need a further checkbox on the GUI for this auto option
                self.ui.data[s].mask[dt]['d'] = copy.deepcopy(self.ui.roi[self.ui.currentroi].data[dt])
        self.updatesliders()

    # eliminate latest ROI if there are multiple ROIs in current case
    def clearROI(self):
        n = len(self.ui.roi[self.ui.s])
        if n>1:    
            self.ui.roi[self.ui.s].pop(self.ui.currentroi)
            n -= 1
            if self.ui.currentroi > 1 or n==1:
                # new current roi is decremented as an arbitrary choice
                # or if all rois are now gone
                self.currentroi.set(self.currentroi.get()-1)
            self.update_roinumber_options()
            if n > 1:
                self.roinumber_callback()
            if n==1:
                self.resetROI()
                self.ui.updateslice()

    # eliminate all ROIs, ie for loading another case
    def resetROI(self):
        self.currentroi.set(0)
        self.ui.roi[self.ui.s] = [0]
        self.ui.roiframe.finalROI_overlay_value.set(False)
        self.ui.roiframe.enhancingROI_overlay_value.set(False)
        self.ui.roiframe.layertype.set('blast')
        self.ui.roiframe.layer.set('ET')
        if self.ui.chselection in ['t1+','flair']:
            for l in ['ET','T2 hyper']:
                for sl in ['t12','flair','bc']:
                    self.thresholds[l][sl].set(self.ui.config.thresholddefaults[sl])
                    self.updatesliderlabel(l,sl)
            self.update_roinumber_options()

    def append_roi(self,d):
        for k,v in d.items():
            if isinstance(v,dict):
                self.append_roi(d)
            else:
                v.append(0)

    # not recently updated.
    def ROIstats(self):
        
        roi = self.ui.get_currentroi()
        s = self.ui.s
        data = self.ui.roi[s][roi].data
        for t in ['ET','TC','WT']:
            # check for a complete segmentation
            if t not in data.keys():
                continue
            elif data[t] is None:
                continue
            self.ui.roi[s][roi].stats['vol'][t] = len(np.where(data[t])[0])

            if self.ui.data['label'] is not None:
                sums = data['manual_'+t] + data[t]
                subs = data['manual_'+t] - data[t]
                        
                TP = len(np.where(sums == 2)[0])
                FP = len(np.where(subs == -1)[0])
                TN = len(np.where(sums == 0)[0])
                FN = len(np.where(subs == 1)[0])

                self.ui.roi[s][roi].stats['spec'][t] = TN/(TN+FP)
                self.ui.roi[s][roi].stats['sens'][t] = TP/(TP+FN)
                self.ui.roi[s][roi].stats['dsc'][t] = 1-dice(data['manual_'+t].flatten(),data[t].flatten()) 

                # Calculate volumes
                self.ui.roi[s][roi].stats['vol']['manual_'+t] = len(np.where(data['manual_'+t])[0])

    # tumour segmenation by SAM
    def segment_sam(self,dpath=None,sam='SAM'):
        print('SAM segment tumour')
        if dpath is None:
            dpath = os.path.join(self.ui.data[self.ui.s].studydir,'sam')
            if not os.path.exists(dpath):
                os.mkdir(dpath)

        if os.name == 'posix':
            if sam == 'medSAM':
                command = 'conda run -n ptorch python scripts/medsam.py  --checkpoint /media/jbishop/WD4/brainmets/sam/medsam_vit_b.pth '
                command += ' --input ' + self.ui.caseframe.casedir
                command += ' --output ' + self.ui.caseframe.casedir
                command += ' --model-type vit_b'
            elif sam == 'SAM':
                command = 'conda run -n ptorch python scripts/sam.py  --checkpoint /media/jbishop/WD4/brainmets/sam/sam_vit_b_01ec64.pth '
                command += ' --input ' + self.ui.caseframe.casedir
                command += ' --output ' + self.ui.caseframe.casedir
                command += ' --model-type vit_b'
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
            if os.path.isdir(os.path.expanduser('~')+'\\anaconda3\envs\\pytorch118_310'):
                envpath = os.path.expanduser('~')+'\\anaconda3\envs\\pytorch118_310'
            elif os.path.isdir(os.path.expanduser('~')+'\\.conda\envs\\pytorch118_310'):
                envpath = os.path.expanduser('~')+'\\.conda\envs\\pytorch118_310'
            else:
                raise FileNotFoundError('pytorch118_310')

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
                
        if False:
            os.remove(os.path.join(dpath,sfile))

        return 