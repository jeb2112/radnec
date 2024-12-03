import tkinter as tk
from tkinter import ttk

import numpy as np
import re
import copy

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Ellipse

from src.CreateFrame import CreateFrame,Command
from src.roi.ROI import ROIBLAST,ROISAM,ROIPoint

##########################################################
# frame layout for the different ROI overlays in BLAST/SAM
# ########################################################

class CreateROIOverlayFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)


        # ROI buttons for raw BLAST segmentation
        enhancingROI_label = ttk.Label(self.frame,text='overlay on/off')
        enhancingROI_label.grid(row=1,column=0,sticky='e')
        enhancingROI_overlay = ttk.Checkbutton(self.frame,text='',
                                               variable=self.overlay_value['BLAST'],
                                               command=self.enhancingROI_overlay_callback)
        enhancingROI_overlay.grid(row=1,column=1,sticky='w')

        layerlabel = ttk.Label(self.frame,text='prompt:')
        layerlabel.grid(row=0,column=0,sticky='w')
        self.layer.trace_add('write',lambda *args: self.layer.get())
        self.layermenu = ttk.OptionMenu(self.frame,self.layer,self.layerlist['blast'][0],
                                        *self.layerlist['blast'],command=self.layer_callback)
        self.layermenu.config(width=7)
        self.layermenu.grid(row=0,column=1,sticky='w')

        # ROI buttons for final smoothed segmentation
        # this option will not be used in the SAM viewer
        finalROI_overlay = ttk.Checkbutton(self.frame,text='',
                                           variable=self.overlay_value['finalROI'],
                                           command=self.finalROI_overlay_callback)
        if False:
           finalROI_overlay.grid(row=1,column=3,sticky='w')
        layerlabel = ttk.Label(self.frame,text='ROI layer:')
        if False:
            layerlabel.grid(row=0,column=2,sticky='w')
        self.layerROI.trace_add('write',lambda *args: self.layerROI.get())
        self.layerROImenu = ttk.OptionMenu(self.frame,self.layerROI,self.layerlist['seg'][0],
                                           *self.layerlist['seg'],command=self.layerROI_callback)
        self.layerROImenu.config(width=4)
        if False:
            self.layerROImenu.grid(row=0,column=3,sticky='w')

        # ROI button for SAM segmentation
        SAM_overlay = ttk.Checkbutton(self.frame,text='',
                                           variable=self.overlay_value['SAM'],
                                           command=self.SAM_overlay_callback)
        SAM_overlay.grid(row=1,column=3,sticky='w')
        layerlabel = ttk.Label(self.frame,text='overlay:')
        layerlabel.grid(row=0,column=2,sticky='w')
        self.layerSAM.trace_add('write',lambda *args: self.layerSAM.get())
        self.layerSAMmenu = ttk.OptionMenu(self.frame,self.layerSAM,self.layerlist['sam'][0],
                                           *self.layerlist['sam'],command=self.layerSAM_callback)
        self.layerSAMmenu.config(width=4)
        self.layerSAMmenu.grid(row=0,column=3,sticky='w')


    # main method for handling ET versus WT selection in BLAST raw segmentation
    def layer_callback(self,layer=None,updateslice=True,updatedata=True,overlay=True):

        # if in the opposite mode, then switch same as if the checkbutton was used. 
        # but don't run the checkbutton callback because
        # don't yet have logic to check if the existing overlay is correct or
        # needs to be redone.
        # also if in ROI mode, then copy the relevant data back for BLAST mode.
        if self.overlay_value['finalROI'].get() == True:
            self.updateData()
        self.set_overlay('BLAST')
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
        if self.ui.function.get() != 'SAM': # not using sliders for now in SAM
            self.sliderframe[layer].lift()

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
                        self.ui.roiframe.generate_blast_overlay(self.ui.data[s].dset['raw'][ch]['d'],
                                                self.ui.data[s].dset['seg_raw'][self.ui.chselection]['d'],
                                                layer=layer,overlay_intensity=self.config.OverlayIntensity)
                    self.ui.data[s].dset['seg_raw_fusion'][ch]['ex'] = True
                    # self.ui.data[s].dset['seg_raw_fusion_d'][ch]['d'+layer] = copy.deepcopy(self.ui.data[s].dset['seg_raw_fusion'][self.ui.chselection]['d'+layer])

        if updateslice:
            self.ui.updateslice()

    # main method for handling ET,TC,WT selection in final ROI smoothed segmentation
    def layerROI_callback(self,layer=None,updateslice=True,updatedata=True):

        # switch roi context
        self.ui.roi = self.ui.rois['blast']
        self.update_roinumber_options()

        roi = self.ui.get_currentroi()
        if roi == 0:
            return
        # if in the opposite mode, then switch
        self.set_overlay('finalROI')
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
            data['seg_fusion'] = self.ui.roiframe.generate_blast_overlay(self.ui.data[self.ui.s].dset['raw'][self.ui.chselection]['d'],
                                                        data['seg'],contour=data['contour'],layer=layer,
                                                        overlay_intensity=self.config.OverlayIntensity)
        else:
            for ch in [self.ui.chselection,'flair']:
                data['seg_fusion'][ch] = self.ui.roiframe.generate_blast_overlay(self.ui.data[self.ui.s].dset['raw'][ch]['d'],
                                                                data['seg'],layer=layer,
                                                            overlay_intensity=self.config.OverlayIntensity)
        if updatedata:
            self.updateData()

        if updateslice:
            self.ui.updateslice()

        return
    
    # method for displaying SAM results
    def layerSAM_callback(self,layer=None,updateslice=True, updatedata=True):

        if layer is None:
            layer = self.layerROI.get()

        # switch roi context
        self.ui.roi = self.ui.rois['sam']
        self.update_roinumber_options()

        roi = self.ui.get_currentroi()
        if roi == 0:
            return
        if self.overlay_value['SAM'].get() == True:
            self.ui.dataselection = 'seg_fusion'
        else:
            self.ui.dataselection = 'raw'
        # a convenience reference
        data = self.ui.roi[self.ui.s][roi].data
        # in seg mode, the context is an existing ROI, so the overlays are first stored directly in the ROI dict
        # then also copied back to main ui data
        # TODO: check mouse event, versus layer_callback called by statement
        for ch in [self.ui.chselection]:
            data['seg_fusion'][ch] = self.ui.roiframe.generate_blast_overlay(self.ui.data[self.ui.s].dset['raw'][ch]['d'],
                                                            data['seg'],layer=layer,
                                                        overlay_intensity=self.config.OverlayIntensity)

        if updatedata:
            self.updateSAMData()

        if updateslice:
            # ie the most recent bbox from the list of bboxs. or maybe currentslice is already correct?
            self.ui.set_currentslice(self.ui.roi[self.ui.s][roi].bbox['slice'])
            self.ui.updateslice()
        
        # restore roi context
        self.ui.roi = self.ui.rois['blast']

        # set layer if necessary
        if layer != self.layerSAM.get():
            self.layerSAM.set(layer)
        
        return

    # convenience method
    def set_overlay(self,overlay=''):
        for k in self.overlay_value.keys():
            self.overlay_value[k].set(False)
        if len(overlay):
            self.overlay_value[overlay].set(True)        

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
