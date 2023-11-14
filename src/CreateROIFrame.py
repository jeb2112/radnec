import os,sys
import numpy as np
import pickle
import copy
import logging
import time
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import SimpleITK as sitk
from skimage.morphology import disk,square,binary_dilation,binary_closing,flood_fill,ball,cube,reconstruction
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
from src.CreateFrame import CreateFrame
from src.ROI import ROI

# contains various ROI methods and variables
class CreateROIFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)

        self.buttonpress_id = None # temp var for keeping track of button press event
        self.finalROI_overlay_value = tk.BooleanVar(value=False)
        self.enhancingROI_overlay_value = tk.BooleanVar(value=False)
        self.currentt1threshold = tk.DoubleVar(value=self.ui.config.T1default)
        self.currentt2threshold = tk.DoubleVar(value=self.ui.config.T2default)
        self.currentbcsize = tk.DoubleVar(value=self.ui.config.BCdefault[0])
        self.currentbcT1size = tk.DoubleVar(value=self.ui.config.BCdefault[0])
        self.currentbcT2size = tk.DoubleVar(value=self.ui.config.BCdefault[1])
        self.layerlist = {'blast':['ET','T2 hyper'],'seg':['ET','TC','WT','all']}
        self.layer = tk.StringVar(value='ET')
        self.layerROI = tk.StringVar(value='ET')
        self.layertype = tk.StringVar(value='blast')
        self.currentroi = tk.IntVar(value=0)
        self.roilist = []

        ########################
        # layout for the buttons
        ########################

        self.frame.grid(column=2,row=3,rowspan=5,sticky='e')

        # ROI buttons
        enhancingROI_label = ttk.Label(self.frame,text='overlay on/off')
        enhancingROI_label.grid(row=1,column=0,sticky='e')
        enhancingROI_overlay = ttk.Checkbutton(self.frame,text='',
                                               variable=self.enhancingROI_overlay_value,
                                               command=self.enhancingROI_overlay_callback)
        enhancingROI_overlay.grid(row=1,column=1,sticky='w')

        # enhancing layer choice
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
        self.layerROImenu.config(width=6)
        self.layerROImenu.grid(row=0,column=3,sticky='w')

        # n'th roi number choice
        roinumberlabel = ttk.Label(self.frame,text='ROI number:')
        roinumberlabel.grid(row=0,column=4,sticky='w')
        self.currentroi.trace_add('write',self.set_currentroi)
        self.roinumbermenu = ttk.OptionMenu(self.frame,self.currentroi,*self.roilist,command=self.roinumber_callback)
        self.roinumbermenu.config(width=2)
        self.roinumbermenu.grid(row=0,column=5,sticky='w')
        self.roinumbermenu.configure(state='disabled')

        # save ROI button
        saveROI = ttk.Button(self.frame,text='save ROI',command = self.saveROI)
        saveROI.grid(row=1,column=5,sticky='w')

        # clear ROI button
        clearROI = ttk.Button(self.frame,text='clear ROI',command = self.clearROI)
        clearROI.grid(row=1,column=4,sticky='w')
        self.frame.update()

        ########################
        # layout for the sliders
        ########################

        # t1 slider
        self.t1sliderframe = ttk.Frame(self.frame,padding='0')
        self.t1sliderframe.grid(column=0,row=2,columnspan=7,sticky='e')
        t1label = ttk.Label(self.t1sliderframe, text='T1')
        t1label.grid(column=0,row=0,sticky='w')
        self.t1slider = ttk.Scale(self.t1sliderframe,from_=-4,to=4,variable=self.currentt1threshold,state='disabled',
                                  length='3i',command=self.updatet1label,orient='horizontal')
        self.t1slider.grid(column=1,row=0,sticky='e')
        self.t1sliderlabel = ttk.Label(self.t1sliderframe,text=self.currentt1threshold.get())
        self.t1sliderlabel.grid(column=2,row=0,sticky='e')

        # t2 slider
        t2label = ttk.Label(self.t1sliderframe, text='T2')
        t2label.grid(column=0,row=1,stick='w')
        self.t2slider = ttk.Scale(self.t1sliderframe,from_=-4,to=4,variable=self.currentt2threshold,state='disabled',
                                  length='3i',command=self.updatet2label,orient='horizontal')
        self.t2slider.grid(column=1,row=1,sticky='e')
        self.t2sliderlabel = ttk.Label(self.t1sliderframe,text=self.currentt2threshold.get())
        self.t2sliderlabel.grid(column=2,row=1,sticky='e')

        #brain cluster slider
        bclabel = ttk.Label(self.t1sliderframe,text='b.c.')
        bclabel.grid(column=0,row=2,sticky='w')
        self.bcslider = ttk.Scale(self.t1sliderframe,from_=0.0,to=4,variable=self.currentbcsize,state='disabled',
                                  length='3i',command=self.updatebclabel,orient='horizontal')
        self.bcslider.grid(column=1,row=2,sticky='e')
        self.bcsliderlabel = ttk.Label(self.t1sliderframe,text='{:.1f}'.format(self.currentbcsize.get()))
        self.bcsliderlabel.grid(row=2,column=2,sticky='e')

        # bind ROI select callbacks
        self.ui.sliceviewerframe.canvas.get_tk_widget().bind('<Enter>',self.selectROI)
        self.ui.sliceviewerframe.canvas.get_tk_widget().bind('<Leave>',self.resetCursor)


    #############
    # ROI methods
    ############# 

    # methods for BLAST layer options menu
    def layer_callback(self,layer=None,updateslice=True,updatedata=True,overlay=True):
        self.ui.sliceviewerframe.updatewl_fusion()

        if layer is None:
            layer = self.layer.get()
        else:
            self.layer.set(layer)
        self.ui.currentlayer = layer
        roi = self.ui.get_currentroi()

        # when switching layers, slider values switch but no need to run re-blast immediately. 
        # T1 slider is not used for T2 hyper.
        self.updatesliders()
        if layer == 'T2 hyper':
            self.t1slider.configure(state='disabled')
            # self.updatebcsize(self.currentbcT2size.get(),blast=False)
        else:
            self.t1slider.configure(state='active')

        # generate a new overlay
        # TODO: check for existing instead of re-generating
        # in blast mode, overlays are stored in main ui data, and are not associated with a ROI yet ( ie until create or update ROI event)
        if overlay:
            self.ui.data['seg_raw_fusion'] = generate_overlay(self.ui.data['raw'],self.ui.data['seg_raw'],layer,
                                                                    overlay_intensity=self.config.OverlayIntensity)
            self.ui.data['seg_raw_fusion_d'] = copy.copy(self.ui.data['seg_raw_fusion'])

        if updateslice:
            self.ui.updateslice()

    # and ROI layer options menu
    def layerROI_callback(self,layer=None,updateslice=True,updatedata=True):
        self.ui.sliceviewerframe.updatewl_fusion()

        if layer is None:
            layer = self.layerROI.get()
        else:
            self.layerROI.set(layer)
        self.ui.currentROIlayer = self.layerROI.get()
        roi = self.ui.get_currentroi()

        # a convenience reference
        data = self.ui.roi[roi].data
        # in seg mode, the context is an existing ROI, so the overlays are first stored directly in the ROI dict
        # then also copied back to main ui data
        # TODO: check mouse event, versus layer_callback called by statement
        data['seg_fusion'] = generate_overlay(self.ui.data['raw'],data['seg'],layer,
                                                            overlay_intensity=self.config.OverlayIntensity)
        data['seg_fusion_d'] = copy.copy(data['seg_fusion'])
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

    # methods for roi number choice menu
    def roinumber_callback(self,item=None):
        self.ui.set_currentroi()
        # reference or copy
        self.updateData()
        self.layer_callback(updatedata=False)
        # current layer doesn't necessarily match data['seg_fusion_d'] on roi switch
        # self.layer.set(self.layerlist[self.layertype.get()][0])
        self.ui.updateslice()
        return
    
    def update_roinumber_options(self,n=None):
        if n is None:
            n = len(self.ui.roi)
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

    # updates blast segmentation
    def updatet1threshold(self,event=None,currentslice=True):
        self.enhancingROI_overlay_value.set(True)
        # for now, this event reverts to BLAST preview mode and will not directly reprocess the final segmentation
        if self.finalROI_overlay_value.get() == True:
            self.finalROI_overlay_value.set(False)
            self.enhancingROI_overlay_callback()

        # force recalc of gates
        layer = self.layer.get()
        self.ui.data['blast']['gates'][layer] = None
        # self.ui.data['blast']['gates'][2] = None
        self.ui.runblast(currentslice=currentslice)
        self.t1sliderlabel['text'] = '{:.1f}'.format(self.currentt1threshold.get())

    # updates the text field showing the value during slider drag
    def updatet1label(self,event=None):
        self.t1sliderlabel['text'] = '{:.1f}'.format(self.currentt1threshold.get())

    def updatet2threshold(self,event=None,currentslice=True):
        self.enhancingROI_overlay_value.set(True)
        if self.finalROI_overlay_value.get() == True:
            self.finalROI_overlay_value.set(False)
            self.enhancingROI_overlay_callback()
        # force recalc of gates
        layer = self.layer.get()
        self.ui.data['blast']['gates'][layer] = None
        # self.ui.data['blast']['gates'][2] = None
        self.ui.runblast(currentslice=currentslice)
        self.t2sliderlabel['text'] = '{:.1f}'.format(self.currentt2threshold.get())
        # ie not using this workflow presently
        if self.finalROI_overlay_value.get() == True:
            self.ROIclick(do3d=True)
        return
    
    def updatet2label(self,event=None):
        self.t2sliderlabel['text'] = '{:.1f}'.format(self.currentt2threshold.get())

    def updatebcsize(self,event=None):
        self.enhancingROI_overlay_value.set(True)
        if self.finalROI_overlay_value.get() == True:
            self.finalROI_overlay_value.set(False)
            self.enhancingROI_overlay_callback()
        layer = self.layer.get()
        self.ui.data['blast']['gates']['brain '+layer] = None
        self.ui.runblast(currentslice=True)
        self.updatebclabel()
        return

    def updatebclabel(self,event=None):
        currentbcsize = self.ui.get_currentbcsize()
        self.bcsliderlabel['text'] = '{:.1f}'.format(currentbcsize)

    def updatesliders(self):
        layer = self.layer.get()
        self.currentt1threshold.set(self.ui.data['blast']['params'][layer]['t1'])
        self.updatet1label()
        self.currentt2threshold.set(self.ui.data['blast']['params'][layer]['t2'])
        self.updatet2label()
        self.currentbcsize.set(self.ui.data['blast']['params'][layer]['bc'])
        self.updatebclabel()
    
    def finalROI_overlay_callback(self,event=None):
        if self.finalROI_overlay_value.get() == False:
            self.ui.dataselection = 'raw'
            self.ui.data['raw'] = copy.deepcopy(self.ui.data['raw_copy'])
            self.ui.updateslice()
        else:
            self.enhancingROI_overlay_value.set(False)
            self.ui.dataselection = 'seg_fusion_d'
            # self.update_layermenu_options('seg')
            self.ui.updateslice(wl=True)

    def enhancingROI_overlay_callback(self,event=None):
        if self.enhancingROI_overlay_value.get() == False:
            self.ui.dataselection = 'raw'
            self.ui.data['raw'] = copy.deepcopy(self.ui.data['raw_copy'])
            self.ui.updateslice()
        else:
            self.finalROI_overlay_value.set(False)
            self.ui.dataselection = 'seg_raw_fusion_d'
            # self.update_layermenu_options('blast')
            self.ui.updateslice(wl=True)

    def enhancingROI_callback(self,event=None):
        self.finalROI_overlay_value.set(False)
        self.enhancingROI_overlay_value.set(True)
        # self.update_layermenu_options('blast')
        self.ui.runblast()

    def selectROI(self,event=None):
        if self.enhancingROI_overlay_value.get(): # only activate cursor in BLAST mode
            self.buttonpress_id = self.ui.sliceviewerframe.canvas.callbacks.connect('button_press_event',self.ROIclick)
            self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='crosshair')
        return None
    
    def resetCursor(self,event=None):
        self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='watch')
        self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()
    
    def ROIclick(self,event=None,do3d=True):
        if event:
            if event.button > 1: # ROI selection on left mouse only
                return
        self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='watch')
        self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()
        if event:
            # print(event.xdata,event.ydata)
            # need check for inbounds
            if event.xdata < 0 or event.ydata < 0:
                return None
            else:
                roi = self.ui.get_currentroi()
                # if current roi has segmentations for both compartments, start a new ROI
                # otherwise if only 1 segmentation is present and the mouse click is for that
                # same compartment, it will be updated.
                if roi > 0:
                    if self.ui.roi[roi].data['ET'] is not None and self.ui.roi[roi].data['WT'] is not None:
                        self.createROI(int(event.xdata),int(event.ydata),self.ui.get_currentslice(),compartment=self.layer.get())
                    else:
                        self.updateROI(event)
                else:
                    self.createROI(int(event.xdata),int(event.ydata),self.ui.get_currentslice(),compartment=self.layer.get())
            
        roi = self.ui.get_currentroi()
        self.closeROI(self.ui.roi[roi].data['seg_raw'],self.ui.get_currentslice(),do3d=do3d)
        # update layer menu
        self.update_layermenu_options(self.ui.roi[roi])

        self.ROIstats()
        fusionstack = np.zeros((2,155,240,240))
        # note some duplicate calls to generate_overlay should be removed
        fusionstack = generate_overlay(self.ui.data['raw'],self.ui.roi[roi].data['seg'],self.ui.roiframe.layer.get(),
                                                    overlay_intensity=self.config.OverlayIntensity)
        self.ui.roi[roi].data['seg_fusion'] = fusionstack
        self.ui.roi[roi].data['seg_fusion_d'] = copy.copy(self.ui.roi[roi].data['seg_fusion'])
        # current roi populates data dict
        self.updateData()

        # if triggered by a button event
        if self.buttonpress_id:
            self.ui.sliceviewerframe.canvas.callbacks.disconnect(self.buttonpress_id)
            self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='')
            self.buttonpress_id = None
        self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='arrow')
        self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()

        # logic stays in BLAST mode if both ET and t2 hyper are not yet selected.
        roi = self.ui.get_currentroi()
        if self.ui.roi[roi].status:
            self.finalROI_overlay_value.set(True)
            self.enhancingROI_overlay_value.set(False)
            self.ui.dataselection = 'seg_fusion_d'
            self.layerROI_callback(layer='ET')
            # self.ui.sliceviewerframe.updateslice()
        else:
            self.finalROI_overlay_value.set(False)
            self.enhancingROI_overlay_value.set(True)
            self.ui.dataselection = 'seg_raw_fusion_d'
            self.ui.sliceviewerframe.updateslice()
            if self.ui.roi[roi].data['WT'] is None:
                self.layer_callback(layer='WT')
            else:
                self.layer_callback(layer='ET')

        return None
    
    def createROI(self,x,y,slice):
        compartment = self.layer.get()
        if compartment == 'T2 hyper': # difference in naming convention between BLAST and final segmentation
            compartment = 'WT'
        roi = ROI(x,y,slice,compartment=compartment)
        roi['blast_params'][compartment]['t1'] = self.t1slider.get()
        roi['blast_params'][compartment]['t2'] = self.t2slider.get()
        roi['blast_params'][compartment]['bc'] = self.ui.get_currentbcsize()
        self.ui.roi.append(roi)
        self.currentroi.set(self.currentroi.get() + 1)
        self.updateROIData()
        self.update_roinumber_options()

    def updateROI(self,event):
        compartment = self.layer.get()
        roi = self.ui.roi[self.ui.get_currentroi()]
        roi['blast_params'][compartment]['t1'] = self.t1slider.get()
        roi['blast_params'][compartment]['t2'] = self.t2slider.get()
        roi['blast_params'][compartment]['bc'] = self.ui.get_currentbcsize()
        if compartment == 'T2 hyper':
            compartment = 'WT'
        roi.coords[compartment]['x'] = int(event.xdata)
        roi.coords[compartment]['y'] = int(event.ydata)
        roi.coords[compartment]['slice'] = self.ui.get_currentslice()


    def closeROI(self,metmaskstack,currentslice,do3d=True):
        # this method needs tidy-up

        # process matching ROI to selected BLAST layer
        m = self.layer.get()
        if m == 'T2 hyper': # difference in naming convention between BLAST and final segmentation
            m = 'WT'
        roi = self.ui.get_currentroi()
        xpos = self.ui.roi[roi].coords[m]['x']
        ypos = self.ui.roi[roi].coords[m]['y']
        roislice = self.ui.roi[roi].coords[m]['slice']

        # a quick config for ET, WT smoothing
        mlist = {'ET':{'threshold':3,'dball':10,'dcube':2},
                    'WT':{'threshold':1,'dball':10,'dcube':2}}
        mask = (metmaskstack >= mlist[m]['threshold']).astype('double')

        # TODO: 2d versus 3d connected components?
        CC_labeled = cc3d.connected_components(mask,connectivity=26)
        stats = cc3d.statistics(CC_labeled)

        # objectnumber = CC_labeled(ypos,xpos,s.SliceNumber)
        objectnumber = CC_labeled[roislice,ypos,xpos]

        # objectmask = ismember(CC_labeled,objectnumber)
        objectmask = (CC_labeled == objectnumber).astype('double')
        # currently there is no additional processing on ET or WT
        self.ui.roi[roi].data[m] = objectmask.astype('uint8')

        # calculate tc
        if m == 'ET':
            objectmask_closed = np.zeros(np.shape(self.ui.data['raw'])[1:])
            objectmask_final = np.zeros(np.shape(self.ui.data['raw'])[1:])

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

            # self.ui.data[m] = close_object
            # se2 = strel('square',2) #added to imdilate
            # se2 = square(2)

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
                objectmask_filled = reconstruction(seed,objectmask_filled,method='erosion')
                objectmask_final = objectmask_filled.astype('int')
                self.ui.roi[roi].data['TC'] = objectmask_final.astype('uint8')
            else:
                se2 = square(mlist[m]['dcube'])
                objectmask_filled = binary_dilation(objectmask_closed[currentslice,:,:],se2)
                objectmask_filled = flood_fill(objectmask_filled,(ypos,xpos),True)
                objectmask_final[currentslice,:,:] = objectmask_filled.astype('int')     
                self.ui.roi[roi].data['TC'] = objectmask_final.astype('uint8')

            # update combined seg mask
            # nnunet convention for labels
            if self.ui.roi[roi].data['WT'] is None:
                # there is a dummy background of +1 for missing 'WT' mask here
                self.ui.roi[roi].data['seg'] = 1*self.ui.roi[roi].data['ET'] + \
                                                    2*self.ui.roi[roi].data['TC']
            else:
                self.ui.roi[roi].data['seg'] += 1*self.ui.roi[roi].data['ET'] + \
                                                    1*self.ui.roi[roi].data['TC']
                self.ui.roi[roi].status = True # ROI has both compartments selected
                                                    

        elif m == 'WT':
            # update combined seg mask
            # nnunet convention for labels
            if self.ui.roi[roi].data['ET'] is None:
                self.ui.roi[roi].data['seg'] = 1*self.ui.roi[roi].data['WT']
            else:
                # update WT based on smoothing for TC
                self.ui.roi[roi].data['WT'] = self.ui.roi[roi].data['WT'] | self.ui.roi[roi].data['TC']
                # if 'ET' exists but 'WT' didn't, then have to rebuild the combined mask because of the dummy +1
                self.ui.roi[roi].data['seg'] = 1*self.ui.roi[roi].data['ET'] + \
                                                    1*self.ui.roi[roi].data['TC'] + \
                                                    1*self.ui.roi[roi].data['WT']
                self.ui.roi[roi].status = True # ROI has both compartments selected


        return None
    
    def saveROI(self,roi=None):
        self.ui.endtime()
        # Save ROI data
        outputpath = self.ui.caseframe.casedir
        fileroot = os.path.join(outputpath,self.ui.caseframe.casefile_prefix + self.ui.caseframe.casename.get())
        filename = fileroot+'_stats.pkl'
        # t1mprage template? need to save separately?

        # BLAST outputs. combined ROI or separate? doing separate for now
        roisuffix = ''
        for img in ['seg','ET','TC','WT']:
            for roi in self.roilist:
                if len(self.roilist) > 1:
                    roisuffix = '_roi'+roi
                outputfilename = fileroot + '_blast_' + img + roisuffix + '.nii'
                self.WriteImage(self.ui.roi[int(roi)].data[img],outputfilename)
        # manual outputs. for now these have only one roi
        for img in ['manual_ET','manual_TC','manual_WT']:
            outputfilename = fileroot + '_' + img + '.nii'
            self.WriteImage(self.ui.data[img],outputfilename)
            # sitk.WriteImage(sitk.GetImageFromArray(self.ui.data[img]),fileroot+'_'+img+'.nii')

        sdict = {}
        bdict = {}
        msdict = {} 
        for i,r in enumerate(self.ui.roi):
            sdict['roi'+str(i)] = r.stats
            bdict['roi'+str(i)] = r.data
            msdict['roi'+str(i)] = {}
            mdict = msdict['roi'+str(i)]
            mdict['greengate_count'] = r.data['blast_params']['gatecount']['t2']
            mdict['redgate_count'] = r.data['blast_params']['gatecount']['t1']
            mdict['objectmask'] = r.data['ET']
            mdict['objectmask_filled'] = r.data['TC']
            mdict['manualmasket'] = r.data['manual_ET']
            mdict['manualmasktc'] = r.data['manual_TC']
            mdict['centreimage'] = 0
            mdict['specificity_et'] = r.stats['spec']['ET']
            mdict['sensitivity_et'] = r.stats['sens']['ET']
            mdict['dicecoefficient_et'] = r.stats['dsc']['ET']
            mdict['specificity_tc'] = r.stats['spec']['TC']
            mdict['sensitivity_tc'] = r.stats['sens']['TC']
            mdict['dicecoefficient_tc'] = r.stats['dsc']['TC']
            mdict['specificity_wt'] = r.stats['spec']['WT']
            mdict['sensitivity_wt'] = r.stats['sens']['WT']
            mdict['dicecoefficient_wt'] = r.stats['dsc']['WT']
            mdict['b'] = 0
            mdict['b2'] = 0
            mdict['manualmask_et_volume'] = r.stats['vol']['manual_ET']
            mdict['manualmask_tc_volume'] = r.stats['vol']['manual_TC']
            mdict['objectmask_filled_volume'] = r.stats['vol']['TC']
            mdict['cumulative_elapsed_time'] = r.stats['elapsed_time']

        with open(filename,'ab') as fp:
            pickle.dump((sdict,bdict),fp)
        # matlab compatible output
        filename = filename[:-3] + 'mat'
        with open(filename,'ab') as fp:
            savemat(filename,sdict,bdict)
        filename = filename[:-4] + '_origvarnames.mat'
        with open(filename,'ab') as fp:
            savemat(filename,msdict)

    # for now output only segmentations so uint8
    def WriteImage(self,img_arr,filename):
        img = sitk.GetImageFromArray(img_arr.astype('uint8'))
        writer = sitk.ImageFileWriter()
        writer.SetImageIO('NiftiImageIO')
        writer.SetFileName(filename)
        writer.Execute(img)
        return
        
    # def updateROI(self):
    #     # rerun segmentation
    #     self.ROIclick()
    #     self.ROIstats()
    #     # save current dataset into the current roi. 
    #     for k,v in self.ui.roi[self.ui.currentroi].data.items():
    #         if k != 'raw':
    #             v = copy.deepcopy(self.ui.data[k])
    #         else: # reference only
    #             v = self.ui.data[k]

    #     # self.ui.roi[self.ui.currentroi].data = copy.deepcopy(self.ui.data)

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
        self.ui.data['blast']['params'][layer]['t1'] = self.currentt1threshold.get()
        self.ui.data['blast']['params'][layer]['t2'] = self.currentt2threshold.get()
        self.ui.data['blast']['params'][layer]['bc'] = self.currentbcsize.get()

        if all(self.ui.data['blast'][x] is not None for x in ['ET','T2 hyper']):
            self.ui.data['seg_raw'] = self.ui.data['blast']['ET'].astype('int')*2 + (self.ui.data['blast'] ['T2 hyper'].astype('int'))
        elif self.ui.data['blast']['ET'] is not None:
            self.ui.data['seg_raw'] = self.ui.data['blast']['ET'].astype('int')*3
        elif self.ui.data['blast']['T2 hyper'] is not None:
            self.ui.data['seg_raw'] = self.ui.data['blast']['T2 hyper'].astype('int')

    def updateData(self):
        self.ui.data = copy.deepcopy(self.ui.roi[self.ui.currentroi].data)

    # eliminate one ROI if multiple ROIs in current case
    def clearROI(self):
        n = len(self.ui.roi)
        if n>1:    
            self.ui.roi.pop(self.ui.currentroi)
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
        self.ui.roi = [0]
        self.ui.roiframe.finalROI_overlay_value.set(False)
        self.ui.roiframe.enhancingROI_overlay_value.set(False)
        self.ui.roiframe.layertype.set('blast')
        self.ui.roiframe.layer.set('ET')
        self.ui.dataselection='raw'
        self.currentt1threshold.set(self.ui.config.T1default)
        self.updatet1label(event=None)
        self.currentt2threshold.set(self.ui.config.T2default)
        self.updatet2label(event=None)
        self.currentbcT1size.set(self.ui.config.BCdefault[0])
        self.currentbcT2size.set(self.ui.config.BCdefault[1])
        self.updatebclabel(event=None)
        self.update_roinumber_options()

    def append_roi(self,d):
        for k,v in d.items():
            if isinstance(v,dict):
                self.append_roi(d)
            else:
                v.append(0)

    def ROIstats(self):
        
        roi = self.ui.get_currentroi()
        data = self.ui.roi[roi].data
        # if roi > len(self.ui.roi):
        #     self.append_roi(self.ui.stats)
        for t in ['ET','TC','WT']:
            # check for a complete segmentation
            if t not in data.keys():
                return
            elif data[t] is None:
                return
            sums = data['manual_'+t] + data[t]
            subs = data['manual_'+t] - data[t]
                    
            TP = len(np.where(sums == 2)[0])
            FP = len(np.where(subs == -1)[0])
            TN = len(np.where(sums == 0)[0])
            FN = len(np.where(subs == 1)[0])

            # self.ui.stats['spec'][t][roi] = TN/(TN+FP)
            # self.ui.stats['sens'][t][roi] = TP/(TP+FN)
            # self.ui.stats['dice'][t][roi] = dice(data['manual_'+t].flatten(),data[t].flatten()) 
            self.ui.roi[roi].stats['spec'][t] = TN/(TN+FP)
            self.ui.roi[roi].stats['sens'][t] = TP/(TP+FN)
            self.ui.roi[roi].stats['dsc'][t] = 1-dice(data['manual_'+t].flatten(),data[t].flatten()) 

        # Calculate volumes
            self.ui.roi[roi].stats['vol']['manual_'+t] = len(np.where(data['manual_'+t])[0])
            self.ui.roi[roi].stats['vol'][t] = len(np.where(data[t])[0])

        # copy gate counts
            # self.ui.roi[roi].stats['gatecount']['t1'] = self.ui.roi[roi].data['blast']['gates'][3]
            # self.ui.roi[roi].stats['gatecount']['t2'] = self.ui.roi[roi].data['blast']['gates'][4]

