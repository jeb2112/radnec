import os,sys
import re
import numpy as np
import pickle
import copy
import logging
import time
import json
import shutil
import getpass
import subprocess
import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import _Mode
matplotlib.use('TkAgg')

from matplotlib.path import Path
from matplotlib.patches import Ellipse
from cProfile import Profile
from pstats import SortKey,Stats

import SimpleITK as sitk
import nibabel as nb
import PIL
from skimage.morphology import disk,square,binary_dilation,binary_closing,flood_fill,ball,cube,reconstruction
from skimage.measure import find_contours
from scipy.spatial.distance import dice,directed_hausdorff
from scipy.ndimage import binary_closing as scipy_binary_closing
from scipy.io import savemat
if os.name == 'posix':
    # from cucim.skimage.morphology import binary_closing as cucim_binary_closing
    from cupyx.scipy.ndimage import binary_closing as cucim_binary_closing
elif os.name == 'nt':
    from cupyx.scipy.ndimage import binary_closing as cupy_binary_closing
import cupy as cp
import cc3d

from src.OverlayPlots import *
from src.CreateFrame import CreateFrame,Command
from src.SegmentSam import segment_sam
from src.roi.ROI import ROIBLAST,ROISAM,ROIPoint
from src.roi.CreateROISliderFrame import CreateROISliderFrame
from src.roi.CreateROIPointFrame import CreateROIPointFrame
from src.roi.CreateROIOverlayFrame import CreateROIOverlayFrame
from src.SSHSession import SSHSession

# contains various ROI methods and variables for 'SAM' mode
class CreateSAMROIFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)

        self.buttonpress_id = None # temp var for keeping track of button press event
        self.overlay_value = {'BLAST':tk.BooleanVar(value=False),'finalROI':tk.BooleanVar(value=False),'SAM':tk.BooleanVar(value=False)}
        roidict = {'ET':{'t12':None,'flair':None,'bc':None},'T2 hyper':{'t12':None,'flair':None,'bc':None}}

        self.overlay_type = tk.IntVar(value=0)
        self.layerlist = {'blast':['ET','T2 hyper'],'seg':['ET','TC','WT','all'],'sam':['TC','WT']}
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

        self.roioverlayframe = CreateROIOverlayFrame(self.frame)

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

        self.pointframe = CreateROIPointFrame(self.frame)

        # currently not using sliders in the SAM viewer, just the point selection
        if False:
            self.sliderframe = CreateROISliderFrame()
 

    #############
    # ROI methods
    ############# 


    # methods for roi number choice menu
    def roinumber_callback(self,item=None):
        if self.overlay_value['BLAST'].get() == True:
            self.overlay_value['BLAST'].set(False)
            self.overlay_value['finalROI'].set(True)
            self.finalROI_overlay_callback()

        self.ui.set_currentroi()
        # reference or copy
        if self.overlay_value['BLAST'].get():
            self.layerROI_callback(updatedata=True)
        elif self.overlay_value['SAM'].get():
            self.layerSAM_callback(updatedata=True)
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
            self.overlay_value['finalROI'].set(False)

    def set_currentroi(self,var,index,mode):
        if mode == 'write':
            self.ui.set_currentroi()    


       
    # callback for final smoothed ROI on/off selection
    def finalROI_overlay_callback(self,event=None):

        # update roi context
        self.ui.roi = self.ui.rois['blast']
        self.update_roinumber_options()

        if self.overlay_value['finalROI'].get() == False:
            # base display, not data selection
            self.ui.dataselection = 'raw'
            if False: # no longer needed?
                self.ui.data[self.ui.dataselection][self.ui.chselection]['d'] = copy.deepcopy(self.ui.data[self.ui.chselection+'_copy']['d'])
            self.ui.updateslice()
        else:
            self.overlay_value['BLAST'].set(False)
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
        if self.overlay_value['finalROI'].get() == True:
            self.updateData()

        if self.overlay_value['BLAST'].get() == False:
            # base display, not data selection
            self.ui.dataselection = 'raw'
            if False:
                self.ui.data['raw'][self.ui.chselection]['d'] = copy.deepcopy(self.ui.data['t1+_copy']['d'])
            self.ui.updateslice()

        else:
            self.set_overlay('BLAST')
            self.ui.dataselection = 'seg_raw_fusion'
            self.ui.updateslice(wl=True)

    def SAM_overlay_callback(self):
        if self.overlay_value['SAM'].get() == False:
            self.ui.dataselection = 'raw'
            self.ui.updateslice()
        else:
            self.set_overlay('SAM')
            # currently have only one 'seg_fusion' overlay image, so have to regenerate each time
            # switching between SAM and finalROI
            self.layerSAM_callback()
        return

    # creates a ROI selection button press event
    def selectROI(self,event=None):
        if self.selectPointstate:
            self.selectPoint()
        # furthermore, clear the points list
        self.ui.reset_pt()

        if self.overlay_value['BLAST'].get(): # only activate cursor in BLAST mode
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
    # do3d is a dummy arg that is only passed on to closeROI()
    # do2d is a flag indicating the method is called during points clicked
    # for assembly of the BLAST ROI. 
    # can be called a mouse event or a pair of coords supplied separately from mouse
    def ROIclick(self,event=None,coords=None,do2d=False,do3d=True):
        if event:
            if event.button > 1: # ROI selection on left mouse only
                return
            # in the new workflow, BLAST roi is not being depicted, so this check no longer applies
            if self.overlay_value['BLAST'].get() == False: # no selection if BLAST mode not active
                pass
                # return
            
        # self.ui.sliceviewerframe.canvas.widgetlock.release(self.ui.sliceviewerframe)
        self.setCursor('watch')
        if event:
            # print(event.xdata,event.ydata)
            # need check for inbounds
            # convert coords from dummy label axis
            s = event.inaxes.format_coord(event.xdata,event.ydata)
            event.xdata,event.ydata = map(float,re.findall(r"(?:\d*\.\d+)",s))
            xdata = int(np.round(event.xdata))
            ydata = int(np.round(event.ydata))
        elif coords:
            xdata,ydata = coords
        if xdata < 0 or ydata < 0:
            return None
        # elif self.ui.data['raw'][0,self.ui.get_currentslice(),int(event.x),int(event.y)] == 0:
        #     print('Clicked in background')
        #     return
        else:
            # roi context
            self.ui.roi = self.ui.rois['blast']
            self.update_roinumber_options()
            roi = self.ui.get_currentroi()
            # in this branch for SAM analysis there are two workflows, both make use
            # of this method. do2d selects the workflow. note this is separate from
            # do3d, which is only a pass-through arg to closeROI.
            # do2d True. a intermediate SAM evaluation on the current point clicked 
            # during the assembly of the BLAST ROI, and just on the current slice
            # do2d False. a full multi-slice SAM after a BLAST ROI has been completed
            if do2d and self.ui.config.SAM2dauto:
                if roi == 0:
                    self.createROI(coords = (xdata,ydata,self.ui.get_currentslice()) )
                else: # for 2d this is only a dummy ROI so re-use it.
                    self.updateROI(coords = (xdata,ydata,self.ui.get_currentslice()))
            else:
                # the current ROI is a dummy, clear it before creating the actual BLAST ROI
                if self.ui.config.SAM2dauto:
                    self.clearROI()
                self.createROI(coords = (xdata,ydata,self.ui.get_currentslice()) )
            
        roi = self.ui.get_currentroi()

        try:
            self.closeROI(self.ui.data[self.ui.s].dset['seg_raw'][self.ui.chselection]['d'],self.ui.get_currentslice(),do3d=do3d)
        except Exception as e:
            print(e)
            print('ROI not completed')
            return

        # update layer menu
        self.update_layermenu_options(self.ui.roi[self.ui.s][roi])

        if True:
            # note some duplicate calls to generate_overlay should be removed
            for ch in [self.ui.chselection]:
                fusionstack = generate_blast_overlay(self.ui.data[self.ui.s].dset['raw'][ch]['d'],
                                                self.ui.roi[self.ui.s][roi].data['seg'],
                                                layer=self.ui.roiframe.layer.get(),
                                                overlay_intensity=self.config.OverlayIntensity)
                self.ui.roi[self.ui.s][roi].data['seg_fusion'][ch] = fusionstack

        # if triggered by selectROI button event
        if not do2d:
            if self.buttonpress_id:
                self.ui.sliceviewerframe.canvas.callbacks.disconnect(self.buttonpress_id)
                self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='')
                self.buttonpress_id = None
            self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='arrow')
            self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()

        roi = self.ui.get_currentroi()

        # in the SAM viewer, there is only ET/TC or WT but not both, so the ROI status is set true immediately
        # once a BLAST segmentation is available in either layer. Now, run the SAM segmentation only on the current slice
        # to illustrate whether the BLAST ROI is adequate for prompting.
        # This code should move to a separate function to be called from ROIclick.
        if self.ui.roi[self.ui.s][roi].status:

            # temporary. experiment.
            # record the 2d SAM point prompt result, based on the clicked point
            # note. saving 2d single-slice point prompt result separately from the ROIstats save in 
            # the saveROI() method. 'tag' ROIstats arg is hard-coded here to put the results in separate attributes
            # in the json. Note that by this arrangement, the same roi number is used for both
            # the 2d save and the later multi-slice 3d save in saveROI.
            if False:
                # update SAM roi with prompts derived from BLAST roi
                self.ui.sliceviewerframe.sam2d_callback(prompt='point')
                # saveROI here
                self.saveROI(roitype='sam',roi=self.ui.currentroi,tag='point_slice'+str(self.ui.currentslice))
            else:
                # normally use 'bbox' and don't save.
                with Profile() as profile:
                    self.ui.sliceviewerframe.sam2d_callback(prompt='bbox',remote=self.ui.config.AWS)
                    print('sam2d_callback')
                    (
                        Stats(profile)
                        .strip_dirs()
                        .sort_stats(SortKey.TIME)
                        .print_stats(15)
                    )

            # automatically switch to SAM display
            self.set_overlay('SAM')
            # in SAM, the ET bounding box segmentation is interpreted directly as TC
            if False: # calling this from sam2d_callback now
                self.layerSAM_callback()
            # activate 3d SAM button
            self.ui.sliceviewerframe.run3dSAM.configure(state='active')
        else:
            # this workflow option shouldn't be triggered in the current implementation of 
            # SAM viewer
            self.set_overlay('BLAST')
            self.ui.dataselection = 'seg_raw_fusion'
            self.ui.sliceviewerframe.updateslice()
            self.layer_callback(layer='ET')


        return None
    
    # records button press coords in a new ROI object
    # create a parallel list of SAM and BLAST roi's. 
    def createROI(self,coords=(0,0,0),bbox={}):
        blast_layer = self.layer.get()
        roi = ROIBLAST(coords,dim=self.ui.sliceviewerframe.dim,layer=blast_layer)
        self.ui.rois['blast'][self.ui.s].append(roi)
        if blast_layer in ['ET','TC']:
            sam_layer = 'TC'
        else:
            sam_layer = blast_layer
        roi2 = ROISAM(dim=self.ui.sliceviewerframe.dim,bbox=bbox,layer=sam_layer)
        self.ui.rois['sam'][self.ui.s].append(roi2)
 
        self.currentroi.set(self.currentroi.get() + 1)
        self.updateROIData()
        self.update_roinumber_options()

    # updates button press coords for current ROI
    def updateROI(self,coords=(0,0,0)):
        blast_layer = self.layer.get()
        roi = self.ui.rois['blast'][self.ui.s][self.ui.get_currentroi()]
        roi.coords[blast_layer]['x'] = coords[0]
        roi.coords[blast_layer]['y'] = coords[1]
        roi.coords[blast_layer]['slice'] = coords[2]

        # self.updateROIData()

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
        # but can't use it if small lesions are present.
        # connectivity = 6 is the minimum, again to filter out noise and tenuously connected regions
        if False:
            CC_labeled = cc3d.dust(mask,connectivity=6,threshold=100,in_place=False)
        else:
            CC_labeled = copy.deepcopy(mask)
        CC_labeled = cc3d.connected_components(CC_labeled,connectivity=6)
        stats = cc3d.statistics(CC_labeled)

        # assemble the composite mask from the sequence of clicked points
        # this is redundant where the projections in image space overlap,
        # but picks up any situation when they don't.
        objectmask = np.zeros_like(metmaskstack)
        for p in self.ui.pt[self.ui.s]:
            # objectnumber = CC_labeled[roislice,ypos,xpos]
            objectnumber = CC_labeled[p.coords['slice'],p.coords['y'],p.coords['x']]
            pmask = (CC_labeled == objectnumber).astype('double')
            objectmask = np.logical_or(pmask,objectmask)
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
 
            # create a combined seg mask from the three layers
            # using nnunet convention for labels
            # there is currently no WT segmentation in the SAM viewer
            if self.ui.roi[s][roi].data['WT'] is None:
                self.ui.roi[s][roi].data['seg'] = 4*self.ui.roi[s][roi].data['ET'] + \
                                                     2*self.ui.roi[s][roi].data['TC']
                # for current SAM mode, ET layer alone is complete
                self.ui.roi[s][roi].status = True # ie ROI is now complete                                                    
            else:
                self.ui.roi[s][roi].data['seg'] = 4*self.ui.roi[s][roi].data['ET'] + \
                                                     2*self.ui.roi[s][roi].data['TC'] + \
                                                     1*self.ui.roi[s][roi].data['WT']
                self.ui.roi[s][roi].status = True # ie ROI has both compartments selected

        elif m == 'WT': # WT gets no additional processing. just create a combined seg mask from the three layers
                        # nnunet convention for labels
            if self.ui.roi[s][roi].data['ET'] is None:
                self.ui.roi[s][roi].data['seg'] = 1*self.ui.roi[s][roi].data['WT']
                self.ui.roi[s][roi].status = True # ie either single compartment ET/WT for ROI is sufficient
            else:
                # update WT based on smoothing for TC
                self.ui.roi[s][roi].data['WT'] = self.ui.roi[s][roi].data['WT'] | self.ui.roi[s][roi].data['TC']
                # if 'ET' exists but 'WT' didn't, then have to rebuild the combined mask because of the dummy +1
                self.ui.roi[s][roi].data['seg'] = 4*self.ui.roi[s][roi].data['ET'] + \
                                                    2*self.ui.roi[s][roi].data['TC'] + \
                                                    1*self.ui.roi[s][roi].data['WT']
                self.ui.roi[s][roi].status = True # ROI has both compartments selected
            if False:
                # WT contouring. not updated lately.
                objectmask_contoured = {}
                for sl in range(self.config.ImageDim[0]):
                    objectmask_contoured[sl] = find_contours(self.ui.roi[s][roi].data['WT'][sl,:,:])
                self.ui.roi[s][roi].data['contour']['WT'] = objectmask_contoured

        return None

    # output images and prompts for sam segmentation
    # using tmpfiles and standalone script for separate ptorch env. ultimately should add ptorch to the blast env
    def save_prompts(self,slice=None,orient='ax'):

        if orient == 'ax':
            slicedim = self.ui.sliceviewerframe.dim[0]
        elif orient == 'sag':
            slicedim = self.ui.sliceviewerframe.dim[2]
        elif orient == 'cor':
            slicedim = self.ui.sliceviewerframe.dim[1]

        if slice == None:
                rslice = list(range(slicedim)) # do all slices
        else:
            if isinstance(slice,list):
                rslice = slice
            else:
                rslice = [slice] # do given slice(s)

        fileroot = os.path.join(self.ui.data[self.ui.s].studydir,'sam')
        for d in ['images','prompts','predictions']:
            filedir = os.path.join(fileroot,orient,d)
            if os.path.exists(filedir):
                shutil.rmtree(filedir)
            os.makedirs(os.path.join(filedir),exist_ok=True)
        filedir = os.path.join(fileroot,'predictions_nifti')
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        # these image channel and prompt keys could later be generalized as args
        for ch,prompt in zip([self.ui.chselection],['bbox']):
            # roi.data['bbox'] is the multi-slice mask for prompting SAM
            rref = self.ui.rois['sam'][self.ui.s][self.ui.currentroi].data[prompt][orient]
            dref = self.ui.data[self.ui.s].dset['raw'][ch]['d']
            if dref is None:
                raise ValueError
            affine_bytes = self.ui.data[self.ui.s].dset['raw'][ch]['affine'].tobytes()
            affine_bytes_str = str(affine_bytes)
            idx = 0
            for slice in rslice:
                if len(np.where(self.get_prompt_slice(slice,rref,orient))[0]):
                    outputfilename = os.path.join(fileroot,orient,'prompts',
                            'img_' + str(idx).zfill(5) + '_case_' + self.ui.caseframe.casename.get() + '_slice_' + str(slice).zfill(3) + '.png')
                    plt.imsave(outputfilename,self.get_prompt_slice(slice,rref,orient),cmap='gray',)
                    outputfilename = os.path.join(fileroot,orient,'images',
                            'img_' + str(idx).zfill(5) + '_case_' + self.ui.caseframe.casename.get() + '_slice_' + str(slice).zfill(3) + '.png')
                    meta = PIL.PngImagePlugin.PngInfo()
                    meta.add_text('slicedim',str(slicedim))
                    meta.add_text('affine',affine_bytes_str)
                    plt.imsave(outputfilename,self.get_prompt_slice(slice,dref,orient),cmap='gray',pil_kwargs={'pnginfo':meta})

                    idx += 1

        # create tar file for multi-slice if remote processing
        if self.ui.sam.remote:
            if len(rslice) > 1:
                for d in ['prompts','images']:
                    command = 'tar -cvf '
                    command += os.path.join(fileroot,orient,d,'png.tar')
                    command += ' -C ' + os.path.join(fileroot,orient,d)
                    command += ' .'
                    os.system(command)
        return

    # convenience method, could use rotations instead
    def get_prompt_slice(self,idx,img_arr,orient):
        if orient == 'ax':
            return img_arr[idx]
        elif orient == 'sag':
            return img_arr[:,:,idx]
        elif orient == 'cor':
            return img_arr[:,idx,:]

    # for exporting BLAST/SAM segmentations.
    # tag - unique string for output filenames
    def saveROI(self,roitype=None,roi=None,outputpath=None,tag=None):

        if roitype is None:
            roitype = ['blast','sam']
        else: 
            if type(roitype) == list:
                roitype = roitype
            else:
                roitype = [roitype]

        if outputpath is None:
            outputpath = self.ui.caseframe.casedir
        if roi is None:
            roilist = list(map(int,self.roilist))
        else:
            if type(roi) == list:
                roilist = roi
            else:
                roilist = [roi]

        fileroot = self.ui.data[self.ui.s].studydir

        for r in roitype:
            if tag is None:
                tag = r
            else:
                tag = r + '_' + tag
            # not sure if this check is needed?
            if len(self.ui.rois[r][self.ui.s]) > self.ui.currentroi: # > for 1-based indexing
                for img in ['ET','TC','WT']:
                    if self.ui.rois[r][self.ui.s][1].data[img] is None:
                        continue
                    img_output = np.zeros(self.ui.sliceviewerframe.dim)
                    for roinumber in roilist:
                        img_output += self.ui.rois[r][self.ui.s][int(roinumber)].data[img]
                    outputfilename = os.path.join(fileroot,'{}_'.format(img) + tag + '.nii')
                    self.ui.data[self.ui.s].writenifti(img_output,
                                                        outputfilename,
                                                        affine=self.ui.data[self.ui.s].dset['raw']['t1+']['affine'])
                    
            # output stats
            # tag is for a unique key in stats.sjon
            for roinumber in roilist:
                # temporary arrangement for experiment.
                if 'slice' in tag:
                    rtag = '_'.join(tag.split('_')[:2])
                else:
                    rtag = tag
                self.ROIstats(save=True,roi=roinumber,roitype=r,tag=rtag)
                
        self.ui.set_message('ROI saved')


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
        self.ui.data[s].dset['seg_raw'][self.ui.chselection]['ex'] = True

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
            self.ui.data[s].mask['blast'][dt]['d'] = copy.deepcopy(self.ui.roi[s][self.ui.currentroi].data[dt])
            self.ui.data[s].mask['blast'][dt]['ex'] = True
            if self.ui.sliceviewerframes['overlay'] is not None:
                self.ui.sliceviewerframes['overlay'].maskdisplay_button['blast'].configure(state='active')
            if updatemask and False:
                # by this option, a BLAST segmentation could overwrite the current UI mask directly as a convenience.
                # otherwise, it will be done in separate step from the Overlay sliceviewer. 
                # would better need a further checkbox on the GUI for this auto option
                self.ui.data[s].mask[dt]['d'] = copy.deepcopy(self.ui.roi[self.ui.currentroi].data[dt])
        self.updatesliders()

    # forward-copy certain results from the SAM ROI to the main dataset
    # duplicates updateData might be able to combine better.
    def updateSAMData(self,updatemask=False):
        s = self.ui.s
        # anything else to copy??  'seg_raw_fusion_d','seg_raw','blast','seg_raw_fusion'
        layer = self.layer.get()
        for dt in ['seg_fusion']:
            for ch in [self.ui.chselection]:
                self.ui.data[s].dset[dt][ch]['d'] = copy.deepcopy(self.ui.rois['sam'][s][self.ui.currentroi].data[dt][ch])
        for dt in ['TC','WT']:
            self.ui.data[s].mask['sam'][dt]['d'] = copy.deepcopy(self.ui.rois['sam'][s][self.ui.currentroi].data[dt])
            self.ui.data[s].mask['sam'][dt]['ex'] = True
            if self.ui.sliceviewerframes['overlay'] is not None:
                self.ui.sliceviewerframes['overlay'].maskdisplay_button['blast'].configure(state='active')


    # eliminate latest ROI if there are multiple ROIs in current case
    def clearROI(self):
        n = len(self.ui.roi[self.ui.s])
        if n>1:    
            self.ui.rois['sam'][self.ui.s].pop(self.ui.currentroi)
            self.ui.rois['blast'][self.ui.s].pop(self.ui.currentroi)
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

    # eliminate all ROIs at once. 
    # note that this does not presently clear threshold settings saved in 
    # blastdata even though it resets the values shown in the slider bars. 
    # this is a loose end that should be resolved, depending on whether it 
    # is preferred to reset the blastdata, or retain it and not change the 
    # slider bars. 
    def resetROI(self,data=True):
        self.currentroi.set(0)
        self.set_overlay('')
        self.ui.reset_roi()
        self.update_roinumber_options(n=1)
        self.ui.roiframe.layertype.set('blast')
        self.ui.roiframe.layer.set('ET')
        self.ui.chselection = self.config.DefaultChannel
        # awkward check here due to using resetROI for clearROI
        if self.ui.sliceviewerframe.canvas is not None and data:
            self.enhancingROI_overlay_callback()
        if self.ui.chselection in ['t1+','flair']:
            for l in ['ET','T2 hyper']:
                for sl in ['t12','flair','bc']:
                    self.thresholds[l][sl].set(self.ui.config.thresholddefaults[sl])
                    self.updatesliderlabel(l,sl)
        # additionally clear any point selections
        self.ui.reset_pt()
        # deactivate 3d SAM
        self.ui.sliceviewerframe.run3dSAM.configure(state='disabled')


    def set_roi(self,roi=None):
        self.roistate = 'blast'
        if roi is None:
            self.ui.roi = self.ui.rois[self.roistate]
        else:
            self.ui.roi = self.ui.rois[roi]

    def append_roi(self,d):
        for k,v in d.items():
            if isinstance(v,dict):
                self.append_roi(d)
            else:
                v.append(0)

    # various output stats, add more as required. 
    # an existing stats file is read in if present, and values added for the current roi,
    # 
    # roi - optionally provide the roi number to process. 
    # tag - top-level section key for output .json file. 
    # roitype - for sam versus blast. dual roi data structure continues to be awkward.
    # slice - process given slice or whole volume if None
    # timer - read the SVFrame timer and record
    def ROIstats(self,roi=None,save=False,tag=None,roitype='blast',slice=None):
        
        if roi is None:
            roi = self.ui.get_currentroi() # ie, there is only 1 roi in SAM viewer for now
        s = self.ui.s
        r = self.ui.rois[roitype][s][roi]
        data = copy.deepcopy(r.data)

        for dt in ['ET','TC','WT']:
            # check for a complete segmentation
            if dt not in data.keys():
                continue
            # checking for == 0 here but this is a bug, the dataset should 
            # either be non-zero or None
            elif data[dt] is None or np.max(data[dt]) == 0:
                continue
            if np.max(data[dt]) > 1:
                data[dt] = data[dt] == np.max(data[dt])
            if slice is None:
                dset = data[dt]
            else:
                dset = data[dt][slice]
            r.stats['vol'][dt] = len(np.where(dset)[0])

            # ground truth comparisons
            if self.ui.data[self.ui.s].mask['gt'][dt]['ex']:

                # pull out the matching lesion using cc3d
                gt_mask = np.copy(self.ui.data[self.ui.s].mask['gt'][dt]['d'])
                CC_labeled = cc3d.connected_components(gt_mask,connectivity=6)
                centroid_point = np.array(list(map(int,np.nanmean(np.where(data[dt]),axis=1)))) 
                objectnumber = CC_labeled[centroid_point[0],centroid_point[1],centroid_point[2]]
                gt_lesion = (CC_labeled == objectnumber).astype('uint8')
                if slice is not None:
                    gt_lesion = gt_lesion[slice]

                # dice
                r.stats['dsc'][dt] = 1-dice(gt_lesion.flatten(),dset.flatten()) 
                # haunsdorff
                r.stats['hd'][dt] = max(directed_hausdorff(np.array(np.where(gt_lesion)).T,np.array(np.where(dset)).T)[0],
                                                        directed_hausdorff(np.array(np.where(dset)).T,np.array(np.where(gt_lesion)).T)[0])
            else:
                print('No ground truth comparison available')
            
        # optional save dict to json
        if save:
            studydir = self.ui.data[self.ui.s].studydir
            statsfile = os.path.join(studydir,'stats.json')
            if os.path.exists(statsfile):
                fp = open(statsfile,'r+')
                sdict = json.load(fp)
                if tag not in sdict.keys():
                    sdict[tag] = {}
                fp.seek(0)
            else:
                fp = open(statsfile,'w')
                sdict = {tag:{}}

            r2 = self.ui.rois[roitype][self.ui.s][roi]
            sdict[tag]['roi'+str(roi)] = {'stats':None,'bbox':None}
            sdict[tag]['roi'+str(roi)]['stats'] = r2.stats
            if hasattr(r2,'bboxs'):
                bboxs = {}
                if bool(r2.bboxs):
                    kset = []
                    if slice is None:
                        kset = r2.bboxs.keys()
                    else:
                        if slice in r2.bboxs.keys():
                            kset = [slice]
                    if len(kset):
                        for k in kset:
                            try:
                                bboxs[k] = {k2:r2.bboxs[k][k2] for k2 in ['p0','p1','slice']}
                            except KeyError:
                                pass
                sdict[tag]['roi'+str(roi)]['bbox'] = bboxs

            json.dump(sdict,fp,indent=4)
            fp.truncate()
            fp.close()

    def clear_stats(self):
        studydir = self.ui.data[self.ui.s].studydir
        statsfile = os.path.join(studydir,'stats.json')
        if os.path.exists(statsfile):
            os.remove(statsfile)
        # experiment temporary. remove all individual slice nifti's.
        files = os.listdir(studydir)
        slicefiles = [os.path.join(studydir,f) for f in files if 'slice' in f]
        if len(slicefiles):
            for f in slicefiles:
                os.remove(f)


