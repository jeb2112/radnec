import os,sys
import numpy as np
import glob
import copy
import re
import time
import tkinter as tk
from tkinter import ttk,StringVar,DoubleVar,PhotoImage
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import SimpleITK as sitk
from sklearn.cluster import KMeans,MiniBatchKMeans
from skimage.morphology import disk,square,binary_dilation,binary_closing,flood_fill,ball,cube
from cucim.skimage.morphology import binary_closing as cucim_binary_closing
from scipy.spatial.distance import dice
from scipy.ndimage import binary_closing as scipy_binary_closing
from cupyx.scipy.ndimage import binary_closing as cupy_binary_closing
import cupy as cp
import cc3d

from Config import Config
import OverlayPlots
from CreateFrame import CreateFrame

# contains various ROI methods and variables
class CreateROIFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)

        self.buttonpress_id = None # temp var for keeping track of button press event
        self.finalROI_overlay_value = tk.BooleanVar(value=False)
        self.enhancingROI_overlay_value = tk.BooleanVar(value=False)
        self.currentt1threshold = tk.DoubleVar()
        self.currentt2threshold = tk.DoubleVar()
        self.currentbcsize = tk.DoubleVar(value=1.5)

        ########################
        # layout for the buttons
        ########################

        self.frame.grid(column=2,row=3,rowspan=4,sticky='e')

        # normal slice button
        normalSlice = ttk.Button(self.frame,text='normal slice',command=self.normalslice_callback)
        normalSlice.grid(column=0,row=0,sticky='e')

        # ROI buttons
        enhancingROI_frame = ttk.Frame(self.frame,padding='0')
        enhancingROI_frame.grid(column=1,row=0,sticky='e')
        enhancingROI = ttk.Button(enhancingROI_frame,text='enhancing ROI',
                                #   command=lambda arg1=None: self.ui.runblast(currentslice=arg1))
                                  command=self.enhancingROI_callback)
        enhancingROI.grid(column=0,row=0,sticky='e')
        enhancingROI_overlay = ttk.Checkbutton(enhancingROI_frame,text='',
                                               variable=self.enhancingROI_overlay_value,
                                               command=self.enhancingROI_overlay_callback)
        enhancingROI_overlay.grid(column=1,row=0,sticky='e')
        enhancingROI_frame.update()

        # select ROI button
        finalROI_frame = ttk.Frame(self.frame,padding='0')
        finalROI_frame.grid(column=2,row=0,sticky='e')
        finalROI = ttk.Button(finalROI_frame,text='select ROI',command = self.selectROI)
        finalROI.grid(row=0,column=0,sticky='e')
        finalROI_overlay = ttk.Checkbutton(finalROI_frame,text='',
                                           variable=self.finalROI_overlay_value,
                                           command=self.finalROI_overlay_callback)
        finalROI_overlay.grid(column=1,row=0,sticky='e')

        # save ROI button
        saveROI = ttk.Button(self.frame,text='save ROI',command = self.saveROI)
        saveROI.grid(row=0,column=4,sticky='e')

        # clear ROI button
        clearROI = ttk.Button(self.frame,text='clear ROI',command = self.clearROI)
        clearROI.grid(row=0,column=5,sticky='e')
        self.frame.update()

        ########################
        # layout for the sliders
        ########################

        # t1 slider
        self.t1sliderframe = ttk.Frame(self.frame,padding='0')
        self.t1sliderframe.grid(column=0,row=1,columnspan=6,sticky='e')
        t1label = ttk.Label(self.t1sliderframe, text='T1')
        t1label.grid(column=0,row=0,sticky='w')
        self.currentt1threshold.set(1.)
        self.t1slider = ttk.Scale(self.t1sliderframe,from_=-4,to=4,variable=self.currentt1threshold,state='disabled',
                                  length='3i',command=self.updatet1label,orient='horizontal')
        self.t1slider.grid(column=1,row=0,sticky='e')
        self.t1sliderlabel = ttk.Label(self.t1sliderframe,text=self.currentt1threshold.get())
        self.t1sliderlabel.grid(column=2,row=0,sticky='e')

        # t2 slider
        t2label = ttk.Label(self.t1sliderframe, text='T2')
        t2label.grid(column=0,row=1,stick='w')
        self.currentt2threshold.set(1.)
        self.t2slider = ttk.Scale(self.t1sliderframe,from_=-4,to=4,variable=self.currentt2threshold,state='disabled',
                                  length='3i',command=self.updatet2label,orient='horizontal')
        self.t2slider.grid(column=1,row=1,sticky='e')
        self.t2sliderlabel = ttk.Label(self.t1sliderframe,text=self.currentt2threshold.get())
        self.t2sliderlabel.grid(column=2,row=1,sticky='e')

        #brain cluster slider
        bclabel = ttk.Label(self.t1sliderframe,text='b.c.')
        bclabel.grid(column=0,row=2,sticky='w')
        self.bcslider = ttk.Scale(self.t1sliderframe,from_=0,to=4,variable=self.currentbcsize,state='disabled',
                                  length='3i',command=self.updatebclabel,orient='horizontal')
        self.bcslider.grid(column=1,row=2,sticky='e')
        self.bcsliderlabel = ttk.Label(self.t1sliderframe,text='{:.1f}'.format(self.currentbcsize.get()))
        self.bcsliderlabel.grid(row=2,column=2,sticky='e')


    # updates blast segmentation upon slider release only
    def updatet1threshold(self,event=None,currentslice=True):
        self.ui.blast.et_gate = None
        self.ui.blast.wt_gate = None
        self.ui.runblast(currentslice=currentslice)
        self.t1sliderlabel['text'] = '{:.1f}'.format(self.currentt1threshold.get())
        # if operating in finalROI mode additionally reprocess the final segmentation
        if self.finalROI_overlay_value.get() == True:
            self.ROIclick(do3d=False)

    # updates the text field showing the value during slider drag
    def updatet1label(self,event):
        self.t1sliderlabel['text'] = '{:.1f}'.format(self.currentt1threshold.get())

    def updatet2threshold(self,event=None,currentslice=True):
        self.ui.blast.et_gate = None
        self.ui.blast.wt_gate = None
        self.ui.runblast(currentslice=currentslice)
        self.t2sliderlabel['text'] = '{:.1f}'.format(self.currentt2threshold.get())
        if self.finalROI_overlay_value.get() == True:
            self.ROIclick(do3d=False)
        return
    
    def updatet2label(self,event):
        self.t2sliderlabel['text'] = '{:.1f}'.format(self.currentt2threshold.get())

    def updatebcsize(self,event=None):
        self.ui.blast.brain = None
        self.ui.runblast(currentslice=True)
        self.bcsliderlabel['text'] = '{:.1f}'.format(self.currentbcsize.get())
        return

    def updatebclabel(self,event):
        self.bcsliderlabel['text'] = '{:.1f}'.format(self.currentbcsize.get())
    
    def finalROI_overlay_callback(self,event=None):
        if self.finalROI_overlay_value.get() == False:
            self.ui.dataselection = 'raw'
            self.ui.updateslice()
        else:
            self.ui.dataselection = 'seg_fusion_d'
            self.ui.updateslice(wl=True)
            self.enhancingROI_overlay_value.set(False)

    def enhancingROI_overlay_callback(self,event=None):
        if self.enhancingROI_overlay_value.get() == False:
            self.ui.dataselection = 'raw'
            self.ui.updateslice()
        else:
            self.ui.dataselection = 'seg_raw_fusion_d'
            self.ui.updateslice(wl=True)
            self.finalROI_overlay_value.set(False)

    def enhancingROI_callback(self,event=None):
        self.finalROI_overlay_value.set(False)
        self.enhancingROI_overlay_value.set(True)
        self.ui.runblast()

    def normalslice_callback(self,event=None):
        self.normalslice=self.ui.get_currentslice()
        # do kmeans
        # Creates a matrix of voxels for normal brain slice
        # Gating Routine
        t1channel_normal = self.ui.data['raw'][0,self.normalslice,:,:]
        t2channel_normal = self.ui.data['raw'][1][self.normalslice,:,:,]

        # kmeans to calculate statistics for brain voxels
        t2 = np.ravel(t2channel_normal)
        t1 = np.ravel(t1channel_normal)
        X = np.column_stack((t2,t1))
        # rng(1)
        np.random.seed(1)
        # [idx,C] = KMeans(n_clusters=2).fit(X)
        kmeans = KMeans(n_clusters=2,n_init='auto').fit(X)

        # Calculate stats for brain cluster
        self.ui.data['params']['stdt1'] = np.std(X[kmeans.labels_==1,1])
        self.ui.data['params']['stdt2'] = np.std(X[kmeans.labels_==1,0])
        self.ui.data['params']['meant1'] = np.mean(X[kmeans.labels_==1,1])
        self.ui.data['params']['meant2'] = np.mean(X[kmeans.labels_==1,0])

        # activate thresholds only after normal slice stats are available
        self.ui.roiframe.bcslider['state']='normal'
        self.ui.roiframe.t2slider['state']='normal'
        self.ui.roiframe.t1slider['state']='normal'
        self.ui.roiframe.t1slider.bind("<ButtonRelease-1>",self.ui.roiframe.updatet1threshold)
        self.ui.roiframe.bcslider.bind("<ButtonRelease-1>",self.ui.roiframe.updatebcsize)
        self.ui.roiframe.t2slider.bind("<ButtonRelease-1>",self.ui.roiframe.updatet2threshold)

        # automatically run the default thresholds in 3d to start things off
        # awkward have to reference to get the tk Var
        self.ui.roiframe.updatet1threshold(currentslice=None)
        self.ui.dataselection = 'seg_raw_fusion_d'

    def selectROI(self):
        self.finalROI_overlay_value.set(True)
        self.enhancingROI_overlay_value.set(False)
        # if there is an existing ROI and this button is clicked again, the assumption is that the 
        # BLAST thresholds have been adjusted. redo 3d BLAST and then reprocess final segmentation
        if self.ui.x:
            self.ui.runblast()
            self.ROIclick()
        else:
            self.buttonpress_id = self.ui.sliceviewerframe.canvas.callbacks.connect('button_press_event',self.ROIclick)
            self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='crosshair')
        return None
    
    def ROIclick(self,event=None,do3d=True):
        self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='watch')
        self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()
        if event:
            # print(event.xdata,event.ydata)
            # need check for inbounds
            if event.xdata < 0 or event.ydata < 0:
                return None
            else:
                self.ui.x = int(event.xdata)
                self.ui.y = int(event.ydata)
            
        self.closeROI(self.ui.x,self.ui.y,self.ui.data['seg_raw'],self.ui.get_currentslice(),do3d=do3d)
        fusionstack = np.zeros((155,240,240,2))
        if False:
            for slice in range(0,155):  
            # newfusion = imfuse(objectmask_filled[slice,:,:],t1mpragestack[slice,:,:],'blend')
                newfusion = 0.5*self.ui.data['wt'][slice,:,:] + 0.5*self.ui.data['raw'][0][slice,:,:]
                segfusionstack[slice] = newfusion
        else:
            fusionstack = OverlayPlots.generate_overlay(self.ui.data['raw'],self.ui.data['seg'])
        self.ui.data['seg_fusion'] = fusionstack
        self.ui.data['seg_fusion_d'] = copy.copy(self.ui.data['seg_fusion'])
        # if triggered by a button event
        if self.buttonpress_id:
            self.ui.sliceviewerframe.canvas.callbacks.disconnect(self.buttonpress_id)
            self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='')
            self.buttonpress_id = None
        self.ui.dataselection = 'seg_fusion_d'
        self.ui.sliceviewerframe.updateslice()
        self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='arrow')
        self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()
        return None
 
    def closeROI(self,xpos,ypos,metmaskstack,currentslice,do3d=True):
        # awkward mess of 2d,3d ET and WT
        # a quick config for ET, WT smoothing
        mlist = {'et':{'threshold':3,'ball':10,'cube':2},
                    'wt':{'threshold':1,'ball':10,'cube':2}}
        for m in mlist.keys():
            mask = (metmaskstack >= mlist[m]['threshold']).astype('double')
            # TODO: 2d versus 3d connected components
            CC_labeled = cc3d.connected_components(mask,connectivity=26)
            stats = cc3d.statistics(CC_labeled)

            # ypos = round(h.Position(2)/2) # divide by ScaleFactor
            # xpos = round(h.Position(1)/2) # divide by ScaleFactor

            # objectnumber = CC_labeled(ypos,xpos,s.SliceNumber)

            # ie one click should be in both compartments
            objectnumber = CC_labeled[currentslice,ypos,xpos]

            # objectmask = ismember(CC_labeled,objectnumber)
            objectmask = (CC_labeled == objectnumber).astype('double')

            # thisBB = BB.BoundingBox[objectnumber,:,:]
            thisBB = stats['bounding_boxes'][objectnumber]

            # Creates filled in contour of objectmask 
            # se = strel('disk',10) 
            # close_object = imclose(objectmask,se) 
            # se2 = strel('square',2) #added to imdilate
            # se2 = square(2)

            # step 1. binary closing
            # for 3d closing, gpu is faster. use cucim library
            if do3d:
                se = ball(mlist[m]['ball'])
                start = time.time()
                objectmask_cp = cp.array(objectmask)
                se_cp = cp.array(se)
                # TODO: iterate binary_closing?
                close_object_cucim = cucim_binary_closing(objectmask_cp,footprint=se_cp)
                close_object = np.array(close_object_cucim.get())
                end = time.time()
                print('binary closing time = {:.2f} sec'.format(end-start))
                objectmask_final = close_object
                # ie not doing the floodfill step on ET
                if m == 'et':
                    self.ui.data[m] = objectmask_final
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
           # cpu only. use skimage library
                # if a final 3d seg already exists, use it
                if self.ui.data[m] is not None:
                    objectmask_final = self.ui.data[m]
                else:
                    objectmask_final = np.zeros(np.shape(self.ui.data['raw']))
                se = disk(mlist[m]['ball'])
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
                objectmask_final[currentslice,:,:] = close_object
                if m == 'et':
                    self.ui.data[m] = objectmask_final

            # self.ui.data[m] = close_object
            # se2 = strel('square',2) #added to imdilate
            # se2 = square(2)

            # step 2. flood fill
            # for small kernel, don't need gpu
            # TODO: does ET need anything further
            if do3d:
                se2 = cube(mlist[m]['cube'])
                objectmask_filled = binary_dilation(objectmask,se2)
                objectmask_filled = flood_fill(objectmask_filled,(currentslice,ypos,xpos),True)
                objectmask_final = objectmask_filled.astype('int')
                if m == 'et':
                    self.ui.data['tc'] = objectmask_final
                elif m == 'wt':
                    self.ui.data['wt'] = objectmask_final
            else:
                # if a final 3d seg already exists, use it
                # for step2, et is already done, and here tc is being calculated from et.
                if m == 'et' and self.ui.data['et'] is not None:
                    objectmask_final = self.ui.data['tc']
                se2 = square(mlist[m]['cube'])
                objectmask_filled = binary_dilation(objectmask[currentslice,:,:],se2)
                objectmask_filled = flood_fill(objectmask_filled,(ypos,xpos),True)
                objectmask_final[currentslice,:,:] = objectmask_filled.astype('int')     
                if m == 'et' and self.ui.data['et'] is not None:
                    self.ui.data['tc'] = objectmask_final
                elif m == 'wt' and self.ui.data['wt'] is not None:
                    self.ui.data['wt'] = objectmask_final

        self.ui.data['seg'] = 1*self.ui.data['et'] + 1*self.ui.data['tc'] + self.ui.data['wt']
        return None
    
    def saveROI(self):
        return
        # Save ROI data
    #     filename = "t1ce_" + self.casename
    #     outputpath = os.path.join(self.config.UIdataroot+self.casename,filename)
    #     save(outputpath + "/" + filename + ".mat",'greengate_count','redgate_count','objectmask','objectmask_filled','manualmasket','manualmasktc','centreimage','specificity_et','sensitivity_et','dicecoefficient_et','specificity_tc','sensitivity_tc','dicecoefficient_tc','b','b2','manualmask_et_volume','manualmask_tc_volume','objectmask_filled_volume','cumulative_elapsed_time')
    #     niftiwrite(objectmask,outputpath + "/" + filename + ".nii")
    #     niftiwrite(objectmask_filled,outputpath + "/" + filename + "_filled" + ".nii")
    #     niftiwrite(manualmasket,outputpath + "/" + filename + "_manualmask_et" + ".nii")
    #     niftiwrite(manualmasktc,outputpath + "/" + filename + "_manualmask_tc" + ".nii")
    #     niftiwrite(t1mprage_template,outputpath + "/" + 't1mprage_template.nii')

    def clearROI(self):
        self.ui.data = {'wt':None,'et':None,'tc':None}
        self.ui.data['params'] = {'stdt1':1,'stdt2':1,'meant1':1,'meant2':1}
        self.ui.dataselection = 'raw'
        # ROI selection coordinates
        self.ui.x = None
        self.ui.y = None
        self.ui.updateslice()

    #######
    # STATS
    #######

    def ROIstats(self,objectmask,objectmask_filled):
        # Load gold standard manual ROIs and create manual mask
        # manualmask = niftiread(inputpath + "/" + "BraTS2021_" + casename + "_seg" )
        manualmask = self.ui.data['label']
        manualmasket = manualmask>3 #enhancing tumor 
        manualmasknc = manualmask==1 #necrotic core
        manualmasket = manualmasket.astype('double')
        manualmasknc = manualmasknc.astype('double')
        manualmasktc = manualmasknc + manualmasket #tumor core
                
        # enhancing tumour
        groundtruth = manualmask['et']
        segmentation = objectmask

        sums = groundtruth + segmentation
        subs = groundtruth - segmentation
                
        TP = len(np.find(sums == 2))
        FP = len(np.find(subs == -1))
        TN = len(np.find(sums == 0))
        FN = len(np.find(subs == 1))

        specificity_et = TN/(TN+FP)
        sensitivity_et = TP/(TP+FN)
        dicecoefficient_et = dice(groundtruth.flatten(),segmentation.flatten()) 

        # tumour core
        groundtruth = manualmask['tc']
        segmentation = objectmask_filled

        sums = groundtruth + segmentation
        subs = groundtruth - segmentation
                
        TP = len(np.find(sums == 2))
        FP = len(np.find(subs == -1))
        TN = len(np.find(sums == 0))
        FN = len(np.find(subs == 1))

        specificity_tc = TN/(TN+FP)
        sensitivity_tc = TP/(TP+FN)
        dicecoefficient_tc = dice(groundtruth.flatten(),segmentation.flatten())

        # Calculate volumes
        manualmask_et_volume = len(np.where(manualmasket))
        manualmask_tc_volume = len(np.where(manualmasktc))
        objectmask_filled_volume = len(np.where(objectmask_filled))

