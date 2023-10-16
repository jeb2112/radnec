import os,sys
import numpy as np
import pickle
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
from scipy.io import savemat
from cupyx.scipy.ndimage import binary_closing as cupy_binary_closing
import cupy as cp
import cc3d

from Config import Config
import OverlayPlots
from CreateFrame import CreateFrame
from ROI import ROI

# contains various ROI methods and variables
class CreateROIFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)

        self.buttonpress_id = None # temp var for keeping track of button press event
        self.finalROI_overlay_value = tk.BooleanVar(value=False)
        self.enhancingROI_overlay_value = tk.BooleanVar(value=False)
        self.currentt1threshold = tk.DoubleVar()
        self.currentt2threshold = tk.DoubleVar()
        self.currentbcsize = tk.DoubleVar(value=2)
        self.layerlist = {'blast':['ET','edema','both'],'seg':['ET','TC','WT','all']}
        self.layer = tk.StringVar(value='ET')
        self.layertype = tk.StringVar(value='blast')
        self.currentroi = tk.IntVar(value=0)
        # self.roilist = [str(i) for i in range(len(self.ui.roi)+1)]
        self.roilist = []

        ########################
        # layout for the buttons
        ########################

        self.frame.grid(column=2,row=3,rowspan=5,sticky='e')

        # normal slice button
        normalSlice = ttk.Button(self.frame,text='normal slice',command=self.normalslice_callback)
        normalSlice.grid(column=0,row=0,sticky='w')

        # ROI buttons
        enhancingROI_frame = ttk.Frame(self.frame,padding='0')
        enhancingROI_frame.grid(column=0,row=1,sticky='w')
        enhancingROI = ttk.Button(enhancingROI_frame,text='BLAST',
                                #   command=lambda arg1=None: self.ui.runblast(currentslice=arg1))
                                  command=self.enhancingROI_callback)
        enhancingROI.grid(column=0,row=0,sticky='w')
        enhancingROI_overlay = ttk.Checkbutton(enhancingROI_frame,text='',
                                               variable=self.enhancingROI_overlay_value,
                                               command=self.enhancingROI_overlay_callback)
        enhancingROI_overlay.grid(column=1,row=0,sticky='w')
        enhancingROI_frame.update()

        # enhancing layer choice
        layerframe = ttk.Frame(self.frame,padding='0')
        layerframe.grid(row=1,column=2,sticky='w')
        layerlabel = ttk.Label(layerframe,text='layer:')
        layerlabel.grid(row=0,column=0,sticky='w')
        self.layer.trace_add('write',lambda *args: self.layer.get())
        self.layermenu = ttk.OptionMenu(layerframe,self.layer,*self.layerlist[self.layertype.get()],command=self.layer_callback)
        self.layermenu.config(width=6)
        self.layermenu.grid(column=1,row=0,sticky='w')

        # n'th roi number choice
        roinumberframe = ttk.Frame(self.frame,padding='0')
        roinumberframe.grid(row=1,column=3,sticky='w')
        roinumberlabel = ttk.Label(roinumberframe,text='roi:')
        roinumberlabel.grid(row=0,column=0,sticky='w')
        self.currentroi.trace_add('write',lambda *args: self.currentroi.get())
        self.roinumbermenu = ttk.OptionMenu(roinumberframe,self.currentroi,*self.roilist,command=self.roinumber_callback)
        self.roinumbermenu.config(width=2)
        self.roinumbermenu.grid(column=1,row=0,sticky='w')

        # select ROI button
        finalROI_frame = ttk.Frame(self.frame,padding='0')
        finalROI_frame.grid(column=1,row=1,sticky='w')
        finalROI = ttk.Button(finalROI_frame,text='select ROI',command = self.selectROI)
        finalROI.grid(row=0,column=0,sticky='w')
        finalROI_overlay = ttk.Checkbutton(finalROI_frame,text='',
                                           variable=self.finalROI_overlay_value,
                                           command=self.finalROI_overlay_callback)
        finalROI_overlay.grid(column=1,row=0,sticky='w')


        # update ROI button. runs full 3d after a round of 2d adjustments
        # this could also be implemented as an automatic update
        updateROI = ttk.Button(self.frame,text='update ROI',command = self.updateROI)
        updateROI.grid(row=0,column=1,sticky='w')

        # save ROI button
        saveROI = ttk.Button(self.frame,text='save ROI',command = self.saveROI)
        saveROI.grid(row=0,column=2,sticky='w')

        # clear ROI button
        clearROI = ttk.Button(self.frame,text='clear ROI',command = self.clearROI)
        clearROI.grid(row=0,column=3,sticky='w')
        self.frame.update()

        ########################
        # layout for the sliders
        ########################

        # t1 slider
        self.t1sliderframe = ttk.Frame(self.frame,padding='0')
        self.t1sliderframe.grid(column=0,row=2,columnspan=7,sticky='e')
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

    # methods for layer options menu
    def layer_callback(self,layer):
        self.ui.currentlayer = self.layer.get()
        # generate a new overlay
        if self.layertype.get() == 'blast':
            self.ui.data['seg_raw_fusion'] = OverlayPlots.generate_overlay(self.ui.data['raw'],self.ui.data['seg_raw'],self.layer.get())
            self.ui.data['seg_raw_fusion_d'] = copy.copy(self.ui.data['seg_raw_fusion'])
        elif self.layertype.get() == 'seg':
            self.ui.data['seg_fusion'] = OverlayPlots.generate_overlay(self.ui.data['raw'],self.ui.data['seg'],self.layer.get())
            self.ui.data['seg_fusion_d'] = copy.copy(self.ui.data['seg_fusion'])
        self.ui.updateslice()

    def update_layermenu_options(self,type):
        self.layertype.set(type)
        menu = self.layermenu['menu']
        menu.delete(0,'end')
        for s in self.layerlist[type]:
            menu.add_command(label=s,command = tk._setit(self.layer,s,self.layer_callback))
        self.layer.set(self.layerlist[type][0])

    # methods for roi number choice menu
    def roinumber_callback(self,item):
        self.ui.currentroi = self.currentroi.get()
        # reference or copy
        self.ui.data = copy.deepcopy(self.ui.roi[self.ui.currentroi].data)
        self.ui.updateslice()
        return
    
    def update_roinumber_options(self,n=None):
        if n is None:
            n = len(self.ui.roi)
        menu = self.roinumbermenu['menu']
        menu.delete(0,'end')
        for s in [str(i) for i in range(len(self.ui.roi))]:
            menu.add_command(label=s,command = tk._setit(self.currentroi,s,self.roinumber_callback))
        self.roilist = [str(i) for i in range(len(self.ui.roi))]

    # updates blast segmentation upon slider release only
    def updatet1threshold(self,event=None,currentslice=True):
        # force recalc of gates
        self.ui.data['gates'][1] = None
        self.ui.data['gates'][2] = None
        self.ui.runblast(currentslice=currentslice)
        self.t1sliderlabel['text'] = '{:.1f}'.format(self.currentt1threshold.get())
        # if operating in finalROI mode additionally reprocess the final segmentation
        if self.finalROI_overlay_value.get() == True:
            # still working on a fast 2d update to final ROI
            self.ROIclick(do3d=True)

    # updates the text field showing the value during slider drag
    def updatet1label(self,event):
        self.t1sliderlabel['text'] = '{:.1f}'.format(self.currentt1threshold.get())

    def updatet2threshold(self,event=None,currentslice=True):
        # force recalc of gates
        self.ui.data['gates'][1] = None
        self.ui.data['gates'][2] = None
        self.ui.runblast(currentslice=currentslice)
        self.t2sliderlabel['text'] = '{:.1f}'.format(self.currentt2threshold.get())
        if self.finalROI_overlay_value.get() == True:
            self.ROIclick(do3d=True)
        return
    
    def updatet2label(self,event):
        self.t2sliderlabel['text'] = '{:.1f}'.format(self.currentt2threshold.get())

    def updatebcsize(self,event=None):
        self.ui.data['gates'][0] = None
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
            self.update_layermenu_options('seg')

    def enhancingROI_overlay_callback(self,event=None):
        if self.enhancingROI_overlay_value.get() == False:
            self.ui.dataselection = 'raw'
            self.ui.updateslice()
        else:
            self.ui.dataselection = 'seg_raw_fusion_d'
            self.ui.updateslice(wl=True)
            self.finalROI_overlay_value.set(False)
            self.update_layermenu_options('blast')

    def enhancingROI_callback(self,event=None):
        self.finalROI_overlay_value.set(False)
        self.enhancingROI_overlay_value.set(True)
        self.update_layermenu_options('blast')
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
        self.update_layermenu_options('seg')

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
                self.createROI(int(event.xdata),int(event.ydata),self.ui.currentslice)
            
        self.closeROI(self.ui.data['seg_raw'],self.ui.get_currentslice(),do3d=do3d)
        self.ROIstats()
        fusionstack = np.zeros((155,240,240,2))
        if False:
            for slice in range(0,155):  
            # newfusion = imfuse(objectmask_filled[slice,:,:],t1mpragestack[slice,:,:],'blend')
                newfusion = 0.5*self.ui.data['wt'][slice,:,:] + 0.5*self.ui.data['raw'][0][slice,:,:]
                segfusionstack[slice] = newfusion
        else:
            fusionstack = OverlayPlots.generate_overlay(self.ui.data['raw'],self.ui.data['seg'],self.ui.roiframe.layer.get())
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
    
    def createROI(self,x,y,slice):
        self.ui.roi.append(ROI(x,y,slice))
        self.ui.currentroi += 1
        self.ui.roi[self.ui.currentroi].data = copy.deepcopy(self.ui.data)
        self.update_roinumber_options()

    def closeROI(self,metmaskstack,currentslice,do3d=True):
        # this method needs tidy-up
        # a quick config for ET, WT smoothing
        xpos = self.ui.roi[self.ui.currentroi].x
        ypos = self.ui.roi[self.ui.currentroi].y
        mlist = {'et':{'threshold':3,'dball':10,'dcube':2},
                    'wt':{'threshold':1,'dball':10,'dcube':2}}
        for m in mlist.keys():
            mask = (metmaskstack >= mlist[m]['threshold']).astype('double')
            # TODO: 2d versus 3d connected components?
            CC_labeled = cc3d.connected_components(mask,connectivity=26)
            stats = cc3d.statistics(CC_labeled)

            # objectnumber = CC_labeled(ypos,xpos,s.SliceNumber)
            # NB. one click is assumed to be in both compartments here
            objectnumber = CC_labeled[currentslice,ypos,xpos]

            # objectmask = ismember(CC_labeled,objectnumber)
            objectmask = (CC_labeled == objectnumber).astype('double')
            # currently there is no additional processing on ET or WT
            self.ui.data[m] = objectmask.astype('uint8')


            # calculate tc
            if m == 'et':
                objectmask_closed = np.zeros(np.shape(self.ui.data['raw']))
                objectmask_final = np.zeros(np.shape(self.ui.data['raw']))

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
                    se = ball(mlist[m]['dball'])
                    start = time.time()
                    objectmask_cp = cp.array(objectmask)
                    se_cp = cp.array(se)
                    # TODO: iterate binary_closing?
                    close_object_cucim = cucim_binary_closing(objectmask_cp,footprint=se_cp)
                    objectmask_closed = np.array(close_object_cucim.get())
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
                    # 2d preview mode. not sure if this is going to be useful
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
                    objectmask_filled = flood_fill(objectmask_filled,(currentslice,ypos,xpos),True)
                    objectmask_final = objectmask_filled.astype('int')
                    self.ui.data['tc'] = objectmask_final
                else:
                    se2 = square(mlist[m]['dcube'])
                    objectmask_filled = binary_dilation(objectmask_closed[currentslice,:,:],se2)
                    objectmask_filled = flood_fill(objectmask_filled,(ypos,xpos),True)
                    objectmask_final[currentslice,:,:] = objectmask_filled.astype('int')     
                    self.ui.data['tc'] = objectmask_final.astype('uint8')

            # update WT with smoothed TC
            elif m == 'wt':
                self.ui.data['wt'] = self.ui.data['wt'] | self.ui.data['tc']

        # nnunet convention for labels
        self.ui.data['seg'] = 1*self.ui.data['et'] + 1*self.ui.data['tc'] + self.ui.data['wt']
        return None
    
    def saveROI(self,roi=None):
        # Save ROI data
        outputpath = self.ui.caseframe.casedir
        fileroot = os.path.join(outputpath,self.ui.caseframe.casefile_prefix + self.ui.caseframe.casename.get())
        filename = fileroot+'_stats.pkl'
        # t1mprage template? need to save separately?

        # BLAST outputs. combined ROI or separate? doing separate for now
        roisuffix = ''
        for img in ['seg','et','tc','wt']:
            for roi in self.roilist:
                if len(self.roilist) > 1:
                    roisuffix = '_roi'+roi
                outputfilename = fileroot + '_blast_' + img + roisuffix + '.nii'
                self.WriteImage(self.ui.roi[int(roi)].data[img],outputfilename)
        # manual outputs. for now these have only one roi
        for img in ['manual_et','manual_tc','manual_wt']:
            outputfilename = fileroot + '_' + img + '.nii'
            self.WriteImage(self.ui.data[img],outputfilename)
            # sitk.WriteImage(sitk.GetImageFromArray(self.ui.data[img]),fileroot+'_'+img+'.nii')

        with open(filename,'ab') as fp:
            pickle.dump(dict(self.ui.stats),fp)
        # matlab compatible output
        filename = filename[:-3] + 'mat'
        with open(filename,'ab') as fp:
            savemat(filename,dict(self.ui.stats))


    def WriteImage(self,img_arr,filename):
        # for now output only segmentations so uint8
        img = sitk.GetImageFromArray(img_arr.astype('uint8'))
        # img.CopyInformation(self.ui.data['nifti'])
        # sitk.WriteImage(img,filename)
        writer = sitk.ImageFileWriter()
        writer.SetImageIO('NiftiImageIO')
        writer.SetFileName(filename)
        writer.Execute(img)

    #     filename = "t1ce_" + self.casename
    #     outputpath = os.path.join(self.config.UIdataroot+self.casename,filename)
    #     save(outputpath + "/" + filename + ".mat",'greengate_count','redgate_count','objectmask','objectmask_filled',
    # 'manualmasket','manualmasktc','centreimage','specificity_et','sensitivity_et','dicecoefficient_et','specificity_tc',
    # 'sensitivity_tc','dicecoefficient_tc','b','b2','manualmask_et_volume','manualmask_tc_volume','objectmask_filled_volume',
    # 'cumulative_elapsed_time')
    #     niftiwrite(objectmask,outputpath + "/" + filename + ".nii")
    #     niftiwrite(objectmask_filled,outputpath + "/" + filename + "_filled" + ".nii")
    #     niftiwrite(manualmasket,outputpath + "/" + filename + "_manualmask_et" + ".nii")
    #     niftiwrite(manualmasktc,outputpath + "/" + filename + "_manualmask_tc" + ".nii")
    #     niftiwrite(t1mprage_template,outputpath + "/" + 't1mprage_template.nii')
        return


    def updateROI(self):
        # save current dataset into the current roi. 
        self.ui.roi[self.ui.currentroi].data = copy.deepcopy(self.ui.data)
        self.ROIstats()

    def clearROI(self):
        self.ui.roi.pop(self.ui.currentroi)
        self.ui.currentroi -= 1
        self.ui.updateslice()

    def ROIstats(self):
        
        for t in ['et','tc','wt']:
            sums = self.ui.data['manual_'+t] + self.ui.data[t]
            subs = self.ui.data['manual_'+t] - self.ui.data[t]
                    
            TP = len(np.where(sums == 2))
            FP = len(np.where(subs == -1))
            TN = len(np.where(sums == 0))
            FN = len(np.where(subs == 1))

            self.ui.stats['spec'][t][self.currentroi.get()] = TN/(TN+FP)
            self.ui.stats['sens'][t][self.currentroi.get()] = TP/(TP+FN)
            self.ui.stats['dice'][t][self.currentroi.get()] = dice(self.ui.data['manual_'+t].flatten(),self.ui.data[t].flatten()) 

        # Calculate volumes
            self.ui.stats['vol']['manual_'+t][self.currentroi.get()] = len(np.where(self.ui.data['manual_'+t]))
            self.ui.stats['vol'][t][self.currentroi.get()] = len(np.where(self.ui.data['tc']))

        # copy gate counts
            self.ui.stats['t1gate_count'] = self.ui.data['gates'][3]
            self.ui.stats['t2gate_count'] = self.ui.data['gates'][4]

