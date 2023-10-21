import os,sys
import numpy as np
import pickle
import copy
import logging
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
from scipy.spatial.distance import dice
from scipy.ndimage import binary_closing as scipy_binary_closing
from scipy.io import savemat
if os.name == 'posix':
    from cucim.skimage.morphology import binary_closing as cucim_binary_closing
elif os.name == 'nt':
    from cupyx.scipy.ndimage import binary_closing as cupy_binary_closing
import cupy as cp
import cc3d

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
        self.currentroi = tk.IntVar(value=-1)
        self.slicevolume_norm = tk.IntVar()
        self.roilist = []

        ########################
        # layout for the buttons
        ########################

        self.frame.grid(column=2,row=3,rowspan=5,sticky='e')

        # normal slice button
        if False:
            normal_frame = ttk.Frame(self.frame,padding='0')
            normal_frame.grid(row=0,column=0,sticky='w')
            normalSlice = ttk.Button(normal_frame,text='normal',command=self.normalslice_callback)
            normalSlice.grid(column=0,row=0,sticky='w')
            slicevolume_slice_button = ttk.Radiobutton(normal_frame,text='slice',variable=self.slicevolume_norm,value=0)
            slicevolume_slice_button.grid(row=0,column=1,sticky='w')
            slicevolume_volume_button = ttk.Radiobutton(normal_frame,text='vol.',variable=self.slicevolume_norm,value=1)
            slicevolume_volume_button.grid(row=0,column=2,sticky='w')

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
        # self.currentroi.trace_add('write',lambda *args: self.currentroi.get())
        self.currentroi.trace_add('write',self.set_currentroi)
        self.roinumbermenu = ttk.OptionMenu(roinumberframe,self.currentroi,*self.roilist,command=self.roinumber_callback)
        self.roinumbermenu.config(width=2)
        self.roinumbermenu.grid(column=1,row=0,sticky='w')
        self.roinumbermenu.configure(state='disabled')

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

    # ROI methods
    # 

    # methods for layer options menu
    def layer_callback(self,layer=None):
        self.ui.currentlayer = self.layer.get()
        roi = self.ui.get_currentroi()
        # generate a new overlay
        if self.layertype.get() == 'blast':
            self.ui.roi[roi].data['seg_raw_fusion'] = OverlayPlots.generate_overlay(self.ui.data['raw'],self.ui.data['seg_raw'],self.layer.get())
            self.ui.roi[roi].data['seg_raw_fusion_d'] = copy.copy(self.ui.data['seg_raw_fusion'])
        elif self.layertype.get() == 'seg':
            self.ui.roi[roi].data['seg_fusion'] = OverlayPlots.generate_overlay(
                self.ui.roi[roi].data['raw'],self.ui.roi[roi].data['seg'],self.layer.get())
            self.ui.roi[roi].data['seg_fusion_d'] = copy.copy(self.ui.roi[roi].data['seg_fusion'])
        self.updateData()
        self.ui.updateslice()

    def update_layermenu_options(self,type):
        self.layertype.set(type)
        menu = self.layermenu['menu']
        menu.delete(0,'end')
        for s in self.layerlist[type]:
            menu.add_command(label=s,command = tk._setit(self.layer,s,self.layer_callback))
        self.layer.set(self.layerlist[type][0])

    # methods for roi number choice menu
    def roinumber_callback(self,item=None):
        self.ui.set_currentroi()
        # reference or copy
        self.updateData()
        self.layer_callback()
        self.ui.updateslice()
        return
    
    def update_roinumber_options(self,n=None):
        if n is None:
            n = len(self.ui.roi)
        menu = self.roinumbermenu['menu']
        menu.delete(0,'end')
        for s in [str(i) for i in range(n)]:
            menu.add_command(label=s,command = tk._setit(self.currentroi,s,self.roinumber_callback))
        self.roilist = [str(i) for i in range(n)]
        if n:
            self.roinumbermenu.configure(state='active')
        else:
            self.roinumbermenu.configure(state='disabled')
            self.finalROI_overlay_value.set(False)

    def set_currentroi(self,var,index,mode):
        if mode == 'write':
            self.ui.set_currentroi()    

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
                self.createROI(int(event.xdata),int(event.ydata),self.ui.get_currentslice())
            
        roi = self.ui.get_currentroi()
        self.closeROI(self.ui.roi[roi].data['seg_raw'],self.ui.get_currentslice(),do3d=do3d)
        self.ROIstats()
        fusionstack = np.zeros((155,240,240,2))
        fusionstack = OverlayPlots.generate_overlay(self.ui.roi[roi].data['raw'],self.ui.roi[roi].data['seg'],self.ui.roiframe.layer.get())
        self.ui.roi[roi].data['seg_fusion'] = fusionstack
        self.ui.roi[roi].data['seg_fusion_d'] = copy.copy(self.ui.roi[roi].data['seg_fusion'])
        # current roi populates data dict
        self.updateData()

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
        self.currentroi.set(self.currentroi.get() + 1)
        self.ui.roi[self.ui.currentroi].data = copy.deepcopy(self.ui.data)
        self.update_roinumber_options()

    def closeROI(self,metmaskstack,currentslice,do3d=True):
        # this method needs tidy-up
        # a quick config for ET, WT smoothing
        roi = self.ui.get_currentroi()
        xpos = self.ui.roi[roi].x
        ypos = self.ui.roi[roi].y
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
            self.ui.roi[roi].data[m] = objectmask.astype('uint8')


            # calculate tc
            if m == 'et':
                objectmask_closed = np.zeros(np.shape(self.ui.data['raw'])[1:])
                objectmask_final = np.zeros(np.shape(self.ui.data['raw'])[1:])

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
                    self.ui.roi[roi].data['tc'] = objectmask_final
                else:
                    se2 = square(mlist[m]['dcube'])
                    objectmask_filled = binary_dilation(objectmask_closed[currentslice,:,:],se2)
                    objectmask_filled = flood_fill(objectmask_filled,(ypos,xpos),True)
                    objectmask_final[currentslice,:,:] = objectmask_filled.astype('int')     
                    self.ui.roi[roi].data['tc'] = objectmask_final.astype('uint8')

            # update WT with smoothed TC
            elif m == 'wt':
                self.ui.roi[roi].data['wt'] = self.ui.roi[roi].data['wt'] | self.ui.roi[roi].data['tc']

        # nnunet convention for labels
        self.ui.roi[roi].data['seg'] = 1*self.ui.roi[roi].data['et'] + \
                                                    1*self.ui.roi[roi].data['tc'] + \
                                                    1*self.ui.roi[roi].data['wt']
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

        sdict = {}
        msdict = {} 
        for i,r in enumerate(self.ui.roi):
            sdict['roi'+str(i)] = r.stats
            msdict['roi'+str(i)] = {}
            mdict = msdict['roi'+str(i)]
            mdict['greengate_count'] = r.stats['gatecount']['t2']
            mdict['redgate_count'] = r.stats['gatecount']['t1']
            mdict['objectmask'] = r.data['et']
            mdict['objectmask_filled'] = r.data['tc']
            mdict['manualmasket'] = r.data['manual_et']
            mdict['manualmasktc'] = r.data['manual_tc']
            mdict['centreimage'] = 0
            mdict['specificity_et'] = r.stats['spec']['et']
            mdict['sensitivity_et'] = r.stats['sens']['et']
            mdict['dicecoefficient_et'] = r.stats['dsc']['et']
            mdict['specificity_tc'] = r.stats['spec']['tc']
            mdict['sensitivity_tc'] = r.stats['sens']['tc']
            mdict['dicecoefficient_tc'] = r.stats['dsc']['tc']
            mdict['specificity_wt'] = r.stats['spec']['wt']
            mdict['sensitivity_wt'] = r.stats['sens']['wt']
            mdict['dicecoefficient_wt'] = r.stats['dsc']['wt']
            mdict['b'] = 0
            mdict['b2'] = 0
            mdict['manualmask_et_volume'] = r.stats['vol']['manual_et']
            mdict['manualmask_tc_volume'] = r.stats['vol']['manual_tc']
            mdict['objectmask_filled_volume'] = r.stats['vol']['tc']
            mdict['cumulative_elapsed_time'] = r.stats['elapsed_time']

        with open(filename,'ab') as fp:
            pickle.dump(sdict,fp)
        # matlab compatible output
        filename = filename[:-3] + 'mat'
        with open(filename,'ab') as fp:
            savemat(filename,sdict)
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
        # rerun segmentation
        self.ROIclick()
        self.ROIstats()
        # save current dataset into the current roi. 
        self.ui.roi[self.ui.currentroi].data = copy.deepcopy(self.ui.data)

    def updateData(self):
        self.ui.data = copy.deepcopy(self.ui.roi[self.ui.currentroi].data)

    # eliminate one ROI if multiple ROIs in current case
    def clearROI(self):
        n = len(self.ui.roi)
        if n:    
            self.ui.roi.pop(self.ui.currentroi)
            if self.ui.currentroi > 0 or len(self.ui.roi)==0:
                # new current roi is decremented as an arbitrary choice
                # or if all rois are now gone
                self.currentroi.set(self.currentroi.get()-1)
            self.update_roinumber_options()
            if len(self.ui.roi):
                self.roinumber_callback()
            else:
                self.ui.dataselection='raw'
                self.ui.updateslice()

    # eliminate all ROIs, ie for loading another case
    def resetROI(self):
        self.currentroi.set(-1)
        self.ui.roi = []
        self.ui.roiframe.finalROI_overlay_value.set(False)
        self.ui.roiframe.enhancingROI_overlay_value.set(False)
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
        for t in ['et','tc','wt']:
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
            self.ui.roi[roi].stats['gatecount']['t1'] = self.ui.roi[roi].data['gates'][3]
            self.ui.roi[roi].stats['gatecount']['t2'] = self.ui.roi[roi].data['gates'][4]

