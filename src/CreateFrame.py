import os,sys
import numpy as np
import glob
import copy
import re
import logging
import tkinter as tk
from tkinter import ttk,StringVar,DoubleVar,PhotoImage
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import SimpleITK as sitk
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice
import cc3d

from NavigationBar import NavigationBar
from FileDialog import FileDialog
import OverlayPlots

# base for various frames
class CreateFrame():
    def __init__(self,frame,ui=None,padding='10'):
        self.ui = ui
        self.parentframe = frame # parent
        self.frame = ttk.Frame(self.parentframe,padding=padding)
        self.config = self.ui.config
        self.padding = padding


class CreateSliceViewerFrame(CreateFrame):
    def __init__(self,parentframe,ui=None,padding='10'):
        super().__init__(parentframe,ui=ui)
        self.currentslice = tk.IntVar()
        self.currentslice.set(75)
        self.slicevolume_norm = tk.IntVar()
        # window/level values
        self.window = np.array([1.,1.],dtype='float')
        self.level = np.array([0.5,0.5],dtype='float')
        self.wlflag = False


        self.frame.grid(column=0,row=0, sticky='NEW',in_=self.parentframe)
        # self.frame.rowconfigure(0,weight=2)
        # self.frame.rowconfigure(1,weight=10)
        # slice viewer widget
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(121)
        ax.axis('off')
        fig.tight_layout(pad=0)
        self.ax_img = ax.imshow(np.zeros((240,240)),vmin=0,vmax=1,cmap='gray')
        ax2 = fig.add_subplot(122,sharex=ax,sharey=ax)
        ax2.axis('off')
        fig.tight_layout(pad=0)
        self.ax2_img = ax2.imshow(np.zeros((240,240)),vmin=0,vmax=1,cmap='gray')

        self.canvas = FigureCanvasTkAgg(fig, master=self.frame)  
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=1, columnspan=3, rowspan=2)
        self.tbar = NavigationBar(self.canvas,self.frame,pack_toolbar=False)
        self.tbar.children['!button4'].pack_forget() # get rid of configure plot
        self.tbar.grid(column=0,row=3,columnspan=3,sticky='w')

        # slice selection slider
        slidersframe = ttk.Frame(self.frame,padding='5')
        slidersframe.grid(column=0,row=4,columnspan=3,sticky='new')
        # slidersframe.columnconfigure(0,weight=2)
        # slidersframe.columnconfigure(1,weight=10)
        # slidersframe.columnconfigure(2,weight=1)
        slicetextlabel = ttk.Label(slidersframe, text='Slice: ')
        slicetextlabel.grid(column=0,row=0,sticky='w')
        self.vsliceslider = ttk.Scale(slidersframe,from_=0,to=154,variable=self.currentslice,
                                      length='3i',command=self.updateslice)
        self.vsliceslider.grid(column=1,row=0,sticky='w')
        self.vslicenumberlabel = ttk.Label(slidersframe, text='{}'.format(self.currentslice.get()))
        self.vslicenumberlabel.grid(column=2,row=0)

        # will use touchpad/mouse instead of onscreen widgets
        # self.window = tk.DoubleVar(value=1)
        # self.level = tk.DoubleVar(value=0.5)
        # wtext = ttk.Label(slidersframe,text='W: ')
        # wtext.grid(row=1, column=0,sticky='w')
        # self.wslider = ttk.Scale(slidersframe,from_=0,to=2,variable=self.window,command=self.updatewl,length='3i')
        # self.wslider.grid(row=1,column=1,sticky='w')
        # self.wslidernumberlabel = ttk.Label(slidersframe,text='{}'.format(self.window.get()))
        # self.wslidernumberlabel.grid(row=1,column=2)
        # ltext = ttk.Label(slidersframe,text='L: ')
        # ltext.grid(row=2, column=0,sticky='w')
        # self.lslider = ttk.Scale(slidersframe,from_=0,to=2,variable=self.level,command=self.updatewl,length='3i')
        # self.lslider.grid(row=2,column=1,sticky='w')
        # self.lslidernumberlabel = ttk.Label(slidersframe,text='{}'.format(self.level.get()))
        # self.lslidernumberlabel.grid(row=2,column=2)

        # normal slice button
        normal_frame = ttk.Frame(self.frame,padding='0')
        normal_frame.grid(row=5,column=0,sticky='w')
        normalSlice = ttk.Button(normal_frame,text='normal',command=self.normalslice_callback)
        normalSlice.grid(column=0,row=0,sticky='w')
        slicevolume_slice_button = ttk.Radiobutton(normal_frame,text='slice',variable=self.slicevolume_norm,value=0)
        slicevolume_slice_button.grid(row=0,column=1,sticky='w')
        slicevolume_volume_button = ttk.Radiobutton(normal_frame,text='vol.',variable=self.slicevolume_norm,value=1)
        slicevolume_volume_button.grid(row=0,column=2,sticky='w')

        # messages text frame
        self.messagelabel = ttk.Label(self.frame,text=self.ui.message.get(),padding='5',borderwidth=0)
        self.messagelabel.grid(row=6,column=0,columnspan=3,sticky='ew')

        if self.ui.OS in ('win32','darwin'):
            # self.frame.bind('<MouseWheel>', self.updatew2())
            # self.frame.bind('<Shift-MouseWheel>', self.updatel2())
            self.ui.root.bind('<Button>',self.touchpad)
        if self.ui.OS == 'linux':
            self.ui.root.bind('<Button>',self.touchpad)
            # self.ui.root.bind('<ButtonRelease>',self.touchpad)

    # TODO: different bindings and callbacks need some organization
    def updateslice(self,event=None,wl=False,blast=False):
        slice=self.currentslice.get()
        self.ui.set_currentslice()
        if blast: # option for previewing enhancing in 2d
            self.ui.runblast(currentslice=slice)
        self.ax_img.set(data=self.ui.data[self.ui.dataselection][0,slice,:,:])
        self.ax2_img.set(data=self.ui.data[self.ui.dataselection][1,slice,:,:])
        self.vslicenumberlabel['text'] = '{}'.format(slice)
        if self.ui.dataselection in['seg_raw_fusion_d','seg_fusion_d']:
            self.ax_img.set(cmap='viridis')
            self.ax2_img.set(cmap='viridis')
        else:
            self.ax_img.set(cmap='gray')
            self.updatewl(ax=0)
            self.ax2_img.set(cmap='gray')
            self.updatewl(ax=1)
        if wl:   
            # possible latency problem here
            if self.ui.dataselection in ['seg_raw_fusion_d','seg_fusion_d']:
                # self.updatewl_fusion()
                self.ui.roiframe.layer_callback(updateslice=False,updatedata=False)
            elif self.ui.dataselection == 'raw':
                self.clipwl_raw()
        self.canvas.draw()

    # special update for previewing BLAST enhancing lesion in 2d
    def updateslice_blast(self,event=None):
        slice = self.currentslice.get()
        self.ui.set_currentslice()
        self.ui.runblast(currentslice=slice)
        self.vslicenumberlabel['text'] = '{}'.format(slice)
        self.canvas.draw()

    # update for previewing final segmentation in 2d
    # probably can't be implemented at this time.
    def updateslice_roi(self,event=None):
        slice = self.currentslice.get()
        self.ui.set_currentslice(slice)
        self.ui.runblast(currentslice=slice)
        # 2d preview of final segmentation will still require 3d connected components, 
        # need to fix this.
        self.ui.roiframe.ROIclick(do3d=True)
        self.vslicenumberlabel['text'] = '{}'.format(slice)
        self.canvas.draw()
        
    # TODO: latency problem for fusions. 
    # for now, don't allow to call this function if overlay is selected
    def updatewl(self,ax=0,lval=None,wval=None):

        self.wlflag = True
        if lval:
            self.level[ax] += lval
        if wval:
            self.window[ax] += wval

        vmin = self.level[ax] - self.window[ax]/2
        vmax = self.level[ax] + self.window[ax]/2

        if ax==0:
            self.ax_img.set_clim(vmin=vmin,vmax=vmax)
        else:
            self.ax2_img.set_clim(vmin=vmin,vmax=vmax)
        self.canvas.draw()

    # color window/level scaling needs to be done separately for latency
    # for now, just tack it onto the fusion toggle button
    def updatewl_fusion(self):
        if self.ui.dataselection in ['seg_raw_fusion_d','seg_fusion_d']:
            for ax in range(2):
                vmin = self.level[ax] - self.window[ax]/2
                vmax = self.level[ax] + self.window[ax]/2
                self.ui.data['raw'][ax] = self.ui.caseframe.rescale(self.ui.data['raw_copy'][ax],vmin=vmin,vmax=vmax)

    # clip the raw data to window and level settings
    def clipwl_raw(self):
        for ax in range(2):
            vmin = self.level[ax] - self.window[ax]/2
            vmax = self.level[ax] + self.window[ax]/2
            self.ui.data['raw'][ax] = self.ui.caseframe.rescale(self.ui.data['raw'][ax],vmin=vmin,vmax=vmax)

    def restorewl_raw(self):
        self.ui.data['raw'] = copy.deepcopy(self.ui.data['raw_copy'])

            
    # touchpad event for window/level adjustment
    # TODO: extend to mouse and windows
    def touchpad(self,event):
        # print(event)
        # only allow adjustment in raw data view. overlays have latency to scale in 3d.
        if self.ui.dataselection != 'raw':
            return
        if event.y < 0 and event.y > 400:
            return
        if event.x <=400:
            ax = 0
        else:
            ax = 1
        if self.ui.OS == 'linux':
            if event.state:
                if event.num == 4:
                    # increment is hard-coded
                    self.updatewl(ax=ax,wval=.01)
                elif event.num == 5:
                    self.updatewl(ax=ax,wval=-.01)
            else:
                if event.num == 4:
                    self.updatewl(ax=ax,lval=.01)
                elif event.num == 5:
                    self.updatewl(ax=ax,lval=-.01)
        elif self.ui.OS == 'nt':
            if event.state:
                if event.num == 4:
                    # increment is hard-coded
                    self.updatewl(ax=ax,wval=.01)
                elif event.num == 5:
                    self.updatewl(ax=ax,wval=-.01)
            else:
                if event.num == 4:
                    self.updatewl(ax=ax,lval=.01)
                elif event.num == 5:
                    self.updatewl(ax=ax,lval=-.01)


    def normalslice_callback(self,event=None):
        # do kmeans
        # Creates a matrix of voxels for normal brain slice
        # Gating Routine

        if self.slicevolume_norm.get() == 0:
            self.normalslice=self.ui.get_currentslice()
            region_of_support = np.where(self.ui.data['raw'][0,self.normalslice]>=0) 
            t1channel_normal = self.ui.data['raw'][0,self.normalslice][region_of_support]
            t2channel_normal = self.ui.data['raw'][1,self.normalslice][region_of_support]
        else:
            self.normalslice = None
            region_of_support = np.where(self.ui.data['raw'][0]>=0) 
            t1channel_normal = self.ui.data['raw'][0][region_of_support]
            t2channel_normal = self.ui.data['raw'][1][region_of_support]

        # kmeans to calculate statistics for brain voxels
        t2 = np.ravel(t2channel_normal)
        t1 = np.ravel(t1channel_normal)
        X = np.column_stack((t2,t1))
        # rng(1)
        np.random.seed(1)
        # [idx,C] = KMeans(n_clusters=2).fit(X)
        kmeans = KMeans(n_clusters=2,n_init='auto').fit(X)

        if False:
            plt.figure(2)
            ax = plt.subplot(1,1,1)
            plt.scatter(X[kmeans.labels_==0,0],X[kmeans.labels_==0,1],c='b')
            plt.scatter(X[kmeans.labels_==1,0],X[kmeans.labels_==1,1],c='r')
            ax.set_aspect('equal')
            ax.set_xlim(left=0,right=0.6)
            ax.set_ylim(bottom=0,top=1.0)
            plt.show(block=False)

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
        self.ui.roiframe.updatet1threshold(currentslice=None)
        self.ui.dataselection = 'seg_raw_fusion_d'
        self.ui.roiframe.enhancingROI_overlay_value.set(True)
        self.ui.roiframe.finalROI_overlay_value.set(False)



class CreateCaseFrame(CreateFrame):
    def __init__(self,parent,ui=None,load=True):
        super().__init__(parent,ui=ui)

        self.fd = FileDialog(initdir=self.config.UIdatadir)
        self.datadir = StringVar()
        self.datadir.set(self.fd.dir)
        self.casename = StringVar()
        self.casefile_prefix = None
        self.caselist = []
        self.n4_check_value = tk.BooleanVar(value=True)

        # case selection
        caseframe = ttk.Frame(parent,padding='5')
        caseframe.grid(row=0,column=0,columnspan=3,sticky='ew')
        # datadir
        self.fdbicon = PhotoImage(file=os.path.join(self.config.UIResourcesPath,'folder_icon_16.png'))
        # select a parent dir for a group of case sub-dirs
        self.fdbutton = ttk.Button(caseframe,image=self.fdbicon, command=self.select_dir)
        self.fdbutton.grid(row=0,column=2)
        self.datadirentry = ttk.Entry(caseframe,width=40,textvariable=self.datadir)
        self.datadirentry.bind('<Return>',self.datadirentry_callback)
        self.datadirentry.grid(column=3,row=0,columnspan=5)
        caselabel = ttk.Label(caseframe, text='Case: ')
        caselabel.grid(column=0,row=0,sticky='we')
        self.casename.trace_add('write',self.case_callback)
        self.w = ttk.Combobox(caseframe,width=6,textvariable=self.casename,values=self.caselist)
        self.w.grid(column=1,row=0)
        self.n4_check = ttk.Checkbutton(caseframe,text='N4',variable=self.n4_check_value)
        self.n4_check.grid(row=0,column=8,sticky='w')

        # initialize default directory
        self.datadirentry_callback(load=load)

    # callback for file dialog 
    def select_dir(self):
        self.fd.select_dir()
        self.datadir.set(self.fd.dir)
        self.datadirentry.update()
        self.datadirentry_callback()

    def case_callback(self,casevar=None,val=None,event=None):
        case = self.casename.get()
        self.ui.set_casename()
        print('Loading case {}'.format(case))
        self.loadCase()
        self.ui.dataselection = 'raw'
        self.ui.sliceviewerframe.tbar.home()
        self.ui.roiframe.resetROI()
        self.ui.updateslice()
        self.ui.starttime()

    def loadCase(self,case=None):
        if case is not None:
            self.casename.set(case)
            self.ui.set_casename()
        self.casedir = os.path.join(self.datadir.get(),self.config.UIdataroot+self.casename.get())
        files = os.listdir(self.casedir)
        # create t1mprage template
        t1ce_file = next((f for f in files if 't1ce' in f),None)
        t1ce = sitk.ReadImage(os.path.join(self.casedir,t1ce_file))
        img_arr = sitk.GetArrayFromImage(t1ce)
        # 2 channels hard-coded
        self.ui.data['raw'] = np.zeros((2,)+np.shape(img_arr),dtype='float32')
        self.ui.data['raw'][0] = img_arr

        # Create t2flair template 
        t2flair_file = next((f for f in files if 'flair' in f),None)
        t2flair = sitk.ReadImage(os.path.join(self.casedir,t2flair_file))
        img_arr = sitk.GetArrayFromImage(t2flair)
        self.ui.data['raw'][1] = img_arr

        # bias correction. by convention, any pre-corrected files should have 'bias' in the filename
        if self.n4_check_value.get() and 'bias' not in t1ce_file:  
            self.n4()
        # rescale the data
        for ch in range(np.shape(self.ui.data['raw'])[0]):
            self.ui.data['raw'][ch] = self.rescale(self.ui.data['raw'][ch])

        # save copy of the raw data
        self.ui.data['raw_copy'] = copy.deepcopy(self.ui.data['raw'])

        # create the label
        label = sitk.ReadImage(os.path.join(self.casedir,self.config.UIdataroot+self.casename.get()+'_seg.nii'))
        img_arr = sitk.GetArrayFromImage(label)
        self.ui.data['label'] = img_arr

        # supplementary labels. brats and nnunet conventions are differnt.
        if False: # nnunet
            self.ui.data['manual_et'] = (self.ui.data['label'] == 3).astype('int') #enhancing tumor 
            self.ui.data['manual_tc'] = (self.ui.data['label'] >= 2).astype('int') #tumour core
            self.ui.data['manual_wt'] = (self.ui.data['label'] >= 1).astype('int') #whole tumour
        else: # brats
            self.ui.data['manual_et'] = (self.ui.data['label'] == 4).astype('int') #enhancing tumor 
            self.ui.data['manual_tc'] = ((self.ui.data['label'] == 1) | (self.ui.data['label'] == 4)).astype('int') #tumour core
            self.ui.data['manual_wt'] = (self.ui.data['label'] >= 1).astype('int') #whole tumour


    # operates on a single image channel 
    def rescale(self,img_arr,vmin=None,vmax=None):
        scaled_arr =  np.zeros(np.shape(img_arr))
        if vmin is None:
            minv = np.min(img_arr)
        else:
            minv = vmin
        if vmax is None:
            maxv = np.max(img_arr)
        else:
            maxv = vmax
        assert(maxv>minv)
        scaled_arr = (img_arr-minv) / (maxv-minv)
        scaled_arr = np.clip(scaled_arr,a_min=0,a_max=1)
        return scaled_arr
    
    def n4(self,shrinkFactor=4,nFittingLevels=4):
        # self.ui.set_message('Performing N4 bias correction')
        print('N4 bias correction')
        for ch in range(np.shape(self.ui.data['raw'])[0]):
            data = self.ui.data['raw'][ch]
            dataImage = sitk.Cast(sitk.GetImageFromArray(data),sitk.sitkFloat32)
            sdataImage = sitk.Shrink(dataImage,[shrinkFactor]*dataImage.GetDimension())
            maskImage = sitk.Cast(sitk.GetImageFromArray(np.where(data,True,False).astype('uint8')),sitk.sitkUInt8)
            maskImage = sitk.Shrink(maskImage,[shrinkFactor]*maskImage.GetDimension())
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            lowres_img = corrector.Execute(sdataImage,maskImage)
            log_bias_field = corrector.GetLogBiasFieldAsImage(dataImage)
            log_bias_field_arr = sitk.GetArrayFromImage(log_bias_field)
            corrected_img = dataImage / sitk.Exp(log_bias_field)
            corrected_img_arr = sitk.GetArrayFromImage(corrected_img)
            self.ui.data['raw'][ch] = corrected_img_arr
        return

    # main callback for selecting dir either by file dialog or text entry
    def datadirentry_callback(self,event=None,load=True):
        dir = self.datadir.get().strip()
        if os.path.exists(dir):
            files = os.listdir(dir)
            self.casefile_prefix = re.match('(^.*)0[0-9]{4}',files[0]).group(1)
            casefiles = [re.match('.*(0[0-9]{4})',f).group(1) for f in files if re.search('_0[0-9]{4}$',f)]
            self.ui.set_message('')
            if len(casefiles):
                # TODO: will need a better sort here
                self.caselist = sorted(casefiles)
                self.w['values'] = self.caselist
                # autoload first case
                if load:
                    self.casename.set(self.caselist[0])
                    self.ui.set_casename()
            else:
                print('No cases found in directory {}'.format(dir))
                self.ui.set_message('No cases found in directory {}'.format(dir))
        else:
            print('Directory {} not found.'.format(dir))
            self.ui.set_message('Directory {} not found.'.format(dir))
            self.w.config(state='disable')
            self.datadirentry.update()
        return
