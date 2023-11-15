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
from matplotlib.artist import Artist
matplotlib.use('TkAgg')
import SimpleITK as sitk
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice

from src.NavigationBar import NavigationBar
from src.FileDialog import FileDialog

# base for various frames
class CreateFrame():
    def __init__(self,frame,ui=None,padding='10'):
        self.ui = ui
        self.parentframe = frame # parent
        self.frame = ttk.Frame(self.parentframe,padding=padding)
        self.config = self.ui.config
        self.padding = padding


##############
# Slice Viewer
##############

class CreateSliceViewerFrame(CreateFrame):
    def __init__(self,parentframe,ui=None,padding='10'):
        super().__init__(parentframe,ui=ui)

        # ui variables
        self.currentslice = tk.IntVar(value=75)
        self.currentsagslice = tk.IntVar(value=120)
        self.currentcorslice = tk.IntVar(value=120)
        self.labels = {'Im_A':None,'Im_B':None,'Im_C':None,'W_A':None,'L_A':None,'W_B':None,'L_B':None}
        self.axslicelabel = None
        self.corslicelabel = None
        self.sagslicelabel = None
        self.windowlabel = None
        self.levellabel = None
        self.sagcordisplay = tk.IntVar(value=0)
        self.slicevolume_norm = tk.IntVar(value=1)
        # window/level values for T1,T2
        self.window = np.array([1.,1.],dtype='float')
        self.level = np.array([0.5,0.5],dtype='float')
        self.wlflag = False
        self.b1x = self.b1y = None # for tracking window/level mouse drags
        self.b3y = None # mouse drag for cor,sag slice
        # image dimensions
        self.dim = self.ui.config.ImageDim

        self.ui = ui

        self.frame.grid(column=0,row=0, sticky='NEW',in_=self.parentframe)
        # self.frame.rowconfigure(0,weight=2)
        # self.frame.rowconfigure(1,weight=10)
        # slice viewer widget
        slicefovratio = self.ui.config.ImageDim[0]/self.ui.config.ImageDim[1]
        fig,self.axs = plt.subplot_mosaic([['A','B','C'],['A','B','D']],
                                     width_ratios=[self.ui.config.PanelSize,self.ui.config.PanelSize,
                                                   self.ui.config.PanelSize / (2 * slicefovratio) ],
                                     figsize=((2*self.ui.config.PanelSize + 2/slicefovratio),4),dpi=100)
        self.ax_img = self.axs['A'].imshow(np.zeros((240,240)),vmin=0,vmax=1,cmap='gray')
        self.ax2_img = self.axs['B'].imshow(np.zeros((240,240)),vmin=0,vmax=1,cmap='gray')
        self.ax3_img = self.axs['C'].imshow(np.zeros((155,240)),vmin=0,vmax=1,cmap='gray')
        self.ax4_img = self.axs['D'].imshow(np.zeros((155,240)),vmin=0,vmax=1,cmap='gray')

        # add dummy axes for image labels. absolute canvas coords

        # 1. this dummy axis gets the 'A' axis position in figure coords, then within that range
        # reimposes a (0,1) range, thus resulting in a scaling of the data to figure transform
        # bbox = axs['A'].get_position()
        # axs['label'] = fig.add_axes(bbox)
        # 2. this dummy axis covers the entire canvas including all four subplots in the range 0,1
        # so the x-axis will be stretched analagously to 1. and break the transform
        # self.axs['label'] = fig.add_axes([0,0,1,1])
        # 3. this dummy axis covers the bottom half of subplot mosaic 'A' in range (0,1), which preserves the aspect
        # ratio and transform, and with a simple offset of +1 in y also gets to the top half of 'A'
        # in figure coordinates.
        self.axs['labelA'] = fig.add_subplot(2,3,4)

        self.axs['labelB'] = fig.add_subplot(2,3,5)
        self.axs['labelC'] = fig.add_subplot(2,3,3)
        self.axs['labelD'] = fig.add_subplot(2,3,6)
        # prevent labels from panning or zooming
        self.axs['labelA'].set_navigate(False)
        self.axs['labelB'].set_navigate(False)
        self.axs['labelC'].set_navigate(False)
        self.axs['labelD'].set_navigate(False)

        for a in self.axs.keys():
            self.axs[a].axis('off')
        # set up axis sharing
        self.axs['B']._shared_axes['x'].join(self.axs['B'],self.axs['A'])
        self.axs['B']._shared_axes['y'].join(self.axs['B'],self.axs['A'])
        fig.tight_layout(pad=0)

        # record the data to figure coords of each label for each axis
        self.xyfig={}
        figtrans={}
        for a in ['A','B','C','D']:
            figtrans[a] = self.axs[a].transData + self.axs[a].transAxes.inverted()
        self.xyfig['Im_A']= figtrans['A'].transform((5,15))
        self.xyfig['W_A'] = figtrans['A'].transform((int(self.dim[1]/2),self.dim[1]-10))
        self.xyfig['L_A'] = figtrans['A'].transform((int(self.dim[1]*3/4),self.dim[1]-10))
        self.xyfig['W_B'] = figtrans['B'].transform((int(self.dim[1]/2),self.dim[1]-10))
        self.xyfig['L_B'] = figtrans['B'].transform((int(self.dim[1]*3/4),self.dim[1]-10))
        self.xyfig['Im_C'] = figtrans['C'].transform((5,15))
        self.xyfig['Im_D'] = figtrans['D'].transform((5,15))

        self.canvas = FigureCanvasTkAgg(fig, master=self.frame)  
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(column=0, row=1, columnspan=3, rowspan=2)

        self.tbar = NavigationBar(self.canvas,self.frame,pack_toolbar=False,ui=self.ui,axs=self.axs)
        self.tbar.children['!button4'].pack_forget() # get rid of configure plot
        self.tbar.grid(column=0,row=3,columnspan=3,sticky='w')

        # for updating image labels during pan/zoom
        self.panzoom_id = self.canvas.callbacks.connect('button_press_event',self.update_labels)

        # normal slice button
        normal_frame = ttk.Frame(self.frame,padding='0')
        normal_frame.grid(row=4,column=0,sticky='w')
        normalSlice = ttk.Button(normal_frame,text='normal',command=self.normalslice_callback)
        normalSlice.grid(column=0,row=0,sticky='w')
        slicevolume_slice_button = ttk.Radiobutton(normal_frame,text='slice',variable=self.slicevolume_norm,value=0)
        slicevolume_slice_button.grid(row=0,column=1,sticky='w')
        slicevolume_volume_button = ttk.Radiobutton(normal_frame,text='vol.',variable=self.slicevolume_norm,value=1)
        slicevolume_volume_button.grid(row=0,column=2,sticky='w')

        # t1/t2 selection for sag/cor panes
        sagcordisplay_label = ttk.Label(normal_frame, text='Sag/Cor: ')
        sagcordisplay_label.grid(row=0,column=4,padx=(50,0),sticky='e')
        self.sagcordisplay_button = ttk.Radiobutton(normal_frame,text='T1',variable=self.sagcordisplay,value=0,
                                                    command=self.updateslice)
        self.sagcordisplay_button.grid(column=5,row=0,sticky='w')
        self.sagcordisplay_button = ttk.Radiobutton(normal_frame,text='T2',variable=self.sagcordisplay,value=1,
                                                    command=self.updateslice)
        self.sagcordisplay_button.grid(column=6,row=0,sticky='w')


        # messages text frame
        self.messagelabel = ttk.Label(self.frame,text=self.ui.message.get(),padding='5',borderwidth=0)
        self.messagelabel.grid(row=6,column=0,columnspan=3,sticky='ew')

        if self.ui.OS in ('win32','darwin'):
            # self.frame.bind('<MouseWheel>', self.updatew2())
            # self.frame.bind('<Shift-MouseWheel>', self.updatel2())
            self.ui.root.bind('<Button>',self.touchpad)
            self.ui.root.bind('<B1-Motion>',self.b1motion)
            self.ui.root.bind('<B3-Motion>',self.b3motion)
            self.ui.root.bind('<ButtonRelease-1>',self.b1release)
        if self.ui.OS == 'linux':
            self.ui.root.bind('<Button>',self.touchpad)
            self.ui.root.bind('<B1-Motion>',self.b1motion)
            self.ui.root.bind('<B3-Motion>',self.b3motion)
            self.ui.root.bind('<Button-3>',self.b3motion_reset)
            self.ui.root.bind('<ButtonRelease-1>',self.b1release)
            # self.ui.root.bind('<ButtonRelease>',self.touchpad)

    # TODO: different bindings and callbacks need some organization
    def updateslice(self,event=None,wl=False,blast=False,layer=None):
        slice=self.currentslice.get()
        slicesag = self.currentsagslice.get()
        slicecor = self.currentcorslice.get()
        self.ui.set_currentslice()
        if blast: # option for previewing enhancing in 2d
            self.ui.runblast(currentslice=slice)
        self.ax_img.set(data=self.ui.data[self.ui.dataselection][0,slice,:,:])
        self.ax2_img.set(data=self.ui.data[self.ui.dataselection][1,slice,:,:])
        self.ax3_img.set(data=self.ui.data[self.ui.dataselection][self.sagcordisplay.get(),:,slicecor,:])
        self.ax4_img.set(data=self.ui.data[self.ui.dataselection][self.sagcordisplay.get(),:,:,slicesag])
        # add current slice overlay
        self.update_labels()

        # self.vslicenumberlabel['text'] = '{}'.format(slice)
        if self.ui.dataselection in['seg_raw_fusion_d','seg_fusion_d']:
            self.ax_img.set(cmap='viridis')
            self.ax2_img.set(cmap='viridis')
            self.ax3_img.set(cmap='viridis')
            self.ax4_img.set(cmap='viridis')
        else:
            self.ax_img.set(cmap='gray')
            self.updatewl(ax=0)
            self.ax2_img.set(cmap='gray')
            self.updatewl(ax=1)
            self.ax3_img.set(cmap='gray')
            self.updatewl(ax=2)
            self.ax4_img.set(cmap='gray')
            self.updatewl(ax=3)
        if wl:   
            # possible latency problem here
            if self.ui.dataselection in ['seg_raw_fusion_d','seg_fusion_d']:
                # self.updatewl_fusion()
                self.ui.roiframe.layer_callback(updateslice=False,updatedata=False,layer=layer)
            elif self.ui.dataselection == 'raw':
                self.clipwl_raw()

        self.canvas.draw()
    
    def update_labels(self,item=None):
        for k in self.labels.keys():
            if self.labels[k] is not None:
                try:
                    Artist.remove(self.labels[k])
                except ValueError as e:
                    print(e)
        # convert data units to figure units
        self.labels['Im_A'] = self.axs['labelA'].text(self.xyfig['Im_A'][0],1+self.xyfig['Im_A'][1],'Im:'+str(self.currentslice.get()),color='w')
        self.labels['W_A'] = self.axs['labelA'].text(self.xyfig['W_A'][0],self.xyfig['W_A'][1],'W = '+'{:d}'.format(int(self.window[0]*255)),color='w')
        self.labels['L_A'] = self.axs['labelA'].text(self.xyfig['L_A'][0],self.xyfig['L_A'][1],'L = '+'{:d}'.format(int(self.level[0]*255)),color='w')
        self.labels['W_B'] = self.axs['labelB'].text(self.xyfig['W_B'][0],self.xyfig['W_B'][1],'W = '+'{:d}'.format(int(self.window[1]*255)),color='w')
        self.labels['L_B'] = self.axs['labelB'].text(self.xyfig['L_B'][0],self.xyfig['L_B'][1],'L = '+'{:d}'.format(int(self.level[1]*255)),color='w')
        self.labels['Im_C'] = self.axs['labelC'].text(self.xyfig['Im_C'][0],self.xyfig['Im_C'][1],'Im:'+str(self.currentcorslice.get()),color='w')
        self.labels['Im_D'] = self.axs['labelD'].text(self.xyfig['Im_D'][0],self.xyfig['Im_D'][1],'Im:'+str(self.currentsagslice.get()),color='w')

    # special update for previewing BLAST enhancing lesion in 2d
    def updateslice_blast(self,event=None):
        slice = self.currentslice.get()
        self.ui.set_currentslice()
        self.ui.runblast(currentslice=slice)
        # self.vslicenumberlabel['text'] = '{}'.format(slice)
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
        # self.vslicenumberlabel['text'] = '{}'.format(slice)
        self.canvas.draw()
        
    # TODO: latency problem for fusions. 
    # for now, don't allow to call this function if overlay is selected
    def updatewl(self,ax=0,lval=None,wval=None):

        self.wlflag = True
        if ax < 2:
            if lval:
                self.level[ax] += lval
            if wval:
                self.window[ax] += wval

            vmin = self.level[ax] - self.window[ax]/2
            vmax = self.level[ax] + self.window[ax]/2
        else: # currently hard-coded for T1
            vmin = self.level[0] - self.window[0]/2
            vmax = self.level[0] + self.window[0]/2

        if ax==0:
            self.ax_img.set_clim(vmin=vmin,vmax=vmax)
        elif ax==1:
            self.ax2_img.set_clim(vmin=vmin,vmax=vmax)
        elif ax==2:
            self.ax3_img.set_clim(vmin=vmin,vmax=vmax)
        elif ax==3:
            self.ax4_img.set_clim(vmin=vmin,vmax=vmax)
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

            
    def b1release(self,event):
        self.b1x = self.b1y = None

    # mouse drag for slice selection
    def b3motion_reset(self,event):
        self.b3y=None

    def b3motion(self,event):
        if event.y < 0 or event.y > self.ui.config.PanelSize*self.ui.config.dpi:
            return
        if event.x < 2*self.ui.config.PanelSize*self.ui.config.dpi:
            item = self.currentslice
            maxslice = self.dim[0]-1
        else:
            if event.y <= (self.ui.config.PanelSize*self.ui.config.dpi)/2:
                item = self.currentcorslice
            else:
                item = self.currentsagslice
            maxslice = self.dim[1]-1

        if self.b3y is None:
            self.b3y = event.y
        newslice = item.get() + (event.y-self.b3y)
        newslice = min(max(newslice,0),maxslice)
        item.set(newslice)
        self.updateslice()
        self.b3y = event.y
        self.update_labels()
        return

    # mouse drag event for window/level adjustment
    def b1motion(self,event):
        # print(event.num,event.state,event.type)
        # only allow adjustment in raw data view. overlays have latency to scale in 3d.
        if self.ui.dataselection != 'raw':
            return
        # no adjustment if nav bar is activated
        if 'zoom' in self.tbar.mode:
            return
        # no adjustment from outside the pane
        if event.y < 0 or event.y > self.config.PanelSize*self.config.dpi:
            return
        if event.x <=self.config.PanelSize*self.config.dpi:
            ax = [0,2,3]
        else:
            ax = [1]
        if self.b1x is None:
            self.b1x,self.b1y = event.x,event.y
            return
        if np.abs(event.x-self.b1x) > np.abs(event.y-self.b1y):
            if event.x-self.b1x > 0:
                for a in ax:
                    self.updatewl(ax=a,wval=.02)
            else:
                for a in ax:
                    self.updatewl(ax=a,wval=-.02)
        else:
            if event.y - self.b1y > 0:
                for a in ax:
                    self.updatewl(ax=a,lval=.02)
            else:
                for a in ax:
                    self.updatewl(ax=a,lval=-.02)

        self.b1x,self.b1y = event.x,event.y
        self.update_labels()


    # touchpad event for window/level adjustment. 
    # not updated lately
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
        print('normal stats')
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
        self.ui.data['blast']['params']['stdt1'] = np.std(X[kmeans.labels_==1,1])
        self.ui.data['blast']['params']['stdt2'] = np.std(X[kmeans.labels_==1,0])
        self.ui.data['blast']['params']['meant1'] = np.mean(X[kmeans.labels_==1,1])
        self.ui.data['blast']['params']['meant2'] = np.mean(X[kmeans.labels_==1,0])

        # activate thresholds only after normal slice stats are available
        self.ui.roiframe.bcslider['state']='normal'
        self.ui.roiframe.t2slider['state']='normal'
        self.ui.roiframe.t1slider['state']='normal'
        self.ui.roiframe.t1slider.bind("<ButtonRelease-1>",self.ui.roiframe.updatet1threshold)
        self.ui.roiframe.bcslider.bind("<ButtonRelease-1>",self.ui.roiframe.updatebcsize)
        self.ui.roiframe.t2slider.bind("<ButtonRelease-1>",self.ui.roiframe.updatet2threshold)

        # automatically run BLAST
        # self.ui.roiframe.updatet1threshold(currentslice=None)
        for layer in ['ET','T2 hyper']:
            self.ui.roiframe.layer_callback(layer=layer,updateslice=False,overlay=False)
            self.ui.runblast(currentslice=None,layer=layer)


################
# Case Selection
################

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
        # event currently a dummy arg since not being used in datadirentry_callback
        self.datadirentry.bind('<Return>',lambda event=None,load=load:self.datadirentry_callback(event=event,load=load))
        self.datadirentry.grid(column=3,row=0,columnspan=5)
        caselabel = ttk.Label(caseframe, text='Case: ')
        caselabel.grid(column=0,row=0,sticky='we')
        self.casename.trace_add('write',self.case_callback)
        self.w = ttk.Combobox(caseframe,width=6,textvariable=self.casename,values=self.caselist)
        self.w.grid(column=1,row=0)
        self.n4_check = ttk.Checkbutton(caseframe,text='N4',variable=self.n4_check_value)
        self.n4_check.grid(row=0,column=8,sticky='w')

        # initialize default directory. not a tkinter event so no event arg explicitly needed.
        self.datadirentry_callback(load=load)

    # callback for file dialog 
    def select_dir(self):
        self.fd.select_dir()
        self.datadir.set(self.fd.dir)
        self.datadirentry.update()
        self.datadirentry_callback()

    def case_callback(self,casevar=None,val=None,event=None):
        case = self.casename.get()
        self.ui.set_casename(val=case)
        print('Loading case {}'.format(case))
        self.loadCase()
        # if normal stats is 3d then seg runs automatically, can show 'seg_raw' directly
        if self.ui.sliceviewerframe.slicevolume_norm.get() == 0:
            self.ui.dataselection = 'raw'
        else:
            self.ui.roiframe.enhancingROI_overlay_value.set(True)
            # self.ui.roiframe.enhancingROI_overlay_callback()
        self.ui.sliceviewerframe.tbar.home()
        self.ui.updateslice()
        self.ui.starttime()

    def loadCase(self,case=None):

        # reset and reinitialize
        self.ui.resetUI()
        self.ui.roiframe.resetROI()

        if case is not None:
            self.casename.set(case)
            self.ui.set_casename()
        self.casedir = os.path.join(self.datadir.get(),self.config.UIdataroot+self.casename.get())
        files = os.listdir(self.casedir)
        # create t1mprage template
        t1ce_file = next((f for f in files if 't1ce' in f),None)
        t1ce = sitk.ReadImage(os.path.join(self.casedir,t1ce_file))
        # TODO: do coordinates properly for now just use flip for slice dimension
        img_arr = np.flip(sitk.GetArrayFromImage(t1ce),axis=0)
        # dimensions of panels might have to change depending on dimension of new data loaded.
        self.ui.sliceviewerframe.dim = np.shape(img_arr)
        # 2 channels hard-coded
        self.ui.data['raw'] = np.zeros((2,)+self.ui.sliceviewerframe.dim,dtype='float32')
        self.ui.data['raw'][0] = img_arr

        # Create t2flair template 
        t2flair_file = next((f for f in files if 'flair' in f),None)
        t2flair = sitk.ReadImage(os.path.join(self.casedir,t2flair_file))
        img_arr = np.flip(sitk.GetArrayFromImage(t2flair),axis=0)
        self.ui.data['raw'][1] = img_arr

        # bias correction. by convention, any pre-corrected files should have 'bias' in the filename
        if self.n4_check_value.get() and 'bias' not in t1ce_file:  
            self.n4()
        # rescale the data
        for ch in range(np.shape(self.ui.data['raw'])[0]):
            self.ui.data['raw'][ch] = self.rescale(self.ui.data['raw'][ch])

        # save copy of the raw data
        self.ui.data['raw_copy'] = copy.deepcopy(self.ui.data['raw'])

        # automatically run normal stats if volume selected
        if self.ui.sliceviewerframe.slicevolume_norm.get() == 1:
            self.ui.sliceviewerframe.normalslice_callback()

        # create the label
        label = sitk.ReadImage(os.path.join(self.casedir,self.config.UIdataroot+self.casename.get()+'_seg.nii'))
        img_arr = sitk.GetArrayFromImage(label)
        self.ui.data['label'] = img_arr

        # supplementary labels. brats and nnunet conventions are differnt.
        if False: # nnunet
            self.ui.data['manual_ET'] = (self.ui.data['label'] == 3).astype('int') #enhancing tumor 
            self.ui.data['manual_TC'] = (self.ui.data['label'] >= 2).astype('int') #tumour core
            self.ui.data['manual_WT'] = (self.ui.data['label'] >= 1).astype('int') #whole tumour
        else: # brats
            self.ui.data['manual_ET'] = (self.ui.data['label'] == 4).astype('int') #enhancing tumor 
            self.ui.data['manual_TC'] = ((self.ui.data['label'] == 1) | (self.ui.data['label'] == 4)).astype('int') #tumour core
            self.ui.data['manual_WT'] = (self.ui.data['label'] >= 1).astype('int') #whole tumour


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
            self.w.config(state='normal')            
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
                    # self.ui.set_casename()
            else:
                print('No cases found in directory {}'.format(dir))
                self.ui.set_message('No cases found in directory {}'.format(dir))
        else:
            print('Directory {} not found.'.format(dir))
            self.ui.set_message('Directory {} not found.'.format(dir))
            self.w.config(state='disable')
            self.datadirentry.update()
        return
