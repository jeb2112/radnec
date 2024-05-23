import os,sys
import numpy as np
import glob
import copy
import re
import logging
import subprocess
import tkinter as tk
import nibabel as nb
from nibabel.processing import resample_from_to
import pydicom as pd
from pydicom.fileset import FileSet
from tkinter import ttk,StringVar,DoubleVar,PhotoImage
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.path import Path
from cProfile import Profile
from pstats import SortKey,Stats
from enum import Enum

matplotlib.use('TkAgg')
import SimpleITK as sitk
import itk
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice
import scipy

from src.NavigationBar import NavigationBar

from src.CreateSVFrame import *

#####################################
# Slice Viewer for Blast segmentation
#####################################

class CreateBlastSVFrame(CreateSliceViewerFrame):
    def __init__(self,parentframe,ui=None,padding='10',style=None):
        super().__init__(parentframe,ui=ui,padding=padding,style=style)

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
        self.lines = {'A':{'h':None,'v':None},'B':{'h':None,'v':None},'C':{'h':None,'v':None},'D':{'h':None,'v':None}}
        self.sagcordisplay = tk.IntVar(value=0)
        self.overlaytype = tk.IntVar(value=self.config.OverlayType)
        self.slicevolume_norm = tk.IntVar(value=1)
        # window/level values for T1,T2
        self.window = np.array([1.,1.],dtype='float')
        self.level = np.array([0.5,0.5],dtype='float')
        # window/level values for overlays and images. hard-coded for now.
        self.wl = {'t1+':[600,300],'flair+':[600,300]}
        self.wlflag = False
        self.b1x = self.b1y = None # for tracking window/level mouse drags
        self.b3y = None # mouse drag for cor,sag slices\
        self.sliceinc = 0
        self.prevtime = 0
        # image dimensions
        self.dim = self.ui.config.ImageDim
        self.canvas = None
        self.cw = None
        self.blankcanvas = None
        self.fig = None
        self.resizer_count = 1
        self.ui = ui

        self.frame.grid(row=1, column=0, columnspan=6, in_=self.parentframe,sticky='NSEW')
        self.fstyle.configure('sliceviewerframe.TFrame',background='#000000')
        self.frame.configure(style='sliceviewerframe.TFrame')
        self.create_blank_canvas()

        # dummy frame to hold canvas and slider bars
        self.fstyle.configure('canvasframe.TFrame',background='#000000')
        self.canvasframe = ttk.Frame(self.frame)
        self.canvasframe.configure(style='canvasframe.TFrame')
        self.canvasframe.grid(row=1,column=0,columnspan=3,sticky='NW')

        # normal slice button
        self.normal_frame = ttk.Frame(self.parentframe,padding='0')
        self.normal_frame.grid(row=3,column=0,sticky='NW')
        normalSlice = ttk.Button(self.normal_frame,text='normal',command=self.normalslice_callback)
        normalSlice.grid(column=0,row=0,sticky='w')
        slicevolume_slice_button = ttk.Radiobutton(self.normal_frame,text='slice',variable=self.slicevolume_norm,value=0)
        slicevolume_slice_button.grid(row=0,column=1,sticky='w')
        slicevolume_volume_button = ttk.Radiobutton(self.normal_frame,text='vol.',variable=self.slicevolume_norm,value=1)
        slicevolume_volume_button.grid(row=0,column=2,sticky='w')

        # t1/t2 selection for sag/cor panes
        sagcordisplay_label = ttk.Label(self.normal_frame, text='Sag/Cor: ')
        sagcordisplay_label.grid(row=0,column=4,padx=(50,0),sticky='e')
        self.sagcordisplay_button = ttk.Radiobutton(self.normal_frame,text='T1',variable=self.sagcordisplay,value=0,
                                                    command=self.updateslice)
        self.sagcordisplay_button.grid(column=5,row=0,sticky='w')
        self.sagcordisplay_button = ttk.Radiobutton(self.normal_frame,text='T2',variable=self.sagcordisplay,value=1,
                                                    command=self.updateslice)
        self.sagcordisplay_button.grid(column=6,row=0,sticky='w')

        # overlay type contour mask
        overlaytype_label = ttk.Label(self.normal_frame, text='overlay type: ')
        overlaytype_label.grid(row=1,column=4,padx=(50,0),sticky='e')
        self.overlaytype_button = ttk.Radiobutton(self.normal_frame,text='C',variable=self.overlaytype,value=0,
                                                    command=Command(self.updateslice,wl=True))
        self.overlaytype_button.grid(row=1,column=5,sticky='w')
        self.overlaytype_button = ttk.Radiobutton(self.normal_frame,text='M',variable=self.overlaytype,value=1,
                                                    command=Command(self.updateslice,wl=True))
        self.overlaytype_button.grid(row=1,column=6,sticky='w')

        # messages text frame
        self.messagelabel = ttk.Label(self.normal_frame,text=self.ui.message.get(),padding='5',borderwidth=0)
        self.messagelabel.grid(row=2,column=0,columnspan=3,sticky='ew')

        if self.ui.OS in ('win32','darwin'):
            self.ui.root.bind('<MouseWheel>',self.mousewheel_win32)

        if self.ui.OS == 'linux':
            self.ui.root.bind('<Button-4>',self.mousewheel)
            self.ui.root.bind('<Button-5>',self.mousewheel)

    def resizer(self,event):
        if self.cw is None:
            return
        self.resizer_count *= -1
        # quick hack to improve the latency skip every other Configure event
        if self.resizer_count > 0:
            return
        # print(event)
        slicefovratio = self.dim[0]/self.dim[1]
        # self.hi = (event.height-self.ui.caseframe.frame.winfo_height()-self.normal_frame_minsize)/self.ui.dpi
        self.hi = (event.height-self.ui.caseframe.frame.winfo_height()-self.ui.roiframe.frame.winfo_height())/self.ui.dpi
        self.wi = self.hi*2 + self.hi / (2*slicefovratio)
        if self.wi > event.width/self.ui.dpi:
            self.wi = (event.width-2*int(self.ui.mainframe_padding))/self.ui.dpi
            self.hi = self.wi/(2+1/(2*slicefovratio))
        self.ui.current_panelsize = self.hi
        # print('{:d},{:d},{:.2f},{:.2f},{:.2f}'.format(event.width,event.height,self.wi,self.hi,self.wi/self.hi))
        # self.cw.grid_propagate(0)
        self.cw.configure(width=int(self.wi*self.fig.dpi),height=int(self.hi*self.fig.dpi))
        self.fig.set_size_inches((self.wi,self.hi),forward=True)
        return

    # place holder until a dataset is loaded
    # could create a dummy figure with toolbar, but the background colour during resizing was inconsistent at times
    # with just frame background style resizing behaviour seems correct.
    def create_blank_canvas(self):
        slicefovratio = self.config.ImageDim[0]/self.config.ImageDim[1]
        w = self.ui.current_panelsize*(2 + 1/(2*slicefovratio)) * self.ui.dpi
        h = self.ui.current_panelsize * self.ui.dpi
        if True:
            fig = plt.figure(figsize=(w/self.ui.dpi,h/self.ui.dpi),dpi=self.ui.dpi)
            axs = fig.add_subplot(111)
            axs.axis('off')
            fig.tight_layout(pad=0)
            # fig.patch.set_facecolor('white')
            self.blankcanvas = FigureCanvasTkAgg(fig, master=self.frame)  
            self.blankcanvas.get_tk_widget().grid(row=1, column=0, columnspan=3)
            tbar = NavigationToolbar2Tk(self.blankcanvas,self.parentframe,pack_toolbar=False)
            tbar.children['!button4'].pack_forget() # get rid of configure plot
            tbar.grid(column=0,row=2,columnspan=3,sticky='NW')
        self.frame.configure(width=w,height=h)
     
    # main canvas created when data are loaded
    def create_canvas(self,figsize=None):
        slicefovratio = self.dim[0]/self.dim[1]
        if figsize is None:
            figsize = (self.ui.current_panelsize*(2 + 1/(2*slicefovratio)),self.ui.current_panelsize)
        if self.fig is not None:
            plt.close(self.fig)

        self.fig,self.axs = plt.subplot_mosaic([['A','B','C'],['A','B','D']],
                                     width_ratios=[self.ui.current_panelsize,self.ui.current_panelsize,
                                                   self.ui.current_panelsize / (2 * slicefovratio) ],
                                     figsize=figsize,dpi=self.ui.dpi)
        self.ax_img = self.axs['A'].imshow(np.zeros((self.dim[1],self.dim[2])),vmin=0,vmax=1,cmap='gray',origin='lower',aspect=1)
        self.ax2_img = self.axs['B'].imshow(np.zeros((self.dim[1],self.dim[2])),vmin=0,vmax=1,cmap='gray',origin='lower',aspect=1)
        self.ax3_img = self.axs['C'].imshow(np.zeros((self.dim[0],self.dim[1])),vmin=0,vmax=1,cmap='gray',origin='lower',aspect=1)
        self.ax4_img = self.axs['D'].imshow(np.zeros((self.dim[0],self.dim[1])),vmin=0,vmax=1,cmap='gray',origin='lower',aspect=1)
        self.ax_img.format_cursor_data = self.make_cursordata_format()
        self.ax2_img.format_cursor_data = self.make_cursordata_format()
        self.ax3_img.format_cursor_data = self.make_cursordata_format()
        self.ax4_img.format_cursor_data = self.make_cursordata_format()

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
        self.axs['labelA'] = self.fig.add_subplot(2,3,4)

        self.axs['labelB'] = self.fig.add_subplot(2,3,5)
        self.axs['labelC'] = self.fig.add_subplot(2,3,3)
        self.axs['labelD'] = self.fig.add_subplot(2,3,6)
        for a in ['A','B','C','D']:
            # set axes zorder so label axis is on the bottom
            # self.axs[a].set_zorder(1)
            # prevent labels from panning or zooming
            self.axs['label'+a].set_navigate(False)
            # read mouse coords from underlying image axes on mouse over
            self.axs['label'+a].format_coord = self.make_coord_format(self.axs['label'+a],self.axs[a])

        for a in self.axs.keys():
            self.axs[a].axis('off')
        # set up axis sharing
        self.axs['B']._shared_axes['x'].join(self.axs['B'],self.axs['A'])
        self.axs['B']._shared_axes['y'].join(self.axs['B'],self.axs['A'])
        self.fig.tight_layout(pad=0)
        self.fig.patch.set_facecolor('k')
        # record the data to figure coords of each label for each axis
        self.xyfig={}
        figtrans={}
        for a in ['A','B','C','D']:
            figtrans[a] = self.axs[a].transData + self.axs[a].transAxes.inverted()
        # these label coords are slightly hard-coded
        self.xyfig['Im_A']= figtrans['A'].transform((5,self.dim[1]-20))
        self.xyfig['W_A'] = figtrans['A'].transform((int(self.dim[1]/2),5))
        self.xyfig['L_A'] = figtrans['A'].transform((int(self.dim[1]*3/4),5))
        self.xyfig['W_B'] = figtrans['B'].transform((int(self.dim[1]/2),5))
        self.xyfig['L_B'] = figtrans['B'].transform((int(self.dim[1]*3/4),5))
        # self.xyfig['W_A'] = figtrans['A'].transform((int(self.dim[1]/2),self.dim[1]-10))
        # self.xyfig['L_A'] = figtrans['A'].transform((int(self.dim[1]*3/4),self.dim[1]-10))
        # self.xyfig['W_B'] = figtrans['B'].transform((int(self.dim[1]/2),self.dim[1]-10))
        # self.xyfig['L_B'] = figtrans['B'].transform((int(self.dim[1]*3/4),self.dim[1]-10))
        self.xyfig['Im_C'] = figtrans['C'].transform((5,self.dim[0]-15))
        self.xyfig['Im_D'] = figtrans['D'].transform((5,self.dim[0]-15))
        self.figtrans = figtrans

        # figure canvas
        newcanvas = FigureCanvasTkAgg(self.fig, master=self.canvasframe)  
        newcanvas.get_tk_widget().configure(bg='black')
        newcanvas.get_tk_widget().configure(width=figsize[0]*self.ui.dpi,height=figsize[1]*self.ui.dpi)
        newcanvas.get_tk_widget().grid(row=0, column=0, sticky='')

        self.tbar = NavigationBar(newcanvas,self.parentframe,pack_toolbar=False,ui=self.ui,axs=self.axs)
        self.tbar.grid(column=0,row=2,columnspan=3,sticky='NW')

        if self.canvas is not None:
            self.cw.delete('all')
        self.canvas = newcanvas

        # slider bars
        if True:
            self.axsliceslider = ttk.Scale(self.canvasframe,from_=0,to=self.dim[0]-1,variable=self.currentslice,
                                        orient=tk.VERTICAL, length='3i',command=self.updateslice)
            self.axsliceslider.grid(column=0,row=0,sticky='w')
            self.sagsliceslider = ttk.Scale(self.canvasframe,from_=0,to=self.dim[1]-1,variable=self.currentsagslice,
                                        orient=tk.VERTICAL, length='1.5i',command=self.updateslice)
            self.sagsliceslider.grid(column=0,row=0,sticky='ne')
            self.corsliceslider = ttk.Scale(self.canvasframe,from_=0,to=self.dim[1]-1,variable=self.currentcorslice,
                                        orient=tk.VERTICAL, length='1.5i',command=self.updateslice)
            self.corsliceslider.grid(column=0,row=0,sticky='se')

        # various bindings
        if self.ui.OS == 'linux':
            self.canvas.get_tk_widget().bind('<<MyMouseWheel>>',EventCallback(self.mousewheel,key='Key'))
        self.canvas.get_tk_widget().bind('<Up>',self.keyboard_slice)
        self.canvas.get_tk_widget().bind('<Down>',self.keyboard_slice)
        self.canvas.get_tk_widget().bind('<Enter>',self.focus)
        self.cw = self.canvas.get_tk_widget()

        self.frame.update()

    # TODO: different bindings and callbacks need some organization
    def updateslice(self,event=None,wl=False,blast=False,layer=None):
        slice=self.currentslice.get()
        slicesag = self.currentsagslice.get()
        slicecor = self.currentcorslice.get()
        self.ui.set_currentslice()
        if blast: # option for previewing enhancing in 2d
            self.ui.runblast(currentslice=slice)
        if self.ui.roiframe.layer.get() == 'ET  ':
            self.ax_img.set(data=self.ui.data[0].dset[self.ui.dataselection]['d'][slice,:,:])
        else:
            self.ax_img.set(data=self.ui.data[0].dset[self.ui.dataselection]['d'][slice,:,:])
        self.ax2_img.set(data=self.ui.data[0].dset['flair+']['d'][slice,:,:])
        # self.ax3_img.set(data=self.ui.data[0].dset[self.ui.base][self.sagcordisplay.get(),:,:,slicesag])
        # self.ax4_img.set(data=self.ui.data[0].dset[self.ui.base][self.sagcordisplay.get(),:,slicecor,:])
        self.ax3_img.set(data=self.ui.data[0].dset[self.ui.base]['d'][:,:,slicesag])
        self.ax4_img.set(data=self.ui.data[0].dset[self.ui.base]['d'][:,slicecor,:])
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
            if self.ui.dataselection == 'seg_raw_fusion_d':
                # self.ui.roiframe.layer_callback(updateslice=False,updatedata=False,layer=layer)
                self.ui.roiframe.layer_callback(layer=layer)
            elif self.ui.dataselection == 'seg_fusion_d':
                # self.ui.roiframe.layerROI_callback(updateslice=False,updatedata=False,layer=layer)
                self.ui.roiframe.layerROI_callback(layer=layer)
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
        self.labels['W_A'] = self.axs['labelA'].text(self.xyfig['W_A'][0],self.xyfig['W_A'][1],'W = '+'{:d}'.format(int(self.window[0])),color='w')
        self.labels['L_A'] = self.axs['labelA'].text(self.xyfig['L_A'][0],self.xyfig['L_A'][1],'L = '+'{:d}'.format(int(self.level[0])),color='w')
        self.labels['W_B'] = self.axs['labelB'].text(self.xyfig['W_B'][0],self.xyfig['W_B'][1],'W = '+'{:d}'.format(int(self.window[1])),color='w')
        self.labels['L_B'] = self.axs['labelB'].text(self.xyfig['L_B'][0],self.xyfig['L_B'][1],'L = '+'{:d}'.format(int(self.level[1]*1)),color='w')
        self.labels['Im_C'] = self.axs['labelC'].text(self.xyfig['Im_C'][0],self.xyfig['Im_C'][1],'Im:'+str(self.currentsagslice.get()),color='w')
        self.labels['Im_D'] = self.axs['labelD'].text(self.xyfig['Im_D'][0],self.xyfig['Im_D'][1],'Im:'+str(self.currentcorslice.get()),color='w')

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
        # only process on main panels
        if ax < 2:
            if lval:
                self.level[ax] += lval
            if wval:
                self.window[ax] += wval

            vmin = self.level[ax] - self.window[ax]/2
            vmax = self.level[ax] + self.window[ax]/2

        vmin0 = self.level[0] - self.window[0]/2
        vmax0 = self.level[0] + self.window[0]/2
        vmin1 = self.level[1] - self.window[1]/2
        vmax1 = self.level[1] + self.window[1]/2

        if ax==0:
            self.ax_img.set_clim(vmin=vmin,vmax=vmax)
        elif ax==1:
            self.ax2_img.set_clim(vmin=vmin,vmax=vmax)
        if self.sagcordisplay.get() == 0:
            self.ax3_img.set_clim(vmin=vmin0,vmax=vmax0)
            self.ax4_img.set_clim(vmin=vmin0,vmax=vmax0)
        elif self.sagcordisplay.get() == 1:
            self.ax3_img.set_clim(vmin=vmin1,vmax=vmax1)
            self.ax4_img.set_clim(vmin=vmin1,vmax=vmax1)

        self.canvas.draw()

    # color window/level scaling needs to be done separately for latency
    # for now, just tack it onto the fusion toggle button
    def updatewl_fusion(self):
        if self.ui.dataselection in ['seg_raw_fusion_d','seg_fusion_d']:
            # for ax in range(2):
            for ax in ['t1+','flair+']:
                # vmin = self.level[ax] - self.window[ax]/2
                # vmax = self.level[ax] + self.window[ax]/2
                vmin = self.wl[ax][1] - self.wl[ax][0]/2
                vmax = self.wl[ax][1] + self.wl[ax][0]/2
                if False:
                    self.ui.data[0].dset[ax]['d'] = self.ui.caseframe.rescale(self.ui.data[0].dset[ax+'_copy']['d'],vmin=vmin,vmax=vmax)

    # clip the raw data to window and level settings
    def clipwl_raw(self):
        # for ax in range(2):
        for ax in ['t1+','flair+']:
            vmin = self.wl[ax][1] - self.wl[ax][0]/2
            vmax = self.wl[ax][1] + self.wl[ax][0]/2
            self.ui.data[0].dset[ax]['d'] = self.ui.caseframe.rescale(self.ui.data[0].dset[ax]['d'],vmin=vmin,vmax=vmax)

    def restorewl_raw(self,dt):
        if False:
            self.ui.data[0].dset[dt]['d'] = copy.deepcopy(self.ui.data[0].dset[dt+'_copy']['d'])


    def fitlin(self,x,a,b):
        return a*x + b

    def normalslice_callback(self,event=None):
        print('normal stats')
        # do kmeans
        # Creates a matrix of voxels for normal brain slice
        # Gating Routine

        if self.slicevolume_norm.get() == 0:
            self.normalslice=self.ui.get_currentslice()
            region_of_support = np.where(self.ui.data[0].dset['t1+']['d'][self.normalslice]*self.ui.data[0].dset['flair+']['d'][self.normalslice]>0) 
            vset = np.zeros_like(region_of_support,dtype='float')
            # for i in range(3):
            for i,ax in enumerate(['t1+','flair_+']):
                vset[i] = np.ravel(self.ui.data[0].dset[ax]['d'][self.normalslice][region_of_support])
        else:
            self.normalslice = None
            region_of_support = np.where(self.ui.data[0].dset['t1+']['d']*self.ui.data[0].dset['flair+']['d'] >0)
            vset = np.zeros_like(region_of_support,dtype='float')
            # for i in range(3):
            for i,ax in enumerate(['zt1+','zflair+']):
                vset[i] = np.ravel(self.ui.data[0].dset[ax]['d'][region_of_support])
            # t1channel_normal = self.ui.data['raw'][0][region_of_support]
            # flairchannel_normal = self.ui.data['raw'][1][region_of_support]
            # t2channel_normal = self.ui.data['raw'][2][region_of_support]

        # kmeans to calculate statistics for brain voxels
        # X_et = np.column_stack((flair,t1))
        # X_net = np.column_stack((flair,t2))
        X={}
        X['ET'] = np.column_stack((vset[1],vset[0]))
        # T2 hyper values will just be the same as ET since we do not have plain T2 available for rad nec.
        X['T2 hyper'] = np.column_stack((vset[1],vset[0]))

        for i,layer in enumerate(['ET','T2 hyper']):
            np.random.seed(1)
            kmeans = KMeans(n_clusters=2,n_init='auto').fit(X[layer])
            background_cluster = np.argmin(np.power(kmeans.cluster_centers_[:,0],2)+np.power(kmeans.cluster_centers_[:,1],2))

            # Calculate stats for brain cluster. currently hard-coded to study #0
            self.ui.blastdata['blast']['params'][layer]['stdt12'] = np.std(X[layer][kmeans.labels_==background_cluster,1])
            self.ui.blastdata['blast']['params'][layer]['stdflair'] = np.std(X[layer][kmeans.labels_==background_cluster,0])
            self.ui.blastdata['blast']['params'][layer]['meant12'] = np.mean(X[layer][kmeans.labels_==background_cluster,1])
            self.ui.blastdata['blast']['params'][layer]['meanflair'] = np.mean(X[layer][kmeans.labels_==background_cluster,0])

            if False:
                plt.figure(7)
                ax = plt.subplot(1,2,i+1)
                plt.scatter(X[layer][kmeans.labels_==1-background_cluster,0],X[layer][kmeans.labels_==1-background_cluster,1],c='b',s=1)
                plt.scatter(X[layer][kmeans.labels_==background_cluster,0],X[layer][kmeans.labels_==background_cluster,1],c='r',s=1)
                ax.set_aspect('equal')
                ax.set_xlim(left=0,right=1.0)
                ax.set_ylim(bottom=0,top=1.0)
                plt.text(0,1.02,'{:.3f},{:.3f}'.format(self.ui.data['blast']['params'][layer]['meanflair'],self.ui.data['blast']['params'][layer]['stdflair']))

                plt.savefig('/home/jbishop/Pictures/scatterplot_normal.png')
                plt.clf()
                # plt.show(block=False)

        # automatically run BLAST
            self.ui.roiframe.layer_callback(layer=layer,updateslice=False,overlay=False)
            self.ui.runblast(currentslice=None,layer=layer)

            # activate thresholds only after normal slice stats are available
            for s in ['t12','flair','bc']:
                self.ui.roiframe.sliders[layer][s]['state']='normal'
                self.ui.roiframe.sliders[layer][s].bind("<ButtonRelease-1>",Command(self.ui.roiframe.updateslider,layer,s))
        # since we finish the on the T2 hyper layer, have this slider disabled to begin with
        # self.ui.roiframe.sliders['ET']['t12']['state']='disabled'


