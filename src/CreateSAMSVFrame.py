import os,sys
import numpy as np
import glob
import copy
import re
import logging
import time
import tkinter as tk
import nibabel as nb
from nibabel.processing import resample_from_to
import pydicom as pd
from pydicom.fileset import FileSet
from tkinter import ttk,StringVar,DoubleVar,PhotoImage
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
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice
import scipy

from src.NavigationBar import NavigationBar

from src.CreateSVFrame import *

#####################################
# Slice Viewer for SAM segmentation
#####################################

class CreateSAMSVFrame(CreateSliceViewerFrame):
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
        self.overlay_type = tk.IntVar(value=self.config.BlastOverlayType)
        self.slicevolume_norm = tk.IntVar(value=1)
        # window/level stuff will need further tidying up
        # window/level values for T1,T2
        self.window = np.array([1.,1.],dtype='float')
        self.level = np.array([0.5,0.5],dtype='float')
        # window/level values for overlays and images. hard-coded for now.
        self.wl = {'t1+':[600,300],'flair':[600,300]}
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
        # user interaction time
        self.dwelltime = None
        self.timingtext = tk.StringVar(value='off')
        self.timing = tk.IntVar(value=0)
        self.elapsedtime = 0
        # bbox tool
        self.bbox = {'ax':None,'p0':None,'p1':None,'plot':None,'l':None,'ch':None}        

        self.frame.grid(row=1, column=0, columnspan=6, in_=self.parentframe,sticky='NSEW')
        self.fstyle.configure('sliceviewerframe.TFrame',background='#000000')
        self.frame.configure(style='sliceviewerframe.TFrame')

        # normal frame for stats button etc
        self.normal_frame = ttk.Frame(self.parentframe,padding='0')
        self.normal_frame.grid(row=3,column=0,sticky='NW')

        self.create_blank_canvas()

        # dummy frame to hold canvas and slider bars
        self.fstyle.configure('canvasframe.TFrame',background='#000000')
        self.canvasframe = ttk.Frame(self.frame)
        self.canvasframe.configure(style='canvasframe.TFrame')
        self.canvasframe.grid(row=1,column=0,columnspan=3,sticky='NW')

        # timer button
        self.timerbutton = ttk.Checkbutton(self.normal_frame,style='Toolbutton',textvariable=self.timingtext,variable=self.timing,command=self.timer)
        self.timerbutton.configure(style='frame.TCheckbutton')
        self.timerbutton.grid(row=1,column=1,sticky='w')
        self.timerbuttonlabel = ttk.Label(self.normal_frame,text=' timer:',padding='5')
        self.timerbuttonlabel.grid(row=1,column=0,sticky='e')

        # button to run stats for normal background
        normalSlice = ttk.Button(self.normal_frame,text='normal stats',command=self.normalslice_callback)
        normalSlice.grid(row=1,column=2,sticky='w')
        # button to run 2d SAM on current prompt (point or bbox)
        self.run2dSAM = ttk.Button(self.normal_frame,text='run SAM',command=self.sam2d_callback,state='disabled')
        self.run2dSAM.grid(row=1,column=3,sticky='w')

        # messages text frame
        self.messagelabel = ttk.Label(self.normal_frame,text=self.ui.message.get(),padding='5',borderwidth=0)
        self.messagelabel.grid(row=2,column=0,columnspan=3,sticky='ew')

        if self.ui.OS in ('win32','darwin'):
            self.ui.root.bind('<MouseWheel>',self.mousewheel_win32)

        if self.ui.OS == 'linux':
            self.ui.root.bind('<Button-4>',self.mousewheel)
            self.ui.root.bind('<Button-5>',self.mousewheel)

    # for dragging resizing of GUI window
    def resizer(self,event):
        if self.cw is None:
            return
        self.resizer_count *= -1
        # quick hack to improve the latency skip every other Configure event
        if self.resizer_count > 0:
            return
        slicefovratio = self.dim[0]/self.dim[1]
        min_height = max(self.ui.caseframe.frame.winfo_height(),self.ui.functionmenu.winfo_height()) + \
            max(self.normal_frame.winfo_height(),self.ui.roiframe.frame.winfo_height()) + \
            2*int(self.ui.mainframe_padding)
        min_width = max(self.ui.roiframe.frame.winfo_width() + self.normal_frame.winfo_width(), \
                        self.ui.caseframe.frame.winfo_width()+self.ui.functionmenu.winfo_width())
        
        self.hi = (event.height-min_height)/self.ui.dpi
        self.wi = self.hi*1 + self.hi / (2*slicefovratio)
        self.wi = max(self.wi,min_width)
        if self.wi > event.width/self.ui.dpi:
            self.wi = (event.width-2*int(self.ui.mainframe_padding))/self.ui.dpi
            self.hi = self.wi/(1+1/(2*slicefovratio))
        self.ui.current_panelsize = self.hi
        # print('{:d},{:d},{:.2f},{:.2f},{:.2f}'.format(event.width,event.height,self.wi,self.hi,self.wi/self.hi))
        # self.cw.grid_propagate(0)
        self.cw.configure(width=int(self.wi*self.fig.dpi),height=int(self.hi*self.fig.dpi))
        self.fig.set_size_inches((self.wi,self.hi),forward=True)
        return

    # resize root window to match current sliceviewer
    def resize(self):
        if self.canvas:
            w = self.canvas.get_tk_widget().winfo_width()
            h = self.canvas.get_tk_widget().winfo_height()
        else:
            w = self.blankcanvas.get_tk_widget().winfo_width()
            h = self.blankcanvas.get_tk_widget().winfo_height()

        min_height = max(self.ui.caseframe.frame.winfo_height(),self.ui.functionmenu.winfo_height()) + \
            max(self.normal_frame.winfo_height(),self.ui.roiframe.frame.winfo_height()) + \
            2*int(self.ui.mainframe_padding)
        min_width = max(self.ui.roiframe.frame.winfo_width() + self.normal_frame.winfo_width(), \
                        self.ui.caseframe.frame.winfo_width()+self.ui.functionmenu.winfo_width())
        w = max(w,min_width)
        h += min_height
        print('resize {},{}'.format(w,h))
        self.ui.root.geometry(f'{w}x{h}')
        return


    # place holder until a dataset is loaded
    # could create a dummy figure with toolbar, but the background colour during resizing was inconsistent at times
    # with just frame background style resizing behaviour seems correct.
    def create_blank_canvas(self):
        slicefovratio = self.config.ImageDim[0]/self.config.ImageDim[1]
        w = self.ui.current_panelsize*(1 + 1/(2*slicefovratio)) * self.ui.dpi
        h = self.ui.current_panelsize * self.ui.dpi
        if True:
            fig = plt.figure(figsize=(w/self.ui.dpi,h/self.ui.dpi),dpi=self.ui.dpi)
            axs = fig.add_subplot(111)
            axs.axis('off')
            fig.tight_layout(pad=0)
            # fig.patch.set_facecolor('white')
            self.blankcanvas = FigureCanvasTkAgg(fig, master=self.frame)  
            self.blankcanvas.get_tk_widget().grid(row=1, column=0, columnspan=3)
            self.tbar = NavigationToolbar2Tk(self.blankcanvas,self.normal_frame,pack_toolbar=False)
            self.tbar.children['!button4'].pack_forget() # get rid of configure plot
            self.tbar.grid(column=0,row=0,columnspan=4,sticky='NW')
        self.frame.configure(width=w,height=h)
     
    # main canvas created when data are loaded
    def create_canvas(self,figsize=None):
        slicefovratio = self.dim[0]/self.dim[1]
        if figsize is None:
            figsize = (self.ui.current_panelsize*(1 + 1/(2*slicefovratio)),self.ui.current_panelsize)
        if self.fig is not None:
            plt.close(self.fig)

        self.fig,self.axs = plt.subplot_mosaic([['A','C'],['A','D']],
                                     width_ratios=[self.ui.current_panelsize,
                                                   self.ui.current_panelsize / (2 * slicefovratio) ],
                                     figsize=figsize,dpi=self.ui.dpi)
        self.ax_img = self.axs['A'].imshow(np.zeros((self.dim[1],self.dim[2])),vmin=0,vmax=1,cmap='gray',origin='upper',aspect=1)
        self.ax3_img = self.axs['C'].imshow(np.zeros((self.dim[0],self.dim[1])),vmin=0,vmax=1,cmap='gray',origin='lower',aspect=1)
        self.ax4_img = self.axs['D'].imshow(np.zeros((self.dim[0],self.dim[1])),vmin=0,vmax=1,cmap='gray',origin='lower',aspect=1)
        self.ax_img.format_cursor_data = self.make_cursordata_format()
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
        self.axs['labelA'] = self.fig.add_subplot(2,2,3)
        self.axs['labelC'] = self.fig.add_subplot(2,2,2)
        self.axs['labelD'] = self.fig.add_subplot(2,2,4)
        for a in ['A','C','D']:
            # set axes zorder so label axis is on the bottom
            # self.axs[a].set_zorder(1)
            # prevent labels from panning or zooming
            self.axs['label'+a].set_navigate(False)
            # read mouse coords from underlying image axes on mouse over
            self.axs['label'+a].format_coord = self.make_coord_format(self.axs['label'+a],self.axs[a])

        for a in self.axs.keys():
            self.axs[a].axis('off')
        self.fig.tight_layout(pad=0)
        self.fig.patch.set_facecolor('k')
        # record the data to figure coords of each label for each axis
        self.xyfig={}
        figtrans={}
        for a in ['A','C','D']:
            figtrans[a] = self.axs[a].transData + self.axs[a].transAxes.inverted()
        # these label coords are slightly hard-coded
        self.xyfig['Im_A']= figtrans['A'].transform((5,25))
        self.xyfig['W_A'] = figtrans['A'].transform((int(self.dim[1]/2),self.dim[1]-15))
        self.xyfig['L_A'] = figtrans['A'].transform((int(self.dim[1]*3/4),self.dim[1]-15))
        self.xyfig['Im_C'] = figtrans['C'].transform((5,self.dim[0]-15))
        self.xyfig['Im_D'] = figtrans['D'].transform((5,self.dim[0]-15))
        self.figtrans = figtrans

        # figure canvas
        newcanvas = FigureCanvasTkAgg(self.fig, master=self.canvasframe)  
        newcanvas.get_tk_widget().configure(bg='black')
        newcanvas.get_tk_widget().configure(width=figsize[0]*self.ui.dpi,height=figsize[1]*self.ui.dpi)
        newcanvas.get_tk_widget().grid(row=0, column=0, sticky='')

        self.tbar = NavigationBar(newcanvas,self.normal_frame,pack_toolbar=False,ui=self.ui,axs=self.axs)
        self.tbar.grid(column=0,row=0,columnspan=4,sticky='NW')

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

    # main callback for any changes in the display
    def updateslice(self,event=None,wl=False,blast=False,layer=None):
        s = self.ui.s # local reference
        slice=self.currentslice.get()
        slicesag = self.currentsagslice.get()
        slicecor = self.currentcorslice.get()
        self.ui.set_currentslice()
        if blast: # option for previewing enhancing in 2d. probably not used anymore
            self.ui.runblast(currentslice=slice)
        # which data array to display. 'd' is general
        d = 'd'
        # special case arrangement for displaying BLAST overlays by layer type. slightly awkward.
        if self.ui.roiframe.overlay_value['BLAST'].get():
            d = 'd'+self.ui.roiframe.layer.get()
        self.ax_img.set(data=self.ui.data[s].dset[self.ui.dataselection][self.ui.chselection][d][slice,:,:])
        # by convention, 2nd panel will always be flair, 1st panel could be t1,t1+ or t2
        # for sam currently there will be no flair 2nd panel
        self.ax3_img.set(data=self.ui.data[s].dset[self.ui.dataselection][self.ui.chselection][d][:,:,slicesag])
        self.ax4_img.set(data=self.ui.data[s].dset[self.ui.dataselection][self.ui.chselection][d][:,slicecor,:])
        # add current slice overlay
        self.update_labels()

        if self.ui.dataselection in['seg_raw_fusion','seg_fusion']:
            self.ax_img.set(cmap='viridis')
            self.ax3_img.set(cmap='viridis')
            self.ax4_img.set(cmap='viridis')
        else:
            self.ax_img.set(cmap='gray')
            self.updatewl(ax=0)
            self.ax3_img.set(cmap='gray')
            self.updatewl(ax=2)
            self.ax4_img.set(cmap='gray')
            self.updatewl(ax=3)
        if wl:   
            # possible latency problem here
            if self.ui.dataselection == 'seg_raw_fusion':
                # self.ui.roiframe.layer_callback(updateslice=False,updatedata=False,layer=layer)
                self.ui.roiframe.layer_callback(layer=layer)
            elif self.ui.dataselection == 'seg_fusion':
                # self.ui.roiframe.layerROI_callback(updateslice=False,updatedata=False,layer=layer)
                self.ui.roiframe.layerROI_callback(layer=layer)
            elif self.ui.dataselection == 'raw':
                self.clipwl_raw()

        # if not an event, show an existing bbox, or remove it
        if self.ui.currentroi > 0 and event is None:
            if self.ui.roiframe.overlay_value['SAM'].get() == True or str(self.run2dSAM['state']) == 'active':
                self.update_bboxs()
            else:
                self.clear_bbox()

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
        self.labels['Im_C'] = self.axs['labelC'].text(self.xyfig['Im_C'][0],self.xyfig['Im_C'][1],'Im:'+str(self.currentsagslice.get()),color='w')
        self.labels['Im_D'] = self.axs['labelD'].text(self.xyfig['Im_D'][0],self.xyfig['Im_D'][1],'Im:'+str(self.currentcorslice.get()),color='w')

    # special update for previewing BLAST enhancing lesion in 2d
    # probably not used anymore
    def updateslice_blast(self,event=None):
        slice = self.currentslice.get()
        self.ui.set_currentslice()
        self.ui.runblast(currentslice=slice)
        # self.vslicenumberlabel['text'] = '{}'.format(slice)
        self.canvas.draw()

    # update for previewing final segmentation in 2d
    # was never fully implemented and not used anymore. 
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
        if ax < 1:
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
        if self.ui.dataselection in ['seg_raw_fusion','seg_fusion']:
            # for ax in range(2):
            for ax in ['t1+']: # self.ui.channellist
                # vmin = self.level[ax] - self.window[ax]/2
                # vmax = self.level[ax] + self.window[ax]/2
                vmin = self.wl[ax][1] - self.wl[ax][0]/2
                vmax = self.wl[ax][1] + self.wl[ax][0]/2
                if False:
                    self.ui.data[0].dset[ax]['d'] = self.ui.caseframe.rescale(self.ui.data[0].dset[ax+'_copy']['d'],vmin=vmin,vmax=vmax)

    # clip the raw data to window and level settings
    def clipwl_raw(self):
        # for ax in range(2):
        for ax in ['t1+']:
            vmin = self.wl[ax][1] - self.wl[ax][0]/2
            vmax = self.wl[ax][1] + self.wl[ax][0]/2
            self.ui.data[self.ui.s].dset[ax]['d'] = self.ui.caseframe.rescale(self.ui.data[self.ui.s].dset[ax]['d'],vmin=vmin,vmax=vmax)

    def restorewl_raw(self,dt):
        if False:
            self.ui.data[0].dset[dt]['d'] = copy.deepcopy(self.ui.data[0].dset[dt+'_copy']['d'])

    # main method for BLAST stats in the normal image data
    def normalslice_callback(self,event=None):
        print('normal stats')
        # do kmeans
        # Creates a matrix of voxels for normal brain slice
        # Gating Routine
        s = self.ui.s # local ref

        # main 3d mode for stats
        self.normalslice = None
        region_of_support = np.where(self.ui.data[s].dset['raw']['t1+']['d']*self.ui.data[s].dset['raw']['flair']['d'] >0)
        vset = np.zeros_like(region_of_support,dtype='float')
        # for i in range(3):
        for i,ax in enumerate(['t1+','flair']):
            vset[i] = np.ravel(self.ui.data[s].dset['z'][ax]['d'][region_of_support])

        # kmeans to calculate statistics for brain voxels
        # awkward. indices here are hard-coded according to enumeration above.
        X={}
        X['ET'] = np.column_stack((vset[1],vset[0]))
        # T2 hyper values will just be the same as ET since we do not have plain T2 available for rad nec.
        X['T2 hyper'] = np.column_stack((vset[1],vset[0]))

        # no T2 hyper processing in SAM viewer
        for i,layer in enumerate(['ET']):
            np.random.seed(1)
            kmeans = KMeans(n_clusters=2,n_init='auto').fit(X[layer])
            background_cluster = np.argmin(np.power(kmeans.cluster_centers_[:,0],2)+np.power(kmeans.cluster_centers_[:,1],2))

            # Calculate stats for brain cluster. currently hard-coded to study #0
            self.ui.blastdata[s]['blast']['params'][layer]['stdt12'] = np.std(X[layer][kmeans.labels_==background_cluster,1])
            self.ui.blastdata[s]['blast']['params'][layer]['stdflair'] = np.std(X[layer][kmeans.labels_==background_cluster,0])
            self.ui.blastdata[s]['blast']['params'][layer]['meant12'] = np.mean(X[layer][kmeans.labels_==background_cluster,1])
            self.ui.blastdata[s]['blast']['params'][layer]['meanflair'] = np.mean(X[layer][kmeans.labels_==background_cluster,0])

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
            # removed 'bc' for SAM mode
            for sl in ['t12','flair']:
                self.ui.roiframe.sliders[layer][sl]['state']='normal'
                self.ui.roiframe.sliders[layer][sl].bind("<ButtonRelease-1>",Command(self.ui.roiframe.updateslider,layer,sl))
        # since we finish the on the T2 hyper layer, have this slider disabled to begin with
        # self.ui.roiframe.sliders['ET']['t12']['state']='disabled'

    # run 2d SAM on available prompt. currently this is either a bbox or a single point
    # the ROI is hardcoded [1], there can't be multiple roi's.
    def sam2d_callback(self):

        # switch roi context
        self.ui.roi = self.ui.rois['sam']
        
        print('run sam 2D')
        if len(list(self.ui.roi[self.ui.s][self.ui.currentroi].bboxs.keys())) == 0:
            print('No bbox\'s defined')
            return
        if self.ui.currentslice not in list(self.ui.roi[self.ui.s][self.ui.currentroi].bboxs.keys()):
            print('No bbox defined in current slice')
            return
        if self.ui.roi[self.ui.s][self.ui.currentroi].bboxs[self.ui.currentslice]['p1'] is None:
            prompt = 'point'
        else:
            prompt = 'bbox'
        self.ui.roiframe.save_prompts(sam=self.ui.currentslice,mask='bbox')
        self.ui.roiframe.segment_sam(tag='manual',prompt=prompt)
        self.ui.roiframe.ROIstats(save=True,tag='manual_'+prompt,roitype='sam',slice=self.ui.currentslice)
        # switch to SAM display
        self.ui.roiframe.set_overlay('SAM')
        # in SAM, the ET bounding box segmentation is interpreted directly as TC
        self.ui.roiframe.layerSAM_callback(layer='TC')

    # run 3d SAM on all bbox's. currently this is only for handdrawn not BLAST
    # might not be needed and not properly coded
    def sam3d_callback(self):
        print('run sam 3D')
        if len(list(self.ui.roi[self.ui.s][self.ui.currentroi].bboxs.keys())) == 0:
            print('No bbox\'s defined')
            return
        self.ui.roiframe.saveROI(mask='bbox')
        self.ui.roiframe.segment_sam(tag='bbox')
        # switch to SAM display
        self.ui.roiframe.set_overlay('SAM')
        # in SAM, the ET bounding box segmentation is interpreted directly as TC
        self.ui.roiframe.layerSAM_callback(layer='TC')

    #######################
    # mouse/keyboard events
    #######################

    # mouse drag event for 3d crosshair overlay
    # currently this is over-ridden for the hard-coded axes coords. should better be generalized
    def b1motion_crosshair(self,event):
        self.canvas.get_tk_widget().config(cursor='tcross')
        # no adjustment from outside the pane
        if event.widget.widgetName != 'canvas':
            return
        # which artist axes was clicked
        a = self.tbar.select_artist(event)
        if a is None:
            return
        aax = a.axes._label
        # calculate data coords for all axes relative to clicked axes
        # event origin is bottom left, but data origin is top/left
        # TODO: matrix transforms for sag/cor/ax
        # mouse event returns display coords but which are still flipped in y compared to matplotlib display coords.
        # because there are two axes stacked for sag/cor, the correction for this flip involves +/- 1*self.dim[0]
        x,y = self.axs[aax].transData.inverted().transform((event.x,event.y))
        if False:
            print('crosshair',event.x,event.y,x,y,aax)
        if aax in ['A']:
            y = self.dim[1]-y
            y1 = self.currentslice.get()
            self.draw_crosshair('A',x,y)
            self.draw_crosshair('C',y,y1)
            self.draw_crosshair('D',x,y1)
            self.currentsagslice.set(int(x))
            self.currentcorslice.set(int(y))
        elif aax == 'C':
            y = -y
            y1 = self.currentsagslice.get()
            self.draw_crosshair('A',y1,x)
            self.draw_crosshair('C',x,y)
            self.draw_crosshair('D',y1,y)
            self.currentslice.set(int(y))
            self.currentcorslice.set(int(x))
        elif aax == 'D':
            y = -y + 2*self.dim[0]
            y1 = self.currentcorslice.get()
            self.draw_crosshair('A',x,y1)
            self.draw_crosshair('C',y1,y)
            self.draw_crosshair('D',x,y)
            self.currentslice.set(int(y))
            self.currentsagslice.set(int(x))

        self.updateslice()

        # repeating this here because there are some automatic tk backend events which 
        # can reset it during a sequence of multiple drags
        self.canvas.get_tk_widget().config(cursor='tcross')


    # methods for drawing bounding boxes

    # from global screen pixel event coords, calculate the data coords within the clicked panel
    # because there is still a flip in the y coordinates between matplotlib and gui event,
    # for the bottom two panels C,D the screen pixel matrix dimension must first be subtracted, 
    # negation performed, then the data pixel matrix dimension added back on.
    def calc_panel_xy(self,ex,ey,ax):
        if ax > 'B':
            ey -= int(self.ui.current_panelsize*self.ui.config.dpi/2)
        x,y = self.axs[ax].transData.inverted().transform((ex,ey))
        # y = -y
        y = self.dim[1]-y
        if ax > 'B':
            y += self.dim[1]
        return x,y

    # mouse drag event creates bbox
    def b1motion_bbox(self,event=None):
        self.canvas.get_tk_widget().config(cursor='sizing')
        # no adjustment from outside the pane
        if event.y < 0 or event.y > self.ui.current_panelsize*self.config.dpi:
            return
        # which artist axes was clicked
        a = self.tbar.select_artist(event)
        if a is None:
            return
        aax = a.axes._label
        if aax != self.bbox['ax']:
            self.clear_bbox()
            return
        # mouse event returns display coords but which are still flipped in y compared to matplotlib data coords.
        x,y = self.calc_panel_xy(event.x,event.y,aax)
        self.draw_bbox(x,y,aax)
        self.updateslice(event=event)

        # repeating this here because there are some automatic tk backend events which 
        # can reset it during a sequence of multiple drags
        self.canvas.get_tk_widget().config(cursor='sizing')

    # record coordinates of button click
    def b1click(self,event):
        ex,ey = np.copy(event.x),np.copy(event.y)
        self.canvas.get_tk_widget().config(cursor='sizing')
        if self.bbox['plot']:
            self.clear_bbox()
        # no action if outside the pane
        if event.widget.widgetName != 'canvas':
            return
        # which artist axes was clicked
        a = self.tbar.select_artist(event)
        if a is None:
            return
        aax = a.axes._label
        # data coordinates of the screen click event
        x,y = self.calc_panel_xy(ex,ey,aax)
        if False:
            pdim = int(self.ui.current_panelsize*self.ui.config.dpi/2)
            if ey > pdim:
                ey -= pdim 
            x,y = self.axs[aax].transData.inverted().transform((ex,ey))
            y = -y + self.dim[1]
        self.bbox['ax'] = aax
        self.bbox['p0'] = (x,y)
        self.bbox['slice'] = self.ui.currentslice
        if False:
            self.axs[aax].plot(x,y,'+')
        self.canvas.get_tk_widget().config(cursor='sizing')
        # bind the release event
        self.ui.root.bind('<ButtonRelease-1>',self.b1record)

    # record bbox after left-button release
    def b1record(self,event=None):
        self.record_bbox()
        self.ui.root.unbind('<ButtonRelease-1>')

    # draw line for current linear bbox
    def draw_bbox(self,x,y,ax):
        if self.bbox['ax'] != ax or self.bbox['p0'] is None:
            return
        if self.bbox['plot'] is not None:
            try:
                self.axs[self.bbox['ax']].lines[0].remove() # coded for only 1 line
                self.bbox['plot'] = None
            except ValueError as e:
                print(e)
        lx = np.array([self.bbox['p0'][0],x,x,self.bbox['p0'][0],self.bbox['p0'][0]])
        ly = np.array([self.bbox['p0'][1],self.bbox['p0'][1],y,y,self.bbox['p0'][1]])
        self.bbox['plot'] = self.axs[ax].plot(lx,ly,'b',clip_on=True)[0]
        self.bbox['p1'] = (x,y)
        self.bbox['l'] = np.sqrt(np.power(lx[2]-lx[0],2)+np.power(ly[2]-ly[0],2))
        self.ui.set_message(msg='diameter = {:.1f}'.format(self.bbox['l']))
        return
    
    # draw a point prompt
    def draw_point(self):
        if self.bbox['p0'] is None:
            return
        if self.bbox['p1'] is not None:
            return
        if self.bbox['plot'] is not None:
            try:
                self.axs[self.bbox['ax']].lines[0].remove() # coded for only 1 line
                self.bbox['plot'] = None
            except ValueError as e:
                print(e)
        self.bbox['plot'] = self.axs[self.bbox['ax']].plot(self.bbox['p0'][0],self.bbox['p0'][1],'b+',clip_on=True)[0]
        self.ui.set_message(msg='point = {:.1f},{:.1f}'.format(self.bbox['p0'][0],self.bbox['p0'][1]))


    # remove existing bbox, for using during interactive draw only
    def clear_bbox(self):
        if self.bbox['plot'] is not None:
            self.axs[self.bbox['ax']].lines[0].remove() # coded for only 1 line
        self.bbox = {'ax':None,'p0':None,'p1':None,'plot':None,'l':None,'slice':None}
        self.ui.clear_message()
        self.canvas.draw()

    # compute mask array from bounding box. 
    # this is a round-about arrangement, since sam.py script
    # recomputes the bbox from the mask, but don't want to implement
    # external file storage for bbox's directly at this time. 
    # TODO. box extension
    def create_mask_from_bbox(self, bbox, box_extension=0):
        if np.shape(bbox) == (2,2):
            vxy = np.array([[bbox[0][0],bbox[0][1]],
                        [bbox[1][0],bbox[0][1]],
                        [bbox[1][0],bbox[1][1]],
                        [bbox[0][0],bbox[1][1]]])
            vyx = np.flip(vxy,axis=1)
            bbox_path = Path(vyx,closed=False)
            mask = np.zeros((self.dim[1],self.dim[2]),dtype='uint8')
            mask = bbox_path.contains_points(np.array(np.where(mask==0)).T)
            mask = np.reshape(mask,(self.dim[1],self.dim[2]))        
        elif np.shape(bbox) == (2,):
            mask = np.zeros((self.dim[1],self.dim[2]),dtype='uint8')
            bbox = np.round(np.array(bbox)).astype('int')
            mask[bbox[1],bbox[0]] = 1
        return mask

    # copy existing bbox in current slice to 'bbox' field of the roi. 
    def record_bbox(self):

        # switch roi context
        self.ui.roi = self.ui.rois['sam']

        # for SAM bbox prompt mode, it is hard-coded for one ROI only
        # self.ui.currentroi should always equal 1
        # if self.ui.currentroi > 1 or len(self.ui.roiframe.roilist) > 1:
        #     print('All existing ROI\'s must be cleared before generating a bbox ROI')
        #     self.ui.set_message('All existing ROI\'s must be cleared before generating a bbox ROI')
        #     return
        # if self.ui.currentroi == 0:
        self.ui.roiframe.createROI(bbox = self.bbox)
        # should be in createROI
        # self.ui.roi[self.ui.s][self.ui.currentroi].data['bbox'] = np.zeros(self.dim)
        assert 'p1' in self.bbox.keys()
        # if self.bbox['p1'] is not None: #bbox
        #     self.ui.roi[self.ui.s][self.ui.currentroi].data['bbox'][self.ui.currentslice] = self.create_mask_from_bbox((self.bbox['p0'],self.bbox['p1']))
        # else: # pointprompt
        #     self.ui.roi[self.ui.s][self.ui.currentroi].data['bbox'][self.ui.currentslice] = self.create_mask_from_bbox((self.bbox['p0']))
            # also need to plot here since there was no show_bbox from a drag event
        if self.bbox['p1'] is None:
            self.draw_point()
        
        # here slice is the key for a group of multiple bboxs.
        self.ui.roi[self.ui.s][self.ui.currentroi].bboxs[self.ui.currentslice] = copy.deepcopy(self.bbox)
        if False:
            self.bbox = {'ax':None,'p0':None,'p0':None,'plot':None,'l':None,'slice':None}

        return

    # re-display an existing bbox
    def show_bbox(self,roi=None):

        # roi context
        self.ui.roi = self.ui.rois['sam']

        self.bbox = copy.deepcopy(self.ui.roi[self.ui.s][self.ui.currentroi].bboxs[self.ui.currentslice])
        self.bbox['plot'] = None
        if self.bbox['p1'] is not None:
            self.draw_bbox(self.bbox['p1'][0],self.bbox['p1'][1],self.bbox['ax'])
        else:
            self.draw_point()
        self.canvas.draw()

    # display or remove bbox in the current slice
    def update_bboxs(self):

        # roi context
        self.ui.roi = self.ui.rois['sam']

        self.clear_bbox()
        slice = self.ui.currentslice
        if slice in list(self.ui.roi[self.ui.s][self.ui.currentroi].bboxs.keys()):
            self.bbox = copy.deepcopy(self.ui.roi[self.ui.s][self.ui.currentroi].bboxs[slice])
            self.show_bbox()
            self.canvas.draw()


    ########    
    # timing
    ########

    # start the timer
    def timer(self):
        if self.timing.get():
            self.tstart = time.time()
            self.ct = np.copy(self.tstart)
            self.timingtext.set('on')
        else:
            self.tstop = time.time()
            self.elapsedtime = self.tstop - self.tstart
            self.timingtext.set('off')
            self.ui.set_message('Elapsed time = {:d} seconds'.format(int(np.ceil(self.elapsedtime))))
        return
 