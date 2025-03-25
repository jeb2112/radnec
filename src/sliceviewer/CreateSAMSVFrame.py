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

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

matplotlib.use('TkAgg')
import SimpleITK as sitk
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice
import scipy
import cc3d

from src.NavigationBar import NavigationBar

from src.sliceviewer.CreateSVFrame import *
from src.SSHSession import SSHSession
from src.sam.SAM import SAM

#####################################
# Slice Viewer for SAM segmentation
#####################################

class CreateSAMSVFrame(CreateSliceViewerFrame):
    def __init__(self,parentframe,ui=None,padding='10',style=None):
        super().__init__(parentframe,ui=ui,padding=padding,style=style)

        # ui variables

        self.axslicelabel = None
        self.corslicelabel = None
        self.sagslicelabel = None
        self.windowlabel = None
        self.levellabel = None
        self.overlay_type = tk.IntVar(value=self.config.BlastOverlayType)
        self.prompt_type = tk.StringVar(value='point')

        # user interaction time
        self.dwelltime = None
        self.timingtext = tk.StringVar(value='off')
        self.timing = tk.IntVar(value=0)
        self.elapsedtime = 0
        # sam object
        self.sam = SAM(ui=self.ui)

        # for use with bbox/point tool
        self.bbox = {'ax':None,'p0':None,'p1':None,'plot':None,'l':None,'ch':None}        
        self.pt = {'ax':None,'p0':None,'plot':None,'ch':None,'fg':True}        

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
        self.run2dSAM = ttk.Button(self.normal_frame,text='2D',command=Command(self.sam2d_callback,prompt=self.prompt_type.get(),do_ortho=None),state='disabled')
        self.run2dSAM.grid(row=1,column=3,sticky='w')
        # button to run 3d SAM on current BLAST ROI (bbox)
        self.run3dSAM = ttk.Button(self.normal_frame,text='3D',command=Command(self.sam3d_callback,remote=self.ui.config.AWS,do_ortho=None),state='disabled')
        self.run3dSAM.grid(row=1,column=4,sticky='w')
        # button to select prompt type
        prompt_point_button = ttk.Radiobutton(self.normal_frame,text='point',variable=self.prompt_type,value='point')
        prompt_point_button.grid(row=2,column=3,sticky='w')
        prompt_bbox_button = ttk.Radiobutton(self.normal_frame,text='bbox',variable=self.prompt_type,value='bbox')
        prompt_bbox_button.grid(row=2,column=4,sticky='w')

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
            4*int(self.ui.mainframe_padding)
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
        self.ax4_img = self.axs['D'].imshow(np.zeros((self.dim[0],self.dim[2])),vmin=0,vmax=1,cmap='gray',origin='lower',aspect=1)
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
        if self.ui.roiframe.roioverlayframe.overlay_value['BLAST'].get():
            d = 'd'+self.ui.roiframe.roioverlayframe.layer.get()
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
            self.updatewl(ax=1)
            self.ax4_img.set(cmap='gray')
            self.updatewl(ax=3)
        if wl:   
            # possible latency problem here
            if self.ui.dataselection == 'seg_raw_fusion':
                self.ui.roiframe.roioverlayframe.layer_callback(layer=layer)
            elif self.ui.dataselection == 'seg_fusion':
                self.ui.roiframe.roioverlayframe.layerROI_callback(layer=layer)
            elif self.ui.dataselection == 'raw':
                self.clipwl_raw()

        # if not an event, show an existing bbox, or remove it
        # not using this in the latest SAM viewer
        if False:
            if self.ui.currentroi > 0 and event is None:
                if self.ui.roiframe.roioverlayframe.overlay_value['SAM'].get() == True or str(self.run2dSAM['state']) == 'active':
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

        vmin = self.level[0] - self.window[0]/2
        vmax = self.level[0] + self.window[0]/2

        self.ax_img.set_clim(vmin=vmin,vmax=vmax)
        self.ax3_img.set_clim(vmin=vmin,vmax=vmax)
        self.ax4_img.set_clim(vmin=vmin,vmax=vmax)

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
            # if n4bias hasn't been done then dset['z'] might not be pre-populated, just use the uncorrected 'raw'            
            if not self.ui.data[s].dset['z'][ax]['ex']: 
                self.ui.data[s].dset['z'][ax]['d'] = np.copy(self.ui.data[s].dset['raw'][ax]['d'])
            vset[i] = np.ravel(self.ui.data[s].dset['z'][ax]['d'][region_of_support])

        # kmeans to calculate statistics for brain voxels
        # awkward. indices here are hard-coded according to enumeration above.
        X={}
        X['ET'] = np.column_stack((vset[1],vset[0]))
        # T2 hyper values will just be the same as ET since we do not have plain T2 available for rad nec.
        X['T2 hyper'] = np.column_stack((vset[1],vset[0]))

        for i,layer in enumerate(['ET','T2 hyper']):
            np.random.seed(1)
            kmeans = KMeans(n_clusters=2,n_init='auto').fit(X[layer])
            background_cluster = np.argmin(np.power(kmeans.cluster_centers_[:,0],2)+np.power(kmeans.cluster_centers_[:,1],2))

            # Calculate stats for brain cluster.
            self.ui.blastdata[s]['blast']['params'][layer]['stdt12'] = np.std(X[layer][kmeans.labels_==background_cluster,1])
            self.ui.blastdata[s]['blast']['params'][layer]['stdflair'] = np.std(X[layer][kmeans.labels_==background_cluster,0])
            self.ui.blastdata[s]['blast']['params'][layer]['meant12'] = np.mean(X[layer][kmeans.labels_==background_cluster,1])
            self.ui.blastdata[s]['blast']['params'][layer]['meanflair'] = np.mean(X[layer][kmeans.labels_==background_cluster,0])

            if True:
                plt.figure(7)
                ax = plt.subplot(1,2,i+1)
                plt.scatter(X[layer][kmeans.labels_==1-background_cluster,0],X[layer][kmeans.labels_==1-background_cluster,1],c='b',s=1)
                plt.scatter(X[layer][kmeans.labels_==background_cluster,0],X[layer][kmeans.labels_==background_cluster,1],c='r',s=1)
                ax.set_aspect('equal')
                ax.set_xlim(left=-10,right=10.0)
                ax.set_ylim(bottom=-10,top=10.0)
                plt.text(0,1.02,'{:.3f},{:.3f}'.format(self.ui.blastdata[s]['blast']['params'][layer]['meanflair'],
                                                       self.ui.blastdata[s]['blast']['params'][layer]['stdflair']))

                plt.savefig('/home/jbishop/Pictures/scatterplot_normal.png')
                # plt.show(block=False)
                plt.clf()

        # automatically run BLAST
            self.ui.roiframe.roioverlayframe.layer_callback(layer=layer,updateslice=False,overlay=False)
            self.ui.runblast(currentslice=None,layer=layer)

            # activate thresholds only after normal slice stats are available
            # removed 'bc' for SAM mode
            for sl in ['t12','flair']:
                self.ui.roiframe.sliderframe.sliders[layer][sl]['state']='normal'
                self.ui.roiframe.sliderframe.sliders[layer][sl].bind("<ButtonRelease-1>",Command(self.ui.roiframe.sliderframe.updateslider,layer,sl))
        # since we finish the on the T2 hyper layer, have this slider disabled to begin with
        # self.ui.roiframe.sliderframe.sliders['ET']['t12']['state']='disabled'

    # run 2d SAM on available prompt. currently this is either a bbox or a single point
    def sam2d_callback(self,prompt='bbox',do_ortho=None,remote=False):
    
        print('run sam 2D')

        if remote:
            user = 'ec2-user'
            host = 'ec2-35-182-153-217.ca-central-1.compute.amazonaws.com'
            s1 = SSHSession(user,host)
            # res = s1.run_command('conda activate pytorch')
            # res = s1.run_command('conda info')
        else:
            s1 = None

        if do_ortho is None:
            do_ortho = self.ui.config.SAMortho
        if do_ortho:
            planes = ['ax','sag','cor']
            # set orthogonal slices off the 1st foreground point
            pt = [p for p in self.ui.rois['sam'][self.ui.s][self.ui.currentroi].pts if p['fg']][0]
            self.set_ortho_slice(coords=(pt['p0'][0],pt['p0'][1]))

        else:
            planes = ['ax']

        # for the 2d workflow in this version, there could be multiple SAM point prompts, but they all only 
        # apply to the 'ax' slice, so there isn't the same equivalence in generating the orthogonal
        # 2d segmentations as there was in the previous workflow. further, the multiple points would have to
        # be limited to just one point in the orthogonal slices, and which point to choose would be
        # arbitrary. for now, don't run the orthogonal slices in the 2dcallback
        planes = ['ax']

        for p in planes:
             
            currentslice = self.ui.get_currentslice(ax=p)
             
            if prompt in ['bbox','maskpoint']: # old workflow, derive prompt from existing BLAST segmentation
                self.ui.rois['sam'][self.ui.s][self.ui.currentroi].create_prompts_from_mask( \
                    np.copy(self.ui.rois['blast'][self.ui.s][self.ui.currentroi].data[self.ui.roiframe.layerROI.get()]),prompt=prompt,slice=currentslice,orient=p)
            elif prompt == 'point': # for the new workflow, the 2d point prompt is not derived from a BLAST mask but directly from a list of points
                self.ui.rois['sam'][self.ui.s][self.ui.currentroi].create_prompts_from_points( \
                    np.copy(self.ui.rois['sam'][self.ui.s][self.ui.currentroi].pts),prompt=prompt,slice=currentslice,orient=p)
            self.ui.roiframe.save_prompts(slice=currentslice,orient=p,prompt=prompt)

        # upload prompts to remote
        if remote:
            st1 = time.time()
            self.ui.sam.put_prompts_remote(session=s1)
            upload_time = time.time() - st1
        else:
            upload_time = 0

        st2 = time.time()
        # 'ax' still hard-coded here for the 2d workflow
        self.ui.sam.segment_sam(orient=['ax'],tag='2d',prompt=prompt,session=s1)
        # download results if remote
        if remote:
            download_time = self.ui.sam.get_predictions_remote(session=s1)
        else:
            download_time = 0
        elapsed_time = time.time() - st2
        self.ui.set_message(msg='SAM 2d up = {:.1f}, elapse = {:.1f}, down = {:.1f}'.format(upload_time,elapsed_time,download_time))
        self.ui.root.update_idletasks()

        # do_ortho hard-coded here for 2d workflow
        self.ui.sam.load_sam(tag='2d',prompt=prompt,do_ortho=False,do3d=False)
        # switch to SAM display
        self.ui.roiframe.roioverlayframe.set_overlay('SAM')
        # in SAM, the ET bounding box segmentation is interpreted directly as TC
        self.ui.roiframe.roioverlayframe.layerSAM_callback()

        # activate sam 3d
        self.ui.sliceviewerframe.run3dSAM.configure(state='active')

    # run 3d SAM after a satisfactory 2d SAM result
    def sam3d_callback(self,do_ortho=None,remote=False,prompt='maskpoint'):
        # in this new workflow, BLAST is not run during 2D, so run it here to begin the 3d segmentation
        # then run SAM on all slices with propmts's derived a BLAST ROI. 
        # check for an available blast segmentation or
        # optionally, load an alternative (eg BraTS) mask
        if self.ui.config.UseBraTSMask:
            layer = self.ui.roiframe.roioverlayframe.layerSAM.get()
            self.ui.rois['blast'][self.ui.s][self.ui.currentroi].data[layer] = self.load_brats_mask(layer)
            self.ui.rois['blast'][self.ui.s][self.ui.currentroi].status = True
        else:
            if not self.ui.rois['blast'][self.ui.s][self.ui.currentroi].status:
                # new workflow. take the SAM prompt points, and process them for a BLAST ROI
                self.ui.roiframe.blastpointframe.copy_points()
                self.ui.roiframe.blastpointframe.updateBLASTMask(currentslice=None)
                # any of the clicked points will do to create the BLAST ROI, just use the last one
                self.ui.roiframe.ROIclick(coords = (self.ui.pt[self.ui.s][-1].coords['x'],self.ui.pt[self.ui.s][-1].coords['y']))
                if False: #debugging output of BLAST mask
                    layer = self.ui.roiframe.roioverlayframe.layerSAM.get()
                    outputfilename = os.path.join(self.ui.data[self.ui.s].studydir,'{}_blast.nii'.format(layer))
                    self.ui.data[self.ui.s].writenifti(self.ui.rois['blast'][self.ui.s][self.ui.currentroi].data[layer],
                                                        outputfilename,
                                                        affine=self.ui.data[self.ui.s].dset['raw']['t1+']['affine'])

        # deactivate bbox selection tool if any.
        if self.tbar.mode == "bbox":
            self.tbar.bbox()
        
        print('run SAM 3d')

        if remote:
            user = 'ec2-user'
            host = 'ec2-35-182-153-217.ca-central-1.compute.amazonaws.com'
            s1 = SSHSession(user,host)
        else:
            s1 = None

        if False: # in the previous workflow, point selection mode was activated here. no longer need this.
            self.ui.roiframe.blastpointframe.selectPoint()
            self.ui.roiframe.blastpointframe.setCursor('watch')
        if do_ortho is None:
            do_ortho = self.ui.config.SAMortho
        if do_ortho:
            planes = ['ax','sag','cor']
        else:
            planes = ['ax']

        for p in planes:
 
            if prompt in ['bbox','maskpoint']: # derive prompt from existing BLAST segmentation. in old workflow it was bbox, in new workflow points hence 'maskpoint' option
                self.ui.rois['sam'][self.ui.s][self.ui.currentroi].create_prompts_from_mask( \
                    np.copy(self.ui.rois['blast'][self.ui.s][self.ui.currentroi].data[self.ui.roiframe.roioverlayframe.layerSAM.get()]),prompt=prompt,slice=None,orient=p)
            elif prompt == 'point': # for the new workflow, the 3d multi-slice won't likely ever be directly from a list of points
                self.ui.rois['sam'][self.ui.s][self.ui.currentroi].create_prompts_from_points( \
                    np.copy(self.ui.rois['sam'][self.ui.s][self.ui.currentroi].pts),prompt=prompt,slice=None,orient=p)

            # still need to reconcile maskpoint wtih point
            self.ui.roiframe.save_prompts(orient=p,prompt=prompt)

        # upload prompts to remote
        if remote:
            st1 = time.time()
            self.ui.sam.put_prompts_remote(session=s1,do2d=False)
            upload_time = time.time() - st1
        else:
            upload_time = 0

        st2 = time.time()
        if self.ui.config.SAMortho is False:
            orient = ['ax']
        else:
            orient = None
        self.ui.sam.segment_sam(orient=orient,tag='blast_3d',session=s1,prompt=prompt)
        
        if remote:
            download_time = self.ui.sam.get_predictions_remote(tag = 'blast_3d',session=s1)
        else:
            download_time = 0
        elapsed_time = time.time() - st2
        self.ui.set_message(msg='SAM 3d up = {:.1f}, elapse = {:.1f}, down = {:.1f}'.format(upload_time,elapsed_time,download_time))
        self.ui.root.update_idletasks()

        # maskpoint/point prompt naming
        self.ui.sam.load_sam(tag = 'blast_3d',prompt=prompt,do_ortho=do_ortho)
        # switch to SAM display
        self.ui.roiframe.roioverlayframe.set_overlay('SAM')
        self.ui.roiframe.roioverlayframe.layerSAM_callback()
        self.ui.rois['sam'][self.ui.s][self.ui.currentroi].status = True

        # experimental option. if timer running, stop it.
        if self.timing.get() == True:
            self.timing.set(False)
            self.timer()
            self.ui.rois['sam'][self.ui.s][self.ui.currentroi].stats['elapsedtime'] = np.round(self.elapsedtime*10)/10
            self.ui.set_message(msg='elapsed time = {:.1f}'.format(self.elapsedtime))
        else:
            pass
            # self.ui.set_message(msg="SAM 3d complete")
    
        self.ui.roiframe.blastpointframe.setCursor('arrow')
        return

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

    # convenience method related to the b1motion crosshair
    # for now this is just hard-coded to axis 'A' but should be
    # developed generally
    def set_ortho_slice(self,event=None,coords=None,ax='A'):
        if ax == 'A':
            if event is not None:
                x,y = self.axs[ax].transData.inverted().transform((event.x,event.y))
                if False:
                    y = self.dim[1]-y
            elif coords is not None:
                x,y = coords
            self.currentsagslice.set(int(np.round(x)))
            self.currentcorslice.set(int(np.round(y)))
        self.updateslice()


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

    # record coordinates of left button click
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
        self.ui.root.bind('<ButtonRelease-1>',Command(self.b1record,runsam=False,foreground=True))

    # record coordinates of right button click
    # should be combined with b1click
    def b3click(self,event):
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
        self.ui.root.bind('<ButtonRelease-3>',Command(self.b3record,runsam=False,foreground=False))

    # record bbox/point after left-button release
    def b1record(self,event=None,runsam=True,foreground=True):
        if self.bbox['p1'] is None: # ie there has been no drag during the b1 click event
            self.record_pt(foreground=True)
        else:
            if np.sum(np.abs(np.array(self.bbox['p1'])-np.array(self.bbox['p0']))) < 5: # catch an arbitrarily small bbox
                self.bbox['p1'] = None
                self.record_pt(foreground=True)
            else:
                self.record_bbox() # there has been a mouse drag
        self.ui.root.unbind('<ButtonRelease-1>')
        if runsam:
            self.sam2d_callback()

    # record control point after right-button release
    def b3record(self,event=None,runsam=False,foreground=True):
        self.record_pt(foreground=False)
        self.ui.root.unbind('<ButtonRelease-3>')
        if runsam:
            self.sam2d_callback()

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
    
    # draw a point prompt. with multiple points, this might need to be in ROI.py
    def draw_point(self):
        if self.pt['plot'] is not None:
            try:
                self.axs[self.pt['ax']].lines[0].remove() # coded for only 1 line
                self.pt['plot'] = None
            except ValueError as e:
                print(e)
        if self.pt['fg']:
            self.pt['plot'] = self.axs[self.pt['ax']].plot(self.pt['p0'][0],self.pt['p0'][1],'b+',clip_on=True)[0]
        else:
            self.pt['plot'] = self.axs[self.pt['ax']].plot(self.pt['p0'][0],self.pt['p0'][1],'r+',clip_on=True)[0]
        self.canvas.draw()
        self.ui.set_message(msg='point = {:.1f},{:.1f}'.format(self.pt['p0'][0],self.pt['p0'][1]))


    # remove existing bbox, for using during interactive draw only
    def clear_bbox(self):
        if self.bbox['plot'] is not None:
            self.axs[self.bbox['ax']].lines[0].remove() # coded for only 1 line
        self.bbox = {'ax':None,'p0':None,'p1':None,'plot':None,'l':None,'slice':None}
        self.ui.clear_message()
        self.canvas.draw()

    # remove all existing points
    def clear_points(self):
        if self.pt['plot'] is not None:
            for l in self.axs[self.pt['ax']].lines:
                l.remove()
        self.pt = {'ax':None,'p0':None,'plot':None,'ch':None,'slice':None, 'fg':True}
        self.ui.clear_message()
        if self.canvas is not None:
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
    def record_bbox(self,foreground=False):

        assert 'p1' in self.bbox.keys()
        self.bbox.fg = foreground
        self.ui.roiframe.createROI(bbox = self.bbox)

        # also need to plot here since there was no show_bbox from a drag event
        if self.bbox['p1'] is None:
            self.draw_point()
        
        # here slice is the key for a group of multiple bboxs.
        self.ui.rois['sam'][self.ui.s][self.ui.currentroi].bboxs[self.ui.currentslice] = copy.deepcopy(self.bbox)
        if False:
            self.bbox = {'ax':None,'p0':None,'p0':None,'plot':None,'l':None,'slice':None, 'fg':False}

        return


    # copy existing point in current slice to 'pt' field of the roi. 
    # currently using bbox from b1,b3click to be tidied up
    def record_pt(self,foreground=False):

        self.pt = {'ax':None,'p0':None,'plot':None,'ch':None,'slice':None, 'fg':True}
        for k in ['ax','p0','slice']:
            self.pt[k] = self.bbox[k]
        self.pt['fg'] = foreground
        # initial point creates a new ROI, subsequent points add to it
        if self.ui.currentroi == 0: # if no roi's exist, this point creates one
            self.ui.roiframe.createROI(pt = self.pt)
        elif self.ui.rois['sam'][self.ui.s][self.ui.currentroi].status == True: # if a competed 3d SAM roi exists, this point creates a new one
            self.ui.roiframe.createROI(pt = self.pt)
        else: # this point adds to and updates the current roi
            self.ui.roiframe.updateROI(pt = self.pt)

        # also need to plot here
        self.draw_point()
        
        # copy sliceviewer point to roi.
        self.ui.rois['sam'][self.ui.s][self.ui.currentroi].pts.append(copy.deepcopy(self.pt))

        return


    # re-display an existing bbox
    def show_bbox(self,roi=None):

        # roi context
        self.ui.roi = self.ui.rois['sam']

        self.bbox = copy.deepcopy(self.ui.roi[self.ui.s][self.ui.currentroi].bboxs[self.ui.currentslice])
        self.bbox['plot'] = None
        if self.bbox['ax'] is None:
            return
        if self.bbox['p1'] is not None:
            self.draw_bbox(self.bbox['p1'][0],self.bbox['p1'][1],self.bbox['ax'])
        elif self.bbox['p0'] is not None:
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
            self.prevtime = np.copy(self.tstart)
            self.timingtext.set('on')
        else:
            self.tstop = time.time()
            self.elapsedtime = self.tstop - self.tstart
            self.timingtext.set('off')
            self.ui.set_message('Elapsed time = {:d} seconds'.format(int(np.ceil(self.elapsedtime))))
        return
 

    #######
    # other
    #######

    # set initial window/level values. 
    # in the SAM viewer there is only 1 axis, and 1 channel displayed
    # so the window/level here is arbitrarily extended for 2 axes
    # to be compatible. some arrangement to store for multiple studies is needed.
    def setwl(self):
        self.level = []
        self.window = []
        # for w in ['A','B']:
        ch = self.ui.chselection
        for ax in range(2):
            if self.ui.data[self.ui.s].dset['raw'][ch]['ex']:
                self.level.append(self.ui.data[self.ui.s].dset['raw'][ch]['l'])
                self.window.append(self.ui.data[self.ui.s].dset['raw'][ch]['w'])
            else:
                raise ValueError('No data for channel {}'.format(ch))
        # arbitrarily using timepoint0 here
        self.level = np.array(self.level)
        self.window = np.array(self.window)
        return

    # load an alternate mask.
    # just hard-coded for BraTS now.
    def load_brats_mask(self,layer='WT'):

        # BraTS 2024 convention
        layersdict = {'WT':2,'TC':1,'ET':3}
        # convention to accumulate layers
        layersdict = {'WT':[1,2,3],'TC':1,'ET':3}

        fname = glob.glob(os.path.join(self.ui.data[self.ui.s].studydir,'BraTS-*-seg.nii.gz'))
        if len(fname) == 1:
            mask,_ = self.ui.data[self.ui.s].loadnifti(os.path.split(fname[0])[1],os.path.split(fname[0])[0],type='uint8')
            mask = np.where(np.isin(mask,layersdict[layer]),1,0)
            CC_labeled = cc3d.connected_components(mask,connectivity=6)
            # select the lesion based on the first available clickpoint

            clicked_point =  self.ui.rois['sam'][self.ui.s][self.ui.currentroi].pts[0]
            objectnumber = CC_labeled[clicked_point['slice'],int(np.round(clicked_point['p0'][1])),int(np.round(clicked_point['p0'][0]))]
            mask = (CC_labeled == objectnumber).astype('uint8')

            return mask
        else:
            return None