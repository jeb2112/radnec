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
from cProfile import Profile
from pstats import SortKey,Stats
from enum import Enum

matplotlib.use('TkAgg')
import SimpleITK as sitk
import itk
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice

from src.NavigationBar import NavigationBar
from src.FileDialog import FileDialog

# convenience for indexing data dict
class data(Enum):
    T1 = 0
    FLAIR = 1
    T2 = 2
# utility class for callbacks with args
class Command():
    def __init__(self, callback, *args, **kwargs):
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def __call__(self,event):
        return self.callback(*self.args, **self.kwargs)
    
class EventCallback():
    def __init__(self, callback, *args, **kwargs):
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def __call__(self,event):
        return self.callback(event,*self.args, **self.kwargs)

# base for various frames
class CreateFrame():
    def __init__(self,frame,ui=None,padding='10',style=None):
        self.ui = ui
        self.parentframe = frame # parent
        self.frame = ttk.Frame(self.parentframe,padding=padding,style=style)
        self.config = self.ui.config
        self.padding = padding
        self.fstyle = ttk.Style()

##############
# Slice Viewer
##############

class CreateSliceViewerFrame(CreateFrame):
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
        self.ax_img = self.axs['A'].imshow(np.zeros((self.dim[1],self.dim[2])),vmin=0,vmax=1,cmap='gray',origin='upper',aspect=1)
        self.ax2_img = self.axs['B'].imshow(np.zeros((self.dim[1],self.dim[2])),vmin=0,vmax=1,cmap='gray',origin='upper',aspect=1)
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
        self.xyfig['Im_A']= figtrans['A'].transform((5,25))
        self.xyfig['W_A'] = figtrans['A'].transform((int(self.dim[1]/2),self.dim[1]-10))
        self.xyfig['L_A'] = figtrans['A'].transform((int(self.dim[1]*3/4),self.dim[1]-10))
        self.xyfig['W_B'] = figtrans['B'].transform((int(self.dim[1]/2),self.dim[1]-10))
        self.xyfig['L_B'] = figtrans['B'].transform((int(self.dim[1]*3/4),self.dim[1]-10))
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
        if self.ui.roiframe.layer.get() == 'ET':
            self.ax_img.set(data=self.ui.data[self.ui.dataselection][0,slice,:,:])
        else:
            self.ax_img.set(data=self.ui.data[self.ui.dataselection][2,slice,:,:])
        self.ax2_img.set(data=self.ui.data[self.ui.dataselection][1,slice,:,:])
        self.ax3_img.set(data=self.ui.data[self.ui.dataselection][self.sagcordisplay.get(),:,:,slicesag])
        self.ax4_img.set(data=self.ui.data[self.ui.dataselection][self.sagcordisplay.get(),:,slicecor,:])
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
        self.labels['W_A'] = self.axs['labelA'].text(self.xyfig['W_A'][0],self.xyfig['W_A'][1],'W = '+'{:d}'.format(int(self.window[0]*255)),color='w')
        self.labels['L_A'] = self.axs['labelA'].text(self.xyfig['L_A'][0],self.xyfig['L_A'][1],'L = '+'{:d}'.format(int(self.level[0]*255)),color='w')
        self.labels['W_B'] = self.axs['labelB'].text(self.xyfig['W_B'][0],self.xyfig['W_B'][1],'W = '+'{:d}'.format(int(self.window[1]*255)),color='w')
        self.labels['L_B'] = self.axs['labelB'].text(self.xyfig['L_B'][0],self.xyfig['L_B'][1],'L = '+'{:d}'.format(int(self.level[1]*255)),color='w')
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

    def b1release(self,event=None):
        self.b1x = self.b1y = None

    def focus(self,event):
        self.canvas.get_tk_widget().focus_set()

    # keyboard for slice selection
    def keyboard_slice(self,event):
        # print(event,event.widget)
        if event.y < 0 or event.y > self.ui.current_panelsize*self.ui.config.dpi:
            return
        a = self.tbar.select_artist(event)
        if a is None:
            return
        aax = a.axes._label
        if aax in ['A','B']:
            item = self.currentslice
        elif aax == 'C':
            item = self.currentsagslice
        elif aax == 'D':
            item = self.currentcorslice
        s = item.get()

        if event.keysym == 'Up':
            s += 1
        elif event.keysym == 'Down':
            s -= 1

        item.set(s)

        if False:
            with Profile() as profile:
                self.updateslice()
                (
                    Stats(profile)
                    .strip_dirs()
                    .sort_stats(SortKey.CUMULATIVE)
                    .print_stats(15)
                )
        else:
            self.updateslice()

    # mouse wheel for slice selection
    def mousewheel_win32(self,event):
        # print(event,event.widget,event.delta)
        if event.y < 0 or event.y > self.ui.current_panelsize*self.ui.config.dpi:
            return
        if event.x < 2*self.ui.current_panelsize*self.ui.config.dpi:
            item = self.currentslice
            maxslice = self.dim[0]-1
        else:
            if event.y <= (self.ui.current_panelsize*self.ui.config.dpi)/2:
                item = self.currentsagslice
            else:
                item = self.currentcorslice
            maxslice = self.dim[1]-1
        newslice = item.get() + event.delta/120
        newslice = min(max(newslice,0),maxslice)
        item.set(newslice)
        self.updateslice()

        return

    # mouse wheel for slice selection
    def mousewheel(self,event,key=None):
        # print(event,event.time,event.widget,event.delta,key)
        if event.y < 0 or event.y > self.ui.current_panelsize*self.ui.config.dpi:
            return
        # queue mousewheel events for latency
        if key is None:
            if abs(event.time-self.prevtime) < 100:
                if event.num == 4:
                    self.sliceinc += 1
                elif event.num == 5:
                    self.sliceinc -= 1
                self.prevtime = event.time
                if abs(self.sliceinc) == 5:
                    self.canvas.get_tk_widget().event_generate('<<MyMouseWheel>>',when="tail",x=event.x,y=event.y)
                return
            else:
                self.prevtime = event.time
                if event.num == 4:
                    self.sliceinc = 1
                elif event.num == 5:
                    self.sliceinc = -1

        if event.x < 2*self.ui.current_panelsize*self.ui.dpi:
            item = self.currentslice
            maxslice = self.dim[0]-1
        else:
            if event.y <= (self.ui.current_panelsize*self.ui.dpi)/2:
                item = self.currentsagslice
            else:
                item = self.currentcorslice
            maxslice = self.dim[1]-1
        newslice = item.get() + self.sliceinc
        newslice = min(max(newslice,0),maxslice)

        item.set(newslice)
        self.updateslice()
        if key == 'Key':
            self.prevtime = 0
            self.sliceinc = 0
            # self.canvas.get_tk_widget().unbind('<ButtonPress>')

    # mouse drag for slice selection
    def b3motion_reset(self,event):
        self.b3y=None

    def b3motion(self,event):
        if event.y < 0 or event.y > self.ui.current_panelsize*self.ui.dpi:
            return
        # no adjustment if nav bar is activated
        if 'zoom' in self.tbar.mode:
            return
        if event.x < 2*self.ui.current_panelsize*self.ui.dpi:
            item = self.currentslice
            maxslice = self.dim[0]-1
        else:
            if event.y <= (self.ui.current_panelsize*self.ui.dpi)/2:
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

    # mouse drag event for 3d crosshair overlay
    def b1motion_crosshair(self,event):
        self.canvas.get_tk_widget().config(cursor='tcross')
        # no adjustment from outside the pane
        if event.y < 0 or event.y > self.ui.current_panelsize*self.config.dpi:
            return
        # which artist axes was clicked
        a = self.tbar.select_artist(event)
        if a is None:
            return
        aax = a.axes._label
        # calculate data coords for all axes relative to clicked axes
        # TODO: matrix transforms for sag/cor/ax
        # mouse event returns display coords but which are still flipped in y compared to matplotlib display coords.
        # because there are two axes stacked for sag/cor, the correction for this flip involves +/- 1*self.dim[0]
        x,y = self.axs[aax].transData.inverted().transform((event.x,event.y))
        if aax in ['A','B']:
            y = self.dim[1]-y
            y1 = self.currentslice.get()
            self.draw_crosshair('A',x,y)
            self.draw_crosshair('B',x,y)
            self.draw_crosshair('C',y,y1)
            self.draw_crosshair('D',x,y1)
            self.currentsagslice.set(int(x))
            self.currentcorslice.set(int(y))
        elif aax == 'C':
            y = -y
            y1 = self.currentsagslice.get()
            self.draw_crosshair('A',y1,x)
            self.draw_crosshair('B',y1,x)
            self.draw_crosshair('C',x,y)
            self.draw_crosshair('D',y1,y)
            self.currentslice.set(int(y))
            self.currentcorslice.set(int(x))
        elif aax == 'D':
            y = -y + 2*self.dim[0]
            y1 = self.currentcorslice.get()
            self.draw_crosshair('A',x,y1)
            self.draw_crosshair('B',x,y1)
            self.draw_crosshair('C',y1,y)
            self.draw_crosshair('D',x,y)
            self.currentslice.set(int(y))
            self.currentsagslice.set(int(x))

        self.updateslice()

        # repeating this here because there are some automatic tk backend events which 
        # can reset it during a sequence of multiple drags
        self.canvas.get_tk_widget().config(cursor='tcross')

    def draw_crosshair(self,ax,x,y):
        for hv in ['h','v']:
            if self.lines[ax][hv]:
                try:
                    Artist.remove(self.lines[ax][hv])
                except ValueError as e:
                    print(e)
        self.lines[ax]['h'] = self.axs[ax].axhline(y=y,clip_on=True)
        self.lines[ax]['v'] = self.axs[ax].axvline(x=x,clip_on=True)
        # self.canvas.draw()

    def clear_crosshair(self):
        for ax in self.lines.keys():
            for hv in ['h','v']:
                Artist.remove(self.lines[ax][hv])
        self.canvas.draw()

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
        if event.y < 0 or event.y > self.ui.current_panelsize*self.ui.dpi:
            return
        # process only in two main panels
        if event.x <=self.ui.current_panelsize*self.ui.dpi:
            ax = 0
        elif event.x <= 2*self.ui.current_panelsize*self.ui.dpi:
            ax = 1
        else:
            return
        if self.b1x is None:
            self.b1x,self.b1y = copy.copy((event.x,event.y))
            return
        if np.abs(event.x-self.b1x) > np.abs(event.y-self.b1y):
            if event.x-self.b1x > 0:
                self.updatewl(ax=ax,wval=.02)
            else:
                self.updatewl(ax=ax,wval=-.02)
        else:
            if event.y - self.b1y > 0:
                self.updatewl(ax=ax,lval=.02)
            else:
                self.updatewl(ax=ax,lval=-.02)

        self.b1x,self.b1y = copy.copy((event.x,event.y))
        self.update_labels()

        # repeating this here because there are some automatic tk backend events which 
        # can reset it during a sequence of multiple window/level drags
        self.canvas.get_tk_widget().config(cursor='circle')

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
        elif self.ui.OS == 'win32':
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

    # an override method to format the cursor data value.
    def make_cursordata_format(self):
        # current and other are axes
        def format_cursor_data(data):
            if np.isscalar(data):
                return ('[{:.2f}]'.format(data))
            else:
                return ('[{:.2f}, {:.2f}, {:.2f}]'.format(data[0],data[1],data[2]))
        return format_cursor_data

    # an override method to get cursor coords from the bottom (image) axes.
    def make_coord_format(self,labelax, imageax):
        def format_coord(x, y):
            # x, y are data coordinates
            # convert to display coords
            display_coord = labelax.transData.transform((x,y))
            inv = imageax.transData.inverted()
            # convert back to data coords with respect to ax
            img_coord = inv.transform(display_coord)
            return ('x={:.1f}, y={:.1f}'.format(img_coord[0],img_coord[1]))
        return format_coord

    def normalslice_callback(self,event=None):
        print('normal stats')
        # do kmeans
        # Creates a matrix of voxels for normal brain slice
        # Gating Routine

        if self.slicevolume_norm.get() == 0:
            self.normalslice=self.ui.get_currentslice()
            region_of_support = np.where(self.ui.data['raw'][0,self.normalslice]*self.ui.data['raw'][1,self.normalslice]>0) 
            vset = np.zeros_like(region_of_support,dtype='float')
            for i in range(3):
                vset[i] = np.ravel(self.ui.data['raw'][i,self.normalslice][region_of_support])
        else:
            self.normalslice = None
            region_of_support = np.where(self.ui.data['raw'][0]*self.ui.data['raw'][1]*self.ui.data['raw'][2] >0)
            vset = np.zeros_like(region_of_support,dtype='float')
            for i in range(3):
                vset[i] = np.ravel(self.ui.data['raw'][i][region_of_support])
            # t1channel_normal = self.ui.data['raw'][0][region_of_support]
            # flairchannel_normal = self.ui.data['raw'][1][region_of_support]
            # t2channel_normal = self.ui.data['raw'][2][region_of_support]

        # kmeans to calculate statistics for brain voxels
        # X_et = np.column_stack((flair,t1))
        # X_net = np.column_stack((flair,t2))
        X={}
        X['ET'] = np.column_stack((vset[1],vset[0]))
        X['T2 hyper'] = np.column_stack((vset[1],vset[2]))

        for i,layer in enumerate(['ET','T2 hyper']):
            np.random.seed(1)
            kmeans = KMeans(n_clusters=2,n_init='auto').fit(X[layer])
            background_cluster = np.argmin(np.power(kmeans.cluster_centers_[:,0],2)+np.power(kmeans.cluster_centers_[:,1],2))

            # Calculate stats for brain cluster
            self.ui.data['blast']['params'][layer]['stdt12'] = np.std(X[layer][kmeans.labels_==background_cluster,1])
            self.ui.data['blast']['params'][layer]['stdflair'] = np.std(X[layer][kmeans.labels_==background_cluster,0])
            self.ui.data['blast']['params'][layer]['meant12'] = np.mean(X[layer][kmeans.labels_==background_cluster,1])
            self.ui.data['blast']['params'][layer]['meanflair'] = np.mean(X[layer][kmeans.labels_==background_cluster,0])

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


################
# Case Selection
################

class CreateCaseFrame(CreateFrame):
    def __init__(self,parent,ui=None):
        super().__init__(parent,ui=ui)

        self.fd = FileDialog(initdir=self.config.UIdatadir)
        self.datadir = StringVar()
        self.datadir.set(self.fd.dir)
        self.filenames = None
        self.casename = StringVar()
        self.casefile_prefix = None
        self.caselist = {'casetags':[],'casedirs':[]}
        self.n4_check_value = tk.BooleanVar(value=True)
        self.register_check_value = tk.BooleanVar(value=True)
        self.skullstrip_check_value = tk.BooleanVar(value=True)
        self.segnormal_check_value = tk.BooleanVar(value=True)
        self.processed = False

        # case selection
        self.frame.grid(row=0,column=0,columnspan=3,sticky='ew')

        # select directory
        self.fdbicon = PhotoImage(file=os.path.join(self.config.UIResourcesPath,'folder_icon_16.png'))
        # select a parent dir for a group of case sub-dirs
        self.fdbutton = ttk.Button(self.frame,image=self.fdbicon, command=self.select_dir)
        self.fdbutton.grid(row=0,column=2)
        # select file
        self.fdbicon_file = PhotoImage(file=os.path.join(self.config.UIResourcesPath,'file_icon_16.png'))
        self.fdbutton_file = ttk.Button(self.frame,image=self.fdbicon_file,command = self.datafileentry_callback)
        self.fdbutton_file.grid(row=0,column=3)

        self.datadirentry = ttk.Entry(self.frame,width=40,textvariable=self.datadir)
        # event currently a dummy arg since not being used in datadirentry_callback
        self.datadirentry.bind('<Return>',lambda event=None:self.datadirentry_callback(event=event))
        self.datadirentry.grid(row=0,column=4,columnspan=5)
        caselabel = ttk.Label(self.frame, text='Case: ')
        caselabel.grid(column=0,row=0,sticky='we')
        # self.casename.trace_add('write',self.case_callback)
        self.w = ttk.Combobox(self.frame,width=8,textvariable=self.casename,values=self.caselist['casetags'])
        self.w.grid(column=1,row=0)
        self.w.bind("<<ComboboxSelected>>",self.case_callback)

        #processing options
        self.register_check = ttk.Checkbutton(self.frame,text='register',variable=self.register_check_value)
        self.register_check.grid(row=0,column=9,sticky='w')
        self.skullstrip_check = ttk.Checkbutton(self.frame,text='extract',variable=self.skullstrip_check_value)
        self.skullstrip_check.grid(row=0,column=10,sticky='w')
        self.n4_check = ttk.Checkbutton(self.frame,text='N4',variable=self.n4_check_value)
        self.n4_check.grid(row=0,column=11,sticky='w')
        # self.segnormal_check = ttk.Checkbutton(self.frame,text='segment',variable=self.segnormal_check_value)
        # self.segnormal_check.grid(row=0,column=12,sticky='w')


    # callback for file dialog 
    def select_dir(self):
        self.resetCase()
        self.fd.select_dir()
        self.datadir.set(self.fd.dir)
        self.datadirentry.update()
        self.datadirentry_callback()

    # callback for loading by individual files
    def datafileentry_callback(self):
    # def select_file(self):
        self.resetCase()
        self.fd.select_file()
        if len(self.fd.filenames) != 3:
            self.ui.set_message('Three files must be selected')
            return
        self.ui.set_message('')
        self.casedir = os.path.split(self.fd.filenames[0])[0]
        self.caselist['casedirs'] = [os.path.split(self.casedir)[1]]
        self.datadir.set(os.path.split(self.casedir)[0])
        self.filenames = [os.path.split(f)[1] for f in self.fd.filenames]
        # sort assumes minimal tags t1,t2 are present flair image is the 3rd.
        self.filenames = sorted(self.filenames,key=lambda x:(x.lower().find('t1'),x.lower().find('t2')),reverse=True)
        # self.datafileentry_callback()
        # for indiviudally selected files, datadir is the parent and intermediate 'casedirs' not used
        self.caselist['casetags'] = os.path.split(self.datadir.get())[1]
        self.casefile_prefix = ''
        self.config.UIdataroot = self.casefile_prefix
        self.w['values'] = self.caselist['casetags']
        self.w.current(0)
        self.w.config(width=min(20,len(self.caselist['casetags'])))
        self.casename.set(self.caselist['casetags'])
        # self.datadir.set(os.path.split(dir)[0])
        self.casetype = 0
        self.case_callback(files=self.filenames)
        return

    def case_callback(self,casevar=None,val=None,event=None,files=None):
        case = self.casename.get()
        self.ui.set_casename(val=case)
        print('Loading case {}'.format(case))
        self.loadCase(files=files)
        # if normal stats is 3d then seg runs automatically, can show 'seg_raw' directly
        if self.ui.sliceviewerframe.slicevolume_norm.get() == 0:
            self.ui.dataselection = 'raw'
        else:
            self.ui.roiframe.enhancingROI_overlay_value.set(True)
            # self.ui.roiframe.enhancingROI_overlay_callback()
        self.ui.sliceviewerframe.tbar.home()
        self.ui.updateslice()
        self.ui.starttime()

    def loadCase(self,case=None,files=None):

        # reset and reinitialize
        self.ui.resetUI()
        if case is not None:
            self.casename.set(case)
            self.ui.set_casename()
        caseindex = self.caselist['casetags'].index(self.casename.get())
        # self.casedir = os.path.join(self.datadir.get(),self.config.UIdataroot+self.casename.get())
        if len(self.caselist['casedirs']):
            self.casedir = os.path.join(self.datadir.get(),self.caselist['casedirs'][caseindex])
        else:
            raise ValueError('No cases to load')

        # if three image files are given load them directly
        if files is not None:
            if len(files) != 3:
                self.ui.set_message('Select three image files')
                return
            t1ce_file,t2_file,flair_file = self.filenames
            dset = {'t1pre':{'d':None,'ex':False},'t1':{'d':None,'ex':False},'t2':{'d':None,'ex':False},'flair':{'d':None,'ex':False}}
            dset['t1']['d'],dset['t2']['d'],dset['flair']['d'],affine = self.loadData(t1ce_file,t2_file,flair_file)
            if all(['processed' in f for f in self.filenames]):
                self.processed = True
            else:
                dset = self.processNifti(dset)
            self.ui.affine['t1'] = affine

        # check for nifti image files with matching filenames
        # 'processed' refers to earlier output and is loaded preferentially.
        # for now assuming files are either all or none processed
        elif self.casetype <= 1:
            dset = {'t1pre':{'d':None,'ex':False},'t1':{'d':None,'ex':False},'t2':{'d':None,'ex':False},'flair':{'d':None,'ex':False}}
            files = os.listdir(self.casedir)
            # files = self.get_imagefiles(files)
            t1_files = [f for f in files if 't1' in f.lower()]
            if len(t1_files) > 0:
                if len(t1_files) > 1:
                    t1ce_file = next((f for f in t1_files if re.search('(processed)',f.lower())),None)
                    if t1ce_file is None:
                        t1ce_file = next((f for f in t1_files if re.search('(ce|gad|gd|post)',f.lower())),t1_files[0])
                elif len(t1_files) == 1:
                    t1ce_file = t1_files[0]
            flair_files = [f for f in files if 'flair' in f.lower()]
            if len(flair_files) > 0:
                if len(flair_files) > 1:
                    flair_file = next((f for f in flair_files if re.search('(processed)',f.lower())),None)
                    if flair_file is None:
                        flair_file = next((f for f in flair_files if re.search('(ce|gad|gd|post)',f.lower())),flair_files[0])
                elif len(flair_files) == 1:
                    flair_file = flair_files[0]
            t2_files = [f for f in files if 't2' in f.lower()]
            if len(t2_files) > 0:
                if len(t2_files) > 1:
                    t2_file = next((f for f in t2_files if re.search('(processed)',f.lower())),None)
                    if t2_file is None:
                        t2_file = next((f for f in t2_files if re.search('(ce|gad|gd|post)',f.lower())),t2_files[0])
                elif len(t2_files) == 1:
                    t2_file = t2_files[0]
            else:
                t2_file = None
            dset['t1']['d'],dset['t2']['d'],dset['flair']['d'],affine = self.loadData(t1ce_file,t2_file,flair_file)
            # assuming processed is all or none
            self.processed = True
            for f in [t1ce_file,t2_file,flair_file]:
                if f is not None:
                    if 'processed' not in f:
                        self.processed = False
            if self.processed is False:
                dset = self.processNifti(dset)
            self.ui.affine['t1'] = affine

        # dicom directories each containing several image series
        # for now it will assumed dicom is not multi-frame format
        else:
            dset = self.preprocess([self.casedir])

        # convenience test for presence of dataset
        for t in ['t1','t2','flair']:
            if dset[t]['d'] is not None:
                dset[t]['ex'] = True

        self.ui.sliceviewerframe.dim = np.shape(dset['t1']['d'])
        self.ui.sliceviewerframe.create_canvas()

        # 3 channels hard-coded. not currently loading t1 pre-contrast
        self.ui.data['raw'] = np.zeros((3,)+self.ui.sliceviewerframe.dim,dtype='float32')
        self.ui.data['raw'][0] = dset['t1']['d']
        self.ui.data['raw'][1] = dset['flair']['d']
        # if a t2 image is not available, fall back to using the t1 post image.
        if dset['t2']['ex']:
            self.ui.data['raw'][2] = dset['t2']['d']
        else:
            self.ui.data['raw'][2] = dset['t1']['d']


        if False:
            if img_arr_prob_GM is None:
                try:
                    d = nb.load(os.path.join(self.casedir,'brain_probabilities_GM.nii.gz'))
                    self.ui.data['probGM'] = np.transpose(np.array(d.dataobj),axes=(2,1,0))
                    d = nb.load(os.path.join(self.casedir,'brain_probabilities_WM.nii.gz'))
                    self.ui.data['probWM'] = np.transpose(np.array(d.dataobj),axes=(2,1,0))
                except FileNotFoundError as e:
                    self.ui.data['probGM'] = img_arr_prob_GM
                    self.ui.data['probWM'] = img_arr_prob_WM
            else:
                self.ui.data['probGM'] = img_arr_prob_GM
                self.ui.data['probWM'] = img_arr_prob_WM
            
        # save copy of the raw data
        self.ui.data['raw_copy'] = copy.deepcopy(self.ui.data['raw'])

        # automatically run normal stats if volume selected
        if self.ui.sliceviewerframe.slicevolume_norm.get() == 1:
            self.ui.sliceviewerframe.normalslice_callback()

        # create the label. 'seg' picks up the BraTS convention but may need to be more specific
        if self.casetype <= 1:
            seg_file = next((f for f in files if 'seg' in f),None)
            if seg_file is not None and 'blast' not in seg_file:
                label = sitk.ReadImage(os.path.join(self.casedir,seg_file))
                img_arr = sitk.GetArrayFromImage(label)
                self.ui.data['label'] = img_arr
            else:
                self.ui.data['label'] = None
        else:
            self.ui.data['label'] = None

        # supplementary labels. brats and nnunet conventions are differnt.
        if self.ui.data['label'] is not None:
            if False: # nnunet
                self.ui.data['manual_ET'] = (self.ui.data['label'] == 3).astype('int') #enhancing tumor 
                self.ui.data['manual_TC'] = (self.ui.data['label'] >= 2).astype('int') #tumour core
                self.ui.data['manual_WT'] = (self.ui.data['label'] >= 1).astype('int') #whole tumour
            else: # brats
                self.ui.data['manual_ET'] = (self.ui.data['label'] == 4).astype('int') #enhancing tumor 
                self.ui.data['manual_TC'] = ((self.ui.data['label'] == 1) | (self.ui.data['label'] == 4)).astype('int') #tumour core
                self.ui.data['manual_WT'] = (self.ui.data['label'] >= 1).astype('int') #whole tumour

    def loadData(self,t1_file,t2_file,flair_file,type=None):
        img_arr_t1 = None
        img_arr_t2 = None
        img_arr_flair = None
        if 'nii' in t1_file:
            try:
                img_nb_t1 = nb.load(os.path.join(self.casedir,t1_file))
                img_nb_flair = nb.load(os.path.join(self.casedir,flair_file))
            except IOError as e:
                self.ui.set_message('Can\'t import {} or {}'.format(t1_file,flair_file))
            self.ui.nb_header = img_nb_t1.header.copy()
            # nibabel convention will be transposed to sitk convention
            img_arr_t1 = np.transpose(np.array(img_nb_t1.dataobj),axes=(2,1,0))
            img_arr_flair = np.transpose(np.array(img_nb_flair.dataobj),axes=(2,1,0))
            affine = img_nb_t1.affine
            if t2_file is not None:
                try:
                    img_nb_t2 = nb.load(os.path.join(self.casedir,t2_file))
                except IOError as e:
                    self.ui.set_message('Can\'t import {} or {}'.format(t2_file))
                img_arr_t2 = np.transpose(np.array(img_nb_t2.dataobj),axes=(2,1,0))
            else:
                img_arr_t2 = None

        elif 'dcm' in t1_file: # not finished yet
            try:
                img_dcm_t1 = pd.dcmread(os.path.join(self.casedir,t1ce_file))
                img_dcm_t2 = pd.dcmread(os.path.join(self.casedir,t2flair_file))
            except IOError as e:
                self.ui.set_message('Can\'t import {} or {}'.format(t1ce_file,t2flair_file))
            self.ui.dcm_header = None
            img_arr_t1 = np.transpose(np.array(img_dcm_t1.dataobj),axes=(2,1,0))
            img_arr_t2 = np.transpose(np.array(img_dcm_t2.dataobj),axes=(2,1,0))
            affine = None
        return img_arr_t1,img_arr_t2,img_arr_flair,affine
    

    #################
    # process methods
    #################

    def processNifti(self,dset):
        if self.register_check_value:
            for t in ['t2','flair']:
                fixed_image = itk.GetImageFromArray(dset['t1']['d'])
                moving_image = itk.GetImageFromArray(dset[t]['d'])
                moving_image_reg = self.elastix_affine(fixed_image,moving_image)
                dset[t]['d'] = itk.GetArrayFromImage(moving_image_reg)
        if self.skullstrip_check_value:
            for t in ['t1','t2','flair']:
                dset[t]['d'] = self.extractbrain(dset[t]['d'])
        if self.n4_check_value:
            for t in ['t1','t2','flair']:
                dset[t]['d'] = self.n4bias(dset[t]['d'])
        return dset

    def multires_registration(self, fixed_image, moving_image):
        initial_transform = sitk.CenteredTransformInitializer(fixed_image, 
                                                            moving_image, 
                                                            sitk.Euler3DTransform(), 
                                                            sitk.CenteredTransformInitializerFilter.GEOMETRY)

        registration_method = sitk.ImageRegistrationMethod()
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)

        registration_method.SetInterpolator(sitk.sitkLinear)

        # Optimizer settings.
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1000, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup for the multi-resolution framework.            
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        # Don't optimize in-place, we would possibly like to run this cell multiple times.
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), 
                                                    sitk.Cast(moving_image, sitk.sitkFloat32))
        print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
        print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))        
        moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())         
        return moving_resampled

    def simple_elastix_affine(self, image, template):  
        elastixImageFilter = sitk.ElastixImageFilter()  
        elastixImageFilter.SetFixedImage(image)  
        elastixImageFilter.SetMovingImage(template)  
        elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("affine"))  
        elastixImageFilter.PrintParameterMap(sitk.GetDefaultParameterMap('affine'))
        elastixImageFilter.Execute()  
        sitk.WriteImage(elastixImageFilter.GetResultImage(), 'reg.tif')  
        # load image with SimpleITK  
        sitk_image = sitk.ReadImage('reg.tif')  
        # convert to NumPy array  
        registered_img = sitk.GetArrayFromImage(sitk_image)  
        # delete the tif file  
        os.remove('reg.tif')  
        return sitk_image
    
    def elastix_affine(self,image,template):
        parameter_object = itk.ParameterObject.New()
        default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('affine')
        parameter_object.AddParameterMap(default_rigid_parameter_map)        
        image_reg, params = itk.elastix_registration_method(image, template,parameter_object=parameter_object)
        return image_reg

    # create nb affine from dicom 
    def get_affine(self,metadata):
        dircos = np.array(list(map(float,metadata.ImageOrientationPatient)))
        affine = np.zeros((4,4))
        affine[:3,0] = dircos[0:3]*float(metadata.PixelSpacing[0])
        affine[:3,1] = dircos[3:]*float(metadata.PixelSpacing[1])
        d3 = np.cross(dircos[:3],dircos[3:])
        slthick = float(metadata.SliceThickness)
        # if hasattr(metadata,'SpacingBetweenSlices'):
        #     # a philips tag
        #     slthick += float(metadata.SpacingBetweenSlices)
        affine[:3,2] = d3*slthick
        affine[:3,3] = metadata.ImagePositionPatient
        affine[3,3] = 1
        # print(affine)
        return affine
    
    def extractbrain(self,img_arr):
        print('extract brain')
        img_arr = self.brainmage_clip(img_arr)
        self.ui.roiframe.WriteImage(img_arr,os.path.join(self.casedir,'img_temp.nii'),norm=False,type='float')

        tfile = 'img_temp.nii'
        ofile = 'img_brain.nii'
        if self.ui.OS == 'linux':
            command = 'conda run -n brainmage brain_mage_single_run '
            command += ' -i ' + os.path.join(self.casedir,tfile)
            command += ' -o ' + os.path.join('/tmp/foo')
            command += ' -m ' + os.path.join(self.casedir,ofile) + ' -dev 0'
            res = os.system(command)

        elif self.ui.OS == 'win32':
            # manually escaped for shell. can also use raw string as in r"{}".format(). or subprocess.list2cmdline()
            # some problem with windows, the scrip doesn't get on PATH after env activation, so still have to specify the fullpath here
            # it is currently hard-coded to anaconda3/envs location rather than .conda/envs, but anaconda3 could be installed
            # under either ProgramFiles or Users so check both
            if os.path.isfile(os.path.expanduser('~')+'\\anaconda3\Scripts\\activate.bat'):
                activatebatch = os.path.expanduser('~')+"\\anaconda3\Scripts\\activate.bat"
            elif os.path.isfile("C:\Program Files\\anaconda3\Scripts\\activate.bat"):
                activatebatch = "C:\Program Files\\anaconda3\Scripts\\activate.bat"
            else:
                raise FileNotFoundError('anaconda3/Scripts/activate.bat')
                                
            command1 = '\"'+activatebatch+'\" \"' + os.path.expanduser('~')+'\\anaconda3\envs\\brainmage\"'
            command2 = 'python \"' + os.path.join(os.path.expanduser('~'),'anaconda3','envs','brainmage','Scripts','brain_mage_single_run')
            command2 += '\" -i   \"' + os.path.join(self.casedir,tfile)
            command2 += '\"  -o  \"' + os.path.join(os.path.expanduser('~'),'AppData','Local','Temp','foo')
            command2 += '\"   -m   \"' + os.path.join(self.casedir,ofile) + '\"'
            cstr = 'cmd /c \" ' + command1 + "&" + command2 + '\"'
            if False:   
                info = subprocess.STARTUPINFO()
                info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                info.wShowWindow = subprocess.SW_HIDE
                res = subprocess.run(cstr,shell=True,startupinfo=info,creationflags=subprocess.CREATE_NO_WINDOW)
                # res = subprocess.run(cstr,shell=True)
            else:
                popen = subprocess.Popen(cstr,shell=True,stdout=subprocess.PIPE,universal_newlines=True)
                for stdout_line in iter(popen.stdout.readline,""):
                    if stdout_line != '\n':
                        print(stdout_line)
                popen.stdout.close()
                res = popen.wait()
                if res:
                    raise subprocess.CalledProcessError(res,cstr)
                    print(res)
        # print(res)

        img_nb = nb.load(os.path.join(self.casedir,'img_brain.nii'))
        img_arr = np.transpose(np.array(img_nb.dataobj),axes=(2,1,0))
        os.remove(os.path.join(self.casedir,'img_brain.nii'))
        os.remove(os.path.join(self.casedir,'img_temp.nii'))
        return img_arr

    # clip outliers as in brainmage code
    def brainmage_clip(self,img):
        img_temp = img[np.where(img>0)]
        img_temp = img[img >= img_temp.mean()]
        p1 = np.percentile(img_temp, 1)
        p2 = np.percentile(img_temp, 99)
        img[img > p2] = p2
        img = (img - p1) / p2
        return img.astype(np.float32)


    def segnormal(self,img_arr_t1,affine):
        print('segment normal tissue')
        self.ui.roiframe.WriteImage(img_arr_t1,os.path.join(self.casedir,'img_T1_brain.nii'),type='float',affine=affine)
        command = 'conda run -n deepmrseg deepmrseg_apply --task tissueseg '
        command += ' --inImg ' + os.path.join(self.casedir,'img_T1_brain.nii')
        command += ' --outImg ' + os.path.join(self.casedir,'img_T1_brain_seg.nii')
        command += ' --probs'
        res = os.system(command)
        # print(res)
        # rename and tidy up the probability outputs
        for t in [0,10,50]:
            os.remove(os.path.join(self.casedir,'img_T1_brain__probabilities_'+str(t)+'.nii.gz'))
        os.rename(os.path.join(self.casedir,'img_T1_brain__probabilities_150.nii.gz'),
                      os.path.join(self.casedir,'brain_probabilities_GM.nii.gz'))
        os.rename(os.path.join(self.casedir,'img_T1_brain__probabilities_250.nii.gz'),
                      os.path.join(self.casedir,'brain_probabilities_WM.nii.gz'))
        # img_nb_t1 = nb.load(os.path.join(self.casedir,'img_T1_brain_seg.nii'))
        # img_arr_t1 = np.transpose(np.array(img_nb_t1.dataobj),axes=(2,1,0))
        img_nb_prob_GM = nb.load(os.path.join(self.casedir,'brain_probabilities_GM.nii.gz'))
        img_arr_prob_GM = np.transpose(np.array(img_nb_prob_GM.dataobj),axes=(2,1,0))
        img_nb_prob_WM = nb.load(os.path.join(self.casedir,'brain_probabilities_WM.nii.gz'))
        img_arr_prob_WM = np.transpose(np.array(img_nb_prob_WM.dataobj),axes=(2,1,0))
        return img_arr_prob_GM,img_arr_prob_WM
                
    # if T2 matrix is different resample it to t1
    def resamplet2(self,img_arr_t1,img_arr_t2,a1,a2):
        img_t1 = nb.Nifti1Image(np.transpose(img_arr_t1,axes=(2,1,0)),affine=a1)
        img_t2 = nb.Nifti1Image(np.transpose(img_arr_t2,axes=(2,1,0)),affine=a2)
        img_t2_res = resample_from_to(img_t2,(img_t1.shape[:3],img_t1.affine))
        img_arr_t2 = np.transpose(np.array(img_t2_res.dataobj),axes=(2,1,0))
        return img_arr_t2,img_t2_res.affine
    
    # for RT images, they may have been zero-padded to 512 matrix before exporting to dicom
    # for now there will be no filtering as they are indicated to be zero-padded in the first place
    # simple factor of 2 for now to get started with
    def downsample(self,img_arr,affine):
        img_arr = img_arr[::2,::2,::2]
        affine[:3,:3] *= 2
        return img_arr,affine

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
    
    def n4(self,a1,a2,a3,shrinkFactor=4,nFittingLevels=4):
        # self.ui.set_message('Performing N4 bias correction')
        img_arr = np.stack((a1,a2,a3),axis=0)
        print('N4 bias correction')
        for ch in range(2):
            data = copy.deepcopy(img_arr[ch])
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
            img_arr[ch] = copy.deepcopy(corrected_img_arr)
        return img_arr[0],img_arr[1],img_arr[2]

    def n4bias(self,img_arr,shrinkFactor=4,nFittingLevels=4):
        print('N4 bias correction')
        data = copy.deepcopy(img_arr)
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
        img_arr = copy.deepcopy(corrected_img_arr)
        return img_arr


    # main callback for selecting a data directory either by file dialog or text entry
    # find the list of cases in the current directory, set the combobox, and optionally load a case
    def datadirentry_callback(self,event=None):
        dir = self.datadir.get().strip()
        if os.path.exists(dir):
            self.w.config(state='normal')            
            files = os.listdir(dir)
            casefiles = []
            if len(files):

                imagefiles = self.get_imagefiles(files)

                # single case directory with image files
                if len(imagefiles) > 1:
                    imagefiles = [i.group(1) for i in imagefiles]
                    self.casefile_prefix = ''
                    casefiles = [os.path.split(dir)[1]]
                    self.caselist['casetags'] = casefiles
                    self.caselist['casedirs'] = [os.path.split(dir)[1]]
                    self.ui.set_message('')
                    self.w.config(width=min(20,len(casefiles[0])))
                    self.datadir.set(os.path.split(dir)[0])
                    self.casetype = 0
                    doload = True

                # one or more case subdirectories
                else:
                    niftidirs,dcmdirs = self.get_imagedirs(dir)
                    # niftidirs option is intended for processing cases from a parent directory. such as BraTS.
                    # imagefiles and dcmdirs intended for processing at the level of the individual case directory.
                    if len(niftidirs):
                        self.datadir.set(dir)

                        # check for BraTS format first
                        brats = re.match('(^.*brats.*)0[0-9]{4}',niftidirs[0],flags=re.I)
                        if brats:
                            self.casefile_prefix = brats.group(1)
                            casefiles = [re.match('.*(0[0-9]{4})',f).group(1) for f in files if re.search('_0[0-9]{4}$',f)]
                            self.caselist['casetags'] = [re.match('.*(0[0-9]{4})',f).group(1) for f in files if re.search('_0[0-9]{4}$',f)]
                            self.caselist['casedirs'] = files
                            self.w.config(width=6)
                            # brats data already have this processing
                            self.register_check_value.set(0)
                            self.skullstrip_check_value.set(0)
                        else:
                            self.casefile_prefix = ''
                            # for niftidirs that are processed dicomdirs, there may be
                            # multiple empty subdirectories. for now assume that the 
                            # immediate subdir of the datadir is the best tag for the casefile
                            # if there are multiple niftidirs, and the selected datadir itself is the
                            # tag for a single nifti file
                            # the casedir is a sub-directory path between the upper datadir,
                            # and the parent of the dicom series dirs where the nifti's get stored
                            if len(niftidirs) > 1:
                                casefiles = [re.split(r'/|\\',d[len(self.datadir.get())+1:])[0] for d in niftidirs]
                                casedirs = [d[len(self.datadir.get())+1:] for d in niftidirs]
                                doload = self.config.AutoLoad
                            elif len(niftidirs) == 1:
                                self.casedir = niftidirs[0]
                                self.datadir.set(os.path.split(dir)[0])
                                casefiles = [re.split(r'/|\\',d[len(self.datadir.get())+1:])[0] for d in niftidirs]
                                casedirs = [d[len(self.datadir.get())+1:] for d in niftidirs]
                                doload = True
                            # may need a future sort
                            if False:
                                casefiles,casedirs = (list(t) for t in zip(*sorted(zip(casefiles,casedirs))))
                            self.caselist['casetags'] = casefiles
                            self.caselist['casedirs'] = casedirs
                            self.w.config(width=max(20,len(casefiles[0])))
                        self.casetype = 1

                    # assumes all nifti dirs or all dicom dirs.
                    # if only a single dicom case directory continue directly to blast
                    elif len(dcmdirs)==1:
                        # self.datadir.set(dir)
                        self.casefile_prefix = ''
                        self.casedir = dcmdirs[0]
                        self.datadir.set(os.path.split(dir)[0])
                        self.caselist['casetags'] = [re.split(r'/|\\',d[len(self.datadir.get())+1:])[0] for d in dcmdirs]
                        # self.caselist['casetags'] = [os.path.split(self.casedir)[1]]
                        self.caselist['casedirs'] = [d[len(self.datadir.get())+1:] for d in dcmdirs]
                        self.w.config(width=max(20,len(self.caselist['casetags'][0])))
                        self.casetype = 2
                        doload = True
                    elif len(dcmdirs) > 1:
                    # if multiple dicom dirs, preprocess only
                        self.datadir.set(dir)
                        self.preprocess(dcmdirs)
                        casefiles = []
                        doload = False
                        return

            if len(self.caselist['casetags']):
                self.config.UIdataroot = self.casefile_prefix
                # TODO: will need a better sort here
                self.caselist['casetags'] = sorted(self.caselist['casetags'])
                self.w['values'] = self.caselist['casetags']
                self.w.current(0)
                # current(0) should do this too, but sometimes it does not
                self.casename.set(self.caselist['casetags'][0])
                # autoload first case
                if doload:
                    self.case_callback()
            else:
                print('No cases found in directory {}'.format(dir))
                self.ui.set_message('No cases found in directory {}'.format(dir))
        else:
            print('Directory {} not found.'.format(dir))
            self.ui.set_message('Directory {} not found.'.format(dir))
            self.w.config(state='disable')
            self.datadirentry.update()
        return

    def get_imagefiles(self,files):
        imagefiles = [re.match('(^.*(t1|t2|flair).*\.(nii|nii\.gz|dcm)$)',f.lower()) for f in files]
        imagefiles = list(filter(lambda item: item is not None,imagefiles))
        if len(imagefiles):
            self.ui.set_message('')
        return imagefiles
    
    def get_imagedirs(self,dir):
        # dir = self.datadir.get()
        dcmdirs = []
        niftidirs = []
        for root,dirs,files in os.walk(dir,topdown=True):
            if len(files):
                dcmfiles = [f for f in files if re.match('.*\.dcm',f.lower())]
                niftifiles = [f for f in files if re.match('.*(t1|t2|flair).*\.(nii|nii\.gz)',f.lower())]
                if len(dcmfiles):
                    # for now assume that the parent of this dir is a series dir, and will take 
                    # the dicomdir as the parent of the series dir
                    # but for exported sunnybrook dicoms at least the more recognizeable dir might 
                    # be two levels above at the date.
                    dcmdirs.append(os.path.split(root)[0])
                if len(niftifiles):
                    niftidirs.append(os.path.join(root))
        if len(niftidirs+dcmdirs):
            self.ui.set_message('')
            # self.datadir.set(dir)
        # due to the intermediate seriesdirs, the above walk generates duplicates
        dcmdirs = list(set(dcmdirs))
        # for nifti dirs, need to set the casefiles for the pulldown at one of the more
        # recognizeable subdirs of the datadir. 

        return niftidirs,dcmdirs
    
    # load up to four dicom series directories in the provided parent directory
    def loaddicom(self,d):
        dset = {'t1pre':{'d':None,'ex':False},'t1':{'d':None,'ex':False},'t2':{'d':None,'ex':False},'flair':{'d':None,'ex':False}}
        # assume a dicomdir is a parent of several series directories
        self.casedir = d
        seriesdirs = os.listdir(d)
        for sd in seriesdirs:
            dpath = os.path.join(d,sd)
            files = sorted(os.listdir(dpath))
            metadata = pd.dcmread(os.path.join(dpath,files[0]))
            print(metadata.SeriesDescription)
            if 't1' in metadata.SeriesDescription.lower():
                if 'pre' in metadata.SeriesDescription.lower():
                    dset['t1pre']['ex'] = True
                    dset['t1pre']['d'] = np.zeros((len(files),metadata.Rows,metadata.Columns))
                    self.ui.affine['t1pre'] = self.get_affine(metadata)
                    dset['t1pre']['d'][0,:,:] = metadata.pixel_array
                    for i,f in enumerate(files[1:]):
                        data = pd.dcmread(os.path.join(dpath,f))
                        dset['t1pre']['d'][i+1,:,:] = data.pixel_array
                else:
                    dset['t1']['ex'] = True
                    dset['t1']['d'] = np.zeros((len(files),metadata.Rows,metadata.Columns))
                    self.ui.affine['t1'] = self.get_affine(metadata)
                    dset['t1']['d'][0,:,:] = metadata.pixel_array
                    for i,f in enumerate(files[1:]):
                        data = pd.dcmread(os.path.join(dpath,f))
                        dset['t1']['d'][i+1,:,:] = data.pixel_array
            elif any([f in metadata.SeriesDescription.lower() for f in ['flair','fluid']]):
                dset['flair']['ex'] = True
                dset['flair']['d'] = np.zeros((len(files),metadata.Rows,metadata.Columns))
                self.ui.affine['flair'] = self.get_affine(metadata)
                dset['flair']['d'][0,:,:] = metadata.pixel_array
                for i,f in enumerate(files[1:]):
                    data = pd.dcmread(os.path.join(dpath,f))
                    dset['flair']['d'][i+1,:,:] = data.pixel_array
            elif 't2' in metadata.SeriesDescription.lower():
                dset['t2']['ex'] = True
                dset['t2']['d'] = np.zeros((len(files),metadata.Rows,metadata.Columns))
                self.ui.affine['t2'] = self.get_affine(metadata)
                for i,f in enumerate(files[1:]):
                    data = pd.dcmread(os.path.join(dpath,f))
                    dset['t2']['d'][i+1,:,:] = data.pixel_array
        return dset
    
    # use for an input dicom directory
    # preprocessing nifti files is handled by the checkbox settings
    def preprocess(self,dirs):
        for d in dirs:

            dset = self.loaddicom(d)

            # downsample any matrix over 300
            if np.shape(dset['t1']['d'])[1] > 300:
                dset['t1']['d'],self.ui.affine['t1'] = self.downsample(dset['t1']['d'],self.ui.affine['t1'])

            # match T2,flair to T1 matrix
            if dset['t2']['ex']:
                if np.shape(dset['t1']['d']) != np.shape(dset['t2']['d']):
                    # self.ui.set_message('Image matrices do not match. Resampling T2flair into T1 space...')
                    print('Image matrices do not match. Resampling T2 into T1 space...')
                    dset['t2']['d'],self.ui.affine['t2'] = self.resamplet2(dset['t1']['d'],dset['t2']['d'],self.ui.affine['t1'],self.ui.affine['t2'])
                    dset['t2']['d']= np.clip(dset['t2']['d'],0,None)

            # assuming flair is always present in any possible dataset
            if np.shape(dset['t1']['d']) != np.shape(dset['flair']['d']):
                print('Image matrices do not match. Resampling flair into T1 space...')
                dset['flair']['d'],self.ui.affine['flair'] = self.resamplet2(dset['t1']['d'],dset['flair']['d'],self.ui.affine['t1'],self.ui.affine['flair'])
                dset['flair']['d'] = np.clip(dset['flair']['d'],0,None)

                if True:
                    for t in ['t1pre','t1','t2','flair']:
                        if dset[t]['ex']:
                            self.ui.roiframe.WriteImage(dset[t]['d'],os.path.join(d,'img_'+t+'_resampled.nii.gz'),
                                                type='float',affine=self.ui.affine['t1'])

            # registration
            print('register T2, flair')
            if True:
                for t in ['t1pre','t1','t2','flair']:
                    if dset[t]['ex']:
                        fname = os.path.join(d,'img_'+t+'_resampled.nii.gz')
                        img = nb.load(fname)
                        dset[t]['d'] = np.transpose(np.array(img.dataobj),axes=(2,1,0))
                        os.remove(fname)

            # reference
            fixed_image = itk.GetImageFromArray(dset['t1']['d'])

            for t in ['t1pre','t2','flair']:
                fname = os.path.join(d,'img_'+t+'_resampled.nii.gz')
                if dset[t]['ex']:
                    moving_image = itk.GetImageFromArray(dset[t]['d'])
                    moving_image_res = self.elastix_affine(fixed_image,moving_image)
                    dset[t]['d'] = itk.GetArrayFromImage(moving_image_res)
                    if False:
                        self.ui.roiframe.WriteImage(dset[t]['d'],os.path.join(d,'img_'+t+'_registered.nii.gz'),
                                                type='float',affine=self.ui.affine['t1'])

            # skull strip. for now assuming only needed on input dicoms
            for t in ['t1pre','t1','t2','flair']:
                if dset[t]['ex']:
                    dset[t]['d'] = self.extractbrain(dset[t]['d'])
                    if False:
                        self.ui.roiframe.WriteImage(dset[t]['d'],os.path.join(d,'img_'+t+'_extracted.nii.gz'),
                                                    type='float',affine=self.ui.affine['t1'])

            # bias correction.
            for t in ['t1pre','t1','t2','flair']:
                if dset[t]['ex']:   
                    dset[t]['d'] = self.n4bias(dset[t]['d'])

            # rescale the data
            # if necessary clip any negative values introduced by the processing
            for t in ['t1pre','t1','t2','flair']:
                if dset[t]['ex']:
                    if np.min(dset[t]['d']) < 0:
                        dset[t]['d'][dset[t]['d'] < 0] = 0
                    dset[t]['d'] = self.rescale(dset[t]['d'])

            # save nifti files for future use
            for t in ['t1pre','t1','t2','flair']:
                if dset[t]['ex']:
                    self.ui.roiframe.WriteImage(dset[t]['d'],os.path.join(d,'img_'+t+'_processed.nii.gz'),
                                                type='float',affine=self.ui.affine['t1'])

        if len(dirs) == 1:
            return dset
        return

    def resetCase(self):
        self.filenames = None
        self.casename = StringVar()
        self.casefile_prefix = None
        self.caselist['casetags'] = []
        self.caselist['casedirs'] = []
