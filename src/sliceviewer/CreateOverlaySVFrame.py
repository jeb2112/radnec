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
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice

from src.NavigationBar import NavigationBar
from src.FileDialog import FileDialog

from src.sliceviewer.CreateSVFrame import *


###########################
# Slice Viewer for overlays
###########################

class CreateOverlaySVFrame(CreateSliceViewerFrame):
    def __init__(self,parentframe,ui=None,padding='10',style=None):
        super().__init__(parentframe,ui=ui,padding=padding,style=style)

        # ui variables
        self.axslicelabel = None
        self.windowlabel = None
        self.levellabel = None
        self.maskdisplay = tk.StringVar(value='unet')

        # flythrough scroll
        slice0_label = ttk.Label(self.normal_frame,text='start: ')
        slice0_label.grid(row=2,column=0,sticky='e')
        self.slice0_entry = ttk.Entry(self.normal_frame,width=8,textvariable=self.scrollslice0)
        self.slice0_entry.grid(row=2,column=1,sticky='w')
        slice1_label = ttk.Label(self.normal_frame,text='end: ')
        slice1_label.grid(row=2,column=2,sticky='e')
        self.slice1_entry = ttk.Entry(self.normal_frame,width=8,textvariable=self.scrollslice1)
        self.slice1_entry.grid(row=2,column=3,sticky='w')
        scroll_button = ttk.Button(self.normal_frame,text='Scroll',
                                               command=Command(self.scroll_callback))
        scroll_button.grid(row=2,column=4,sticky='w')
        record_label = ttk.Label(self.normal_frame,text='record: ')
        record_label.grid(row=2,column=5,sticky='e')
        record_button = ttk.Checkbutton(self.normal_frame,text='',
                                               variable=self.record_scroll)
        record_button.grid(row=2,column=6,sticky='w')        

        self.create_blank_canvas()

        # dummy frame to hold canvas and slider bars
        self.fstyle.configure('canvasframe.TFrame',background='#000000')
        self.canvasframe = ttk.Frame(self.frame)
        self.canvasframe.configure(style='canvasframe.TFrame')
        self.canvasframe.grid(row=1,column=0,columnspan=3,sticky='NW')

        # BLAST/UNet segmentation selection
        if False:
            maskdisplay_label = ttk.Label(self.normal_frame, text='segmentation: ')
            maskdisplay_label.grid(row=1,column=0,padx=(50,0),sticky='e')
            self.maskdisplay_button = {}
            self.maskdisplay_button['unet'] = ttk.Radiobutton(self.normal_frame,text='UNet',variable=self.maskdisplay,value='unet',
                                                        command=self.updatemask)
            self.maskdisplay_button['unet'].grid(column=1,row=1,sticky='w')
            self.maskdisplay_button['blast'] = ttk.Radiobutton(self.normal_frame,text='BLAST',variable=self.maskdisplay,value='blast',
                                                        command=self.updatemask,state='disabled')
            self.maskdisplay_button['blast'].grid(column=2,row=1,sticky='w')

        # overlay type contour mask. no longer used
        if False:
            overlaytype_label = ttk.Label(self.normal_frame, text='overlay type: ')
            overlaytype_label.grid(row=1,column=0,padx=(50,0),sticky='e')
            self.overlaytype_button = ttk.Radiobutton(self.normal_frame,text='z-score',variable=self.overlaytype,value=0,
                                                        command=Command(self.updateslice,wl=True))
            self.overlaytype_button.grid(row=1,column=1,sticky='w')
            self.overlaytype_button = ttk.Radiobutton(self.normal_frame,text='CBV',variable=self.overlaytype,value=1,
                                                        command=Command(self.updateslice,wl=True))
            self.overlaytype_button.grid(row=1,column=2,sticky='w')

        # messages text frame
        # self.messagelabel = ttk.Label(self.normal_frame,text=self.ui.message.get(),padding='5',borderwidth=0)
        # self.messagelabel.grid(row=2,column=0,columnspan=3,sticky='ew')

        self.bindings(action=True)

    def __del__(self):
        self.bindings(action=False)
     

    # main canvas created when data are loaded
    def create_canvas(self,figsize=None):
        slicefovratio = self.dim[0]/self.dim[1]
        if figsize is None:
            figsize = (self.ui.current_panelsize*(2),self.ui.current_panelsize)
        if self.fig is not None:
            plt.close(self.fig)

        self.fig,self.axs = plt.subplot_mosaic([['A','B'],['A','B']],
                                     width_ratios=[self.ui.current_panelsize,self.ui.current_panelsize],
                                     figsize=figsize,dpi=self.ui.dpi)
        self.ax_img = self.axs['A'].imshow(np.zeros((self.dim[1],self.dim[2])),vmin=0,vmax=1,cmap='gray',origin='upper',aspect=1)
        self.ax2_img = self.axs['B'].imshow(np.zeros((self.dim[1],self.dim[2])),vmin=0,vmax=1,cmap='gray',origin='upper',aspect=1)
        self.ax_img.format_cursor_data = self.make_cursordata_format()
        self.ax2_img.format_cursor_data = self.make_cursordata_format()


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
        self.axs['labelA'] = self.fig.add_subplot(1,2,1)
        self.axs['labelB'] = self.fig.add_subplot(1,2,2)
        for a in ['A','B']:
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

        # transform for absolute coords. has to be after tight_layout.
        figtrans={}
        for a in ['A','B']:
            figtrans[a] = self.axs[a].transData + self.axs[a].transAxes.inverted()
        self.xyfig={}

        # position for dummy axis for colorbars. also after tight layout since it's interior
        if True:
            l,b,w,h = self.axs['A'].get_position().bounds
            self.xyfig['colorbar_A'] = np.array([l,b+.25])

        # record the data to figure coords of each label for each axis
        self.xyfig['Im_A']= figtrans['A'].transform((5,25))
        self.xyfig['W_A'] = figtrans['A'].transform((int(self.dim[1]/2),self.dim[1]-15))
        self.xyfig['L_A'] = figtrans['A'].transform((int(self.dim[1]*3/4),self.dim[1]-15))
        self.xyfig['W_B'] = figtrans['B'].transform((int(self.dim[1]/2),self.dim[1]-15))
        self.xyfig['L_B'] = figtrans['B'].transform((int(self.dim[1]*3/4),self.dim[1]-15))
        
        self.xyfig['date_A'] = figtrans['A'].transform((5,15))
        self.xyfig['date_B'] = figtrans['B'].transform((5,15))
        self.figtrans = figtrans

        # figure canvas
        newcanvas = FigureCanvasTkAgg(self.fig, master=self.canvasframe)  
        newcanvas.get_tk_widget().configure(bg='black')
        newcanvas.get_tk_widget().configure(width=figsize[0]*self.ui.dpi,height=figsize[1]*self.ui.dpi)
        newcanvas.get_tk_widget().grid(row=0, column=0, sticky='')

        self.tbar = NavigationBar(newcanvas,self.normal_frame,pack_toolbar=False,ui=self.ui,axs=self.axs)
        self.tbar.grid(column=0,row=0,columnspan=3,sticky='NW')

        if self.canvas is not None:
            self.cw.delete('all')
        self.canvas = newcanvas

        # slider bars
        if False:
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
    def updateslice(self,event=None,wl=False,blast=False,layer=None,update=False):
        slice=self.currentslice.get()
        self.ui.set_currentslice()
        if 'overlay' in self.ui.dataselection:
            # if at least one overlay is existing, don't recalculate. hard-coded two studies
            if not (self.ui.data[0].dset[self.ui.dataselection][self.ui.chselection]['ex'] or \
                    self.ui.data[1].dset[self.ui.dataselection][self.ui.chselection]['ex']):
                # recalculate for new base image
                self.ui.roiframe.overlay_callback(updateslice=False)
        else: 
            self.ui.chselection = self.chdisplay.get()
        # update the image data
        self.ax_img.set(data=self.ui.data[self.ui.timepoints[0]].dset[self.ui.dataselection][self.ui.chselection]['d'][slice])
        self.ax2_img.set(data=self.ui.data[self.ui.timepoints[1]].dset[self.ui.dataselection][self.ui.chselection]['d'][slice])
        # add current slice overlay
        self.update_labels(colorbar=('overlay' in self.ui.dataselection and 'radnec' not in self.ui.dataselection),show=self.anno_label.get())

        if 'overlay' in self.ui.dataselection:
            # need to check in case overlay only available for one study
            if self.ui.data[self.ui.timepoints[0]].dset[self.ui.dataselection][self.ui.chselection]['ex']:  
                if 'tempo' in self.ui.dataselection: # better an attribute for cmap
                    self.ax_img.set(cmap='tempo')
                else: 
                    self.ax_img.set(cmap='viridis')
            else:
                self.ax_img.set(cmap='gray')
            if self.ui.data[self.ui.timepoints[1]].dset[self.ui.dataselection][self.ui.chselection]['ex']:   
                if 'tempo' in self.ui.dataselection:
                    self.ax2_img.set(cmap='tempo')
                else:
                    self.ax2_img.set(cmap='viridis')
            else:
                self.ax2_img.set(cmap='gray')
        else:
            self.ax_img.set(cmap='gray')
            self.updatewl(ax=0)
            self.ax2_img.set(cmap='gray')
            self.updatewl(ax=1)
        if wl:   # not sure if needed
            # possible latency problem here
            if self.ui.dataselection == 'overlay':
                # self.ui.roiframe.layer_callback(updateslice=False,updatedata=False,layer=layer)
                self.ui.roiframe.layer_callback()
            elif self.ui.dataselection == 'raw':
                self.clipwl_raw()

        self.canvas.draw()
        if update:
            self.canvas.draw_idle()
            self.canvas.flush_events()
    
    def updatemask(self):

        mask = self.maskdisplay.get()
        for s in self.ui.data.keys():
            self.ui.data[s].mask['ET']['d'] = self.ui.data[s].mask[mask]['ET']['d']
            self.ui.data[s].mask['WT']['d'] = self.ui.data[s].mask[mask]['WT']['d']
            for ovly in ['z','tempo','cbv']:
                for ch in list(self.ui.data[s].channels.values()):
                    self.ui.data[s].dset[ovly+'overlay'][ch]['ex'] = False
        self.ui.roiframe.overlay_callback(redo=True)


    def update_labels(self,colorbar=False,show=True):

        # handle colorbar separately, since it doesn't have an Artist.remove()
        if 'colorbar_A' in self.labels.keys():
            if self.labels['colorbar_A'] is not None:
                self.labels['colorbar_A'].remove()
                self.labels['colorbar_A'] = None
                if False:
                    try:
                        plt.delaxes(ax=self.axs['colorbar_A'])
                        # self.axs['colorbar_A'].remove()
                    except KeyError:
                        a=1

        for k in self.labels.keys():
            if self.labels[k] is not None:
                try:
                    Artist.remove(self.labels[k])
                except AttributeError as e:
                    print(e)
                except ValueError as e:
                    print(e)
        # convert data units to figure units
        if show:
            self.labels['Im_A'] = self.axs['labelA'].text(self.xyfig['Im_A'][0],0+self.xyfig['Im_A'][1],'Im:'+str(self.currentslice.get()),color='w')
            self.labels['W_A'] = self.axs['labelA'].text(self.xyfig['W_A'][0],self.xyfig['W_A'][1],'W = '+'{:d}'.format(int(self.window[0])),color='w')
            self.labels['L_A'] = self.axs['labelA'].text(self.xyfig['L_A'][0],self.xyfig['L_A'][1],'L = '+'{:d}'.format(int(self.level[0])),color='w')
            self.labels['W_B'] = self.axs['labelB'].text(self.xyfig['W_B'][0],self.xyfig['W_B'][1],'W = '+'{:d}'.format(int(self.window[1])),color='w')
            self.labels['L_B'] = self.axs['labelB'].text(self.xyfig['L_B'][0],self.xyfig['L_B'][1],'L = '+'{:d}'.format(int(self.level[1])),color='w')
            self.labels['date_A'] = self.axs['labelA'].text(self.xyfig['date_A'][0],self.xyfig['date_A'][1],self.ui.data[self.ui.timepoints[0]].date,color='w')
            self.labels['date_B'] = self.axs['labelB'].text(self.xyfig['date_B'][0],self.xyfig['date_B'][1],self.ui.data[self.ui.timepoints[1]].date,color='w')

        # add colorbars. for now just one colorbar on axis 'A'
        if colorbar and True:
            self.axs['colorbar_A'] = self.fig.add_axes([self.xyfig['colorbar_A'][0],self.xyfig['colorbar_A'][1],.02,0.5])
            ovly = self.ui.roiframe.overlay_type.get()

            # special case for tempo. not sure how to code yet.
            if ovly == 'tempo':
                ytick0 = int(self.wl[ovly][1]-self.wl[ovly][0]/2)+0
                ytick1 = int(self.wl[ovly][1]+self.wl[ovly][0]/2)+0
            else:
                ytick0 = int(self.wl[ovly][1]-self.wl[ovly][0]/2)
                ytick1 = int(self.wl[ovly][1]+self.wl[ovly][0]/2)
            ntick = 4
            ytickinc = np.round(np.power(10,np.round(np.log10(ytick1-ytick0)))/ntick)
            if ytickinc == 0:
                ytickinc = 1
            yticks = np.arange(ytick0,ytick1+ytickinc,ytickinc)
            self.labels['colorbar_A'] = self.fig.colorbar(self.ax_img,cax=self.axs['colorbar_A'],ticks=yticks)
            self.axs['colorbar_A'].yaxis.set_ticks_position('right')
            self.axs['colorbar_A'].yaxis.set_label_position('right')
            self.axs['colorbar_A'].yaxis.set_tick_params(color='w')
            self.labels['colorbar_A'].outline.set_edgecolor('w')
            # problems with updating the colorbar after set_data()
            # this didn't do anything
            if False:
                self.labels['colorbar_A'].update_normal()
            # although colorbar is not called until the axesImage data are set_data'd to become the z-score values,
            # the axesImage retains the clim equal to the original gray scale values, and this is passed on to the colorbar
            # object for setting ticks and labels. however, the display of the new
            # set_data is not in accordance with these now fictitious clim values, ticks, and labels, but is correct and is according to 
            # clim values ticks and labels that don't yet exist. In order to get these
            # correct clim values into existence, have to separately call set_clim on the axesImage scalar
            # mappable. Yet this does not then change the display of the scalar mappable in the slightest, which was correct
            # and remains correct. it only changes the ticks and labels of the colorbar.

            # in addition, a further temporary arrangement for tempo until coded properly. the tempo nifti mask file
            # is uint8 with background/zero == 2. have to reset to remove that offset of 2 here, or elsewhere.
            # -0 means it is being removed elsewhere
            if ovly == 'tempo':
                self.ax_img.set_clim((self.wl[ovly][1]-self.wl[ovly][0]/2-0,self.wl[ovly][1]+self.wl[ovly][0]/2-0))
            else:
                self.ax_img.set_clim((self.wl[ovly][1]-self.wl[ovly][0]/2,self.wl[ovly][1]+self.wl[ovly][0]/2))

            plt.setp(plt.getp(self.labels['colorbar_A'].ax.axes,'yticklabels'),color='w')
            

    # resize root window to match current sliceviewer
    # needs updating
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

    def get_canvas_coords(self,canvas=None):
        if canvas is None:
            canvas = self.canvas

        x = canvas.get_tk_widget().winfo_x()  # X position relative to screen
        y = canvas.get_tk_widget().winfo_y()  # Y position relative to screen
        width = canvas.get_tk_widget().winfo_width()  # Width of the canvas
        height = canvas.get_tk_widget().winfo_height()  # Height of the canvas

        xf = self.ui.sliceviewerframe.frame.winfo_x()
        yf = self.ui.sliceviewerframe.frame.winfo_y()

        xr = self.ui.root.winfo_rootx()
        yr = self.ui.root.winfo_rooty()
 
        return (x+xr+xf,y+yr+yf,width,height)

    # this function populates the per-axis window/level lists from the 
    # currently loaded 'raw' data, based on the current channel selection.
    # ie they govern the display of the grayscale 'raw' data.

    # note for now the overlay mode shows images of the same type in both windows.
    # therefore, this arrangement of having separate window/level per axis
    # is redundant, because if the type of image is the same in both axes,
    # the window/level should be the same. this arrangement was originally 
    # assigned from the original BLAST viewer showing T1/Flair in the two main axes.
    # as a result, here the time point 0 values are used for both axes

    # a separate dict{} wl is also populated. this dict is currently being used
    # for the optional colorbar in this viewer when an overlay is displayed. this 
    # code has not been tested lately.

    # these two separate conventions for 
    # window and level are not fully reconciled and probably need simplification.
    def setwl(self):
        self.level = []
        self.window = []
        # for w in ['A','B']:
        ch = self.ui.chselection
        for s in range(2): # hard-coded for only 2 studies 
            if self.ui.data[s].dset['raw'][ch]['ex']:
                self.level.append(self.ui.data[0].dset['raw'][ch]['l'])
                self.window.append(self.ui.data[0].dset['raw'][ch]['w'])
            else:
                raise ValueError('No data for channel {}'.format(ch))
        # arbitrarily using timepoint0 here
        self.wl[ch] = [self.ui.data[0].dset['raw'][ch]['w'],self.ui.data[0].dset['raw'][ch]['l']]
        self.level = np.array(self.level)
        self.window = np.array(self.window)
        return


    # in this viewer, the window/level of either axis tracks the other
    def updatewl(self,ax=0,lval=None,wval=None):

        self.wlflag = True
        # only process on main panels
        if ax < 2:
            if lval:
                self.level = [l + lval for l in self.level]
            if wval:
                self.window = [w + wval for w in self.window]

            vmin = self.level[ax] - self.window[ax]/2
            vmax = self.level[ax] + self.window[ax]/2

            self.ax_img.set_clim(vmin=vmin,vmax=vmax)
            self.ax2_img.set_clim(vmin=vmin,vmax=vmax)

            self.canvas.draw()