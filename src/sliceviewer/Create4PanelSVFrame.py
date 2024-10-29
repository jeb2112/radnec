import os,sys
import numpy as np
import glob
import copy
import time
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
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice
import scipy

from src.NavigationBar import NavigationBar

from src.sliceviewer.CreateSVFrame import *

#####################################
# Slice Viewer for 4 panel image display
#####################################

class Create4PanelSVFrame(CreateSliceViewerFrame):
    def __init__(self,parentframe,ui=None,padding='10',style=None):
        super().__init__(parentframe,ui=ui,padding=padding,style=style)

        # ui variables
        self.currentslice = tk.IntVar(value=75)
        # imageaxes to channel mapping
        self.ax_img = {}
        self.ax2ch = {'A':('raw','t1+'),'B':('raw','flair'),'C':('raw','dwi'),'D':('adc','dwi')}
        self.labels = {'Im_A':None,'Im_B':None,'Im_C':None,'Im_D':None,'W_A':None,'L_A':None,'W_B':None,'L_B':None}
        self.lines = {'A':{'h':None,'v':None},'B':{'h':None,'v':None},'C':{'h':None,'v':None},'D':{'h':None,'v':None}}
        # values of the current linear measurement
        self.measurement = {'ax':None,'p0':None,'p1':None,'plot':None,'l':None,'ch':None}        
        # window/level stuff will need further tidying up
        # window/level values for T1,flair,dwi,adc
        self.window = np.array([1.,1.,1.,1.],dtype='float')
        self.level = np.array([0.5,0.5,0.5,0.5],dtype='float')
        # window/level values for overlays and images. hard-coded for now.
        self.wl = {('raw','t1+'):[600,300],('raw','flair'):[600,300],('raw','dwi'):[1000,500],('adc','dwi'):[2000,1000]}
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

        self.frame.grid(row=1, column=0, columnspan=2, in_=self.parentframe,sticky='NSEW')
        self.fstyle.configure('sliceviewerframe.TFrame',background='#000000')
        self.frame.configure(style='sliceviewerframe.TFrame')

        # override normal frame 
        if False:
            self.fstyle.configure('normal_frame.TFrame',background='red')
        self.normal_frame = ttk.Frame(self.parentframe,padding='0',style='normal_frame.TFrame')
        self.normal_frame.grid(row=3,column=0,sticky='news')

        self.create_blank_canvas()

        # dummy frame to hold canvas and slider bars
        self.fstyle.configure('canvasframe.TFrame',background='#000000')
        self.canvasframe = ttk.Frame(self.frame)
        self.canvasframe.configure(style='canvasframe.TFrame')
        self.canvasframe.grid(row=0,column=0,columnspan=3,sticky='ew')

        # messages text frame
        self.messagelabel = ttk.Label(self.normal_frame,text=self.ui.message.get(),padding='5',borderwidth=0)
        self.messagelabel.grid(row=2,column=0,columnspan=3,sticky='ew')

        if self.ui.OS in ('win32','darwin'):
            self.ui.root.bind('<MouseWheel>',self.mousewheel_win32)

        if self.ui.OS == 'linux':
            self.ui.root.bind('<Button-4>',self.mousewheel)
            self.ui.root.bind('<Button-5>',self.mousewheel)

        # adjust current root window to sliceviewer size
        if False:
            self.frame.update()
            self.resize()

    # resize canvas for dragging resizing of GUI window
    def resizer(self,event):
        if self.cw is None:
            return
        self.resizer_count *= -1
        # quick hack to improve the latency skip every other Configure event
        if self.resizer_count > 0:
            return
        # print(event)
        axfovratio = self.dim[2]/self.dim[1]
        self.hi = (event.height-self.ui.caseframe.frame.winfo_height()-self.ui.roiframe.frame.winfo_height())/self.ui.dpi * 1
        self.wi = self.hi * axfovratio
        if self.wi > event.width/self.ui.dpi:
            self.wi = (event.width-2*int(self.ui.mainframe_padding))/self.ui.dpi
            self.hi = self.wi / axfovratio
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
        w = self.ui.roiframe.frame.winfo_width() + self.normal_frame.winfo_width()
        h += max(self.ui.caseframe.frame.winfo_height(),self.ui.functionmenu.winfo_height()) + \
            max(self.normal_frame.winfo_height(),self.ui.roiframe.frame.winfo_height())
        h += 2*int(self.ui.mainframe_padding)
        # roiframe should govern width, but just in case this check
        w = max(w,self.ui.caseframe.frame.winfo_width()+self.ui.functionmenu.winfo_width())
        print('resize {},{}'.format(w,h))
        self.ui.root.geometry(f'{w}x{h}')
        return


    # place holder until a dataset is loaded
    # could create a dummy figure with toolbar, but the background colour during resizing was inconsistent at times
    # with just frame background style resizing behaviour seems correct.
    def create_blank_canvas(self):
        slicefovratio = self.config.ImageDim[0]/self.config.ImageDim[1]
        w = self.ui.current_panelsize*1 * self.ui.dpi
        h = self.ui.current_panelsize*1 * self.ui.dpi
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
            self.tbar.grid(column=0,row=0,columnspan=3,sticky='NW')
        self.frame.configure(width=w,height=h)
     
    # main canvas created when data are loaded
    def create_canvas(self,figsize=None):
        slicefovratio = self.dim[0]/self.dim[1]
        if figsize is None:
            figsize = (self.ui.current_panelsize*slicefovratio,self.ui.current_panelsize*1)
        if self.fig is not None:
            plt.close(self.fig)

        self.fig,self.axs = plt.subplot_mosaic([['A','B'],['C','D']],
                                    #  width_ratios=[self.ui.current_panelsize,self.ui.current_panelsize],
                                     height_ratios=[self.ui.current_panelsize,self.ui.current_panelsize],
                                     figsize=figsize,dpi=self.ui.dpi)
        for a in ['A','B','C','D']:
            self.ax_img[a] = self.axs[a].imshow(np.zeros((self.dim[1],self.dim[2])),vmin=0,vmax=1,cmap='gray',origin='lower',aspect=1)
            self.ax_img[a].format_cursor_data = self.make_cursordata_format()

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
        self.axs['labelA'] = self.fig.add_subplot(2,2,1)
        self.axs['labelB'] = self.fig.add_subplot(2,2,2)
        self.axs['labelC'] = self.fig.add_subplot(2,2,3)
        self.axs['labelD'] = self.fig.add_subplot(2,2,4)
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
        for ax in ['B','C','D']:
            self.axs[ax]._shared_axes['x'].join(self.axs[ax],self.axs['A'])
            self.axs[ax]._shared_axes['y'].join(self.axs[ax],self.axs['A'])
        self.fig.tight_layout(pad=0)
        self.fig.patch.set_facecolor('k')
        # record the data to figure coords of each label for each axis
        self.xyfig={}
        figtrans={}
        for a in ['A','B','C','D']:
            figtrans[a] = self.axs[a].transData + self.axs[a].transAxes.inverted()
        # these label coords are slightly hard-coded
        self.xyfig['Im_A']= figtrans['A'].transform((5,self.dim[1]-20))
        self.xyfig['W_A'] = figtrans['A'].transform((int(self.dim[2]/2),5))
        self.xyfig['L_A'] = figtrans['A'].transform((int(self.dim[2]*3/4),5))
        self.xyfig['W_B'] = figtrans['B'].transform((int(self.dim[2]/2),5))
        self.xyfig['L_B'] = figtrans['B'].transform((int(self.dim[2]*3/4),5))
        self.xyfig['W_C'] = figtrans['C'].transform((int(self.dim[2]/2),5))
        self.xyfig['L_C'] = figtrans['C'].transform((int(self.dim[2]*3/4),5))
        self.xyfig['W_D'] = figtrans['D'].transform((int(self.dim[2]/2),5))
        self.xyfig['L_D'] = figtrans['D'].transform((int(self.dim[2]*3/4),5))
        self.figtrans = figtrans

        # figure canvas
        newcanvas = FigureCanvasTkAgg(self.fig, master=self.canvasframe)  
        newcanvas.get_tk_widget().configure(bg='black')
        newcanvas.get_tk_widget().configure(width=figsize[0]*self.ui.dpi,height=figsize[1]*self.ui.dpi)
        newcanvas.get_tk_widget().grid(row=0, column=2, sticky='e')

        if False:
            self.tbar = NavigationBar(newcanvas,self.parentframe,pack_toolbar=False,ui=self.ui,axs=self.axs)
            self.tbar.grid(column=0,row=2,columnspan=3,sticky='NW')
        else:
            self.tbar = NavigationBar(newcanvas,self.normal_frame,pack_toolbar=False,ui=self.ui,axs=self.axs)
            # self.tbar.children['!button4'].pack_forget() # get rid of configure plot
            self.tbar.grid(column=0,row=0,columnspan=1,sticky='NW')

        if self.canvas is not None:
            self.cw.delete('all')
        self.canvas = newcanvas

        # slider bars
        if True:
            self.axsliceslider = ttk.Scale(self.canvasframe,from_=0,to=self.dim[0]-1,variable=self.currentslice,
                                        orient=tk.VERTICAL, length='3i',command=self.updateslice)
            self.axsliceslider.grid(column=0,row=0,sticky='w')

        # various bindings
        if self.ui.OS == 'linux':
            self.canvas.get_tk_widget().bind('<<MyMouseWheel>>',EventCallback(self.mousewheel,key='Key'))
        self.canvas.get_tk_widget().bind('<Up>',self.keyboard_slice)
        self.canvas.get_tk_widget().bind('<Down>',self.keyboard_slice)
        self.canvas.get_tk_widget().bind('<Enter>',self.focus)
        self.cw = self.canvas.get_tk_widget()

        self.frame.update()

    # main callback for any changes in the display
    def updateslice(self,event=None,wl=False):
        s = self.ui.s # local reference
        slice=self.currentslice.get()
        self.ui.roiframe.updatedwell()
        self.ui.set_currentslice()
        # set content of each axesImage
        for a in ['A','B','C','D']:
            self.ax_img[a].set(data=self.ui.data[s].dset[self.ax2ch[a][0]][self.ax2ch[a][1]]['d'][slice,:,:])
            self.ax_img[a].set(cmap='gray')
            self.updatewl2(a)
        # add current slice overlay
        self.update_labels()

        if wl:   
            if self.ui.dataselection == 'raw':
                self.clipwl_raw()

        self.canvas.draw()

        # if this is not a mouse event, show an existing measurement, or remove it
        # if self.ui.currentroi > 0 and event is None:
        if self.ui.currentroi > 0:
            self.update_measurements()
    
    def update_labels(self,item=None):
        for k in self.labels.keys():
            if self.labels[k] is not None:
                try:
                    Artist.remove(self.labels[k])
                except ValueError as e:
                    print(e)
        # convert data units to figure units
        self.labels['Im_A'] = self.axs['labelA'].text(self.xyfig['Im_A'][0],self.xyfig['Im_A'][1],'Im:'+str(self.currentslice.get()),color='w')
        for a in ['A','B','C','D']:
            ch = self.ax2ch[a]
            self.labels['W_'+a] = self.axs['label'+a].text(self.xyfig['W_'+a][0],self.xyfig['W_'+a][1],'W = '+'{:d}'.format(int(self.wl[ch][0])),color='w')
            self.labels['L_'+a] = self.axs['label'+a].text(self.xyfig['L_'+a][0],self.xyfig['L_'+a][1],'L = '+'{:d}'.format(int(self.wl[ch][1])),color='w')
        
    # original wl 
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
        elif ax==1:
            self.ax2_img.set_clim(vmin=vmin,vmax=vmax)
        elif ax == 2:
            self.ax3_img.set_clim(vmin=vmin,vmax=vmax)
        elif ax == 3:
            self.ax4_img.set_clim(vmin=vmin,vmax=vmax)

        self.canvas.draw()

    def ch2ax(self,ch):
        return self.ax2ch.keys()[self.ax2ch.values().index(ch)]

    # modified arrangement for wl
    def updatewl2(self,ax=None,lval=None,wval=None):

        if ax==None:
            return
        ch = self.ax2ch[ax]
        self.wlflag = True
        if lval:
            self.wl[ch][1] += lval
        if wval:
            self.wl[ch][0] += wval

        vmin = self.wl[ch][1] - self.wl[ch][0]/2
        vmax = self.wl[ch][1] + self.wl[ch][0]/2

        self.ax_img[ax].set_clim(vmin=vmin,vmax=vmax)

        self.canvas.draw()

    # clip the raw data to window and level settings
    def clipwl_raw(self):
        # for ax in range(2):
        for ax in ['t1+','flair']:
            vmin = self.wl[ax][1] - self.wl[ax][0]/2
            vmax = self.wl[ax][1] + self.wl[ax][0]/2
            self.ui.data[self.ui.s].dset[ax]['d'] = self.ui.caseframe.rescale(self.ui.data[self.ui.s].dset[ax]['d'],vmin=vmin,vmax=vmax)

    def restorewl_raw(self,dt):
        if False:
            self.ui.data[0].dset[dt]['d'] = copy.deepcopy(self.ui.data[0].dset[dt+'_copy']['d'])

    def fitlin(self,x,a,b):
        return a*x + b

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
        if event.y < 0 or event.x < 0:
            return
        # process only in two main panels
        if event.x <=self.ui.current_panelsize/2*self.ui.dpi and event.y <= self.ui.current_panelsize/2*self.ui.dpi:
            ax = 'A'
        elif event.x <= self.ui.current_panelsize*self.ui.dpi and event.y <= self.ui.current_panelsize/2*self.ui.dpi:
            ax = 'B'
        elif event.x <= self.ui.current_panelsize/2*self.ui.dpi and event.y <= self.ui.current_panelsize*self.ui.dpi:
            ax = 'C'
        elif event.x <= self.ui.current_panelsize*self.ui.dpi and event.y <= self.ui.current_panelsize*self.ui.dpi:
            ax = 'D'
        else:
            return
        if self.b1x is None:
            self.b1x,self.b1y = copy.copy((event.x,event.y))
            return
        if np.abs(event.x-self.b1x) > np.abs(event.y-self.b1y):
            if event.x-self.b1x > 0:
                self.updatewl2(ax=ax,wval=10)
            else:
                self.updatewl2(ax=ax,wval=-10)
        else:
            if event.y - self.b1y > 0:
                self.updatewl2(ax=ax,lval=10)
            else:
                self.updatewl2(ax=ax,lval=-10)

        self.b1x,self.b1y = copy.copy((event.x,event.y))
        self.update_labels()

        # repeating this here because there are some automatic tk backend events which 
        # can reset it during a sequence of multiple window/level drags
        self.canvas.get_tk_widget().config(cursor='circle')

    # from global screen pixel event coords, calculate the data coords within the clicked panel
    # because there is still a flip in the y coordinates between matplotlib and gui event,
    # for the bottom two panels C,D the screen pixel matrix dimension must first be subtracted, 
    # negation performed, then the data pixel matrix dimension added back on.
    def calc_panel_xy(self,ex,ey,ax):
        if ax > 'B':
            ey -= int(self.ui.current_panelsize*self.ui.config.dpi/2)
        x,y = self.axs[ax].transData.inverted().transform((ex,ey))
        y = -y
        if ax > 'B':
            y += self.dim[1]
        return x,y

    # mouse drag event linear measurement
    def b1motion_measure(self,event=None):
        self.canvas.get_tk_widget().config(cursor='sizing')
        # no adjustment from outside the pane
        if event.y < 0 or event.y > self.ui.current_panelsize*self.config.dpi:
            return
        # which artist axes was clicked
        a = self.tbar.select_artist(event)
        if a is None:
            return
        aax = a.axes._label
        if aax != self.measurement['ax']:
            self.clear_measurement_line()
            return
        # mouse event returns display coords but which are still flipped in y compared to matplotlib data coords.
        x,y = self.calc_panel_xy(event.x,event.y,aax)
        self.draw_measurement(x,y,aax)
        self.updateslice(event=event)

        # repeating this here because there are some automatic tk backend events which 
        # can reset it during a sequence of multiple drags
        self.canvas.get_tk_widget().config(cursor='sizing')

    # record coordinates of button click
    def b1click(self,event):
        ex,ey = np.copy(event.x),np.copy(event.y)
        self.canvas.get_tk_widget().config(cursor='sizing')
        if self.measurement['plot']:
            self.clear_measurement_line()
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
        self.measurement['ax'] = aax
        self.measurement['p0'] = (x,y)
        self.measurement['slice'] = self.ui.currentslice
        if False:
            self.axs[aax].plot(x,y,'+')
        self.canvas.get_tk_widget().config(cursor='sizing')
        # bind the release event
        self.ui.root.bind('<ButtonRelease-1>',self.b1release)

    # record measurement as ROI after left-button release
    def b1release(self,event):
        self.record_measurement()
        self.ui.root.unbind('<ButtonRelease-1>')

    # draw line for current linear measurement
    def draw_measurement(self,x,y,ax):
        if self.measurement['ax'] != ax or self.measurement['p0'] is None:
            return
        if self.measurement['plot'] is not None:
            try:
                self.axs[self.measurement['ax']].lines[0].remove() # coded for only 1 line
                self.measurement['plot'] = None
            except ValueError as e:
                print(e)
        lx = np.array([self.measurement['p0'][0],x])
        ly = np.array([self.measurement['p0'][1],y])
        self.measurement['plot'] = self.axs[ax].plot(lx,ly,'b',clip_on=True)[0]
        self.measurement['p1'] = (x,y)
        self.measurement['l'] = np.sqrt(np.power(lx[1]-lx[0],2)+np.power(ly[1]-ly[0],2))
        self.ui.set_message(msg='distance = {:.1f}'.format(self.measurement['l']))
        return

    # remove existing measurement line, for using during interactive draw only
    def clear_measurement_line(self):
        if self.measurement['plot'] is not None:
            self.axs[self.measurement['ax']].lines[0].remove() # coded for only 1 line
        self.measurement = {'ax':None,'p0':None,'p0':None,'plot':None,'l':None,'slice':None}
        self.ui.clear_message()
        self.canvas.draw()

    # remove an existing roi line, for use after interactive draw
    def clear_line(self,roi):
        for l in self.axs[roi['ax']].lines:
            l.remove()
        roi['plot'] = None

    # copy existing measurement to list of roi's.
    def record_measurement(self):
        self.ui.roi[self.ui.s].append(copy.deepcopy(self.measurement))
        self.ui.roiframe.currentroi.set(self.ui.roiframe.currentroi.get() + 1)
        self.ui.roiframe.update_roinumber_options()
        self.measurement = {'ax':None,'p0':None,'p0':None,'plot':None,'l':None,'slice':None}

    # re-display an existing measurement
    def show_measurement(self,roi=None):
        if roi is None:     
            r = self.ui.roi[self.ui.s][self.ui.currentroi]
        else:
            r = self.ui.roi[self.ui.s][roi]
        lx = np.array([r['p0'][0],r['p1'][0]])
        ly = np.array([r['p0'][1],r['p1'][1]])
        self.ui.set_currentslice(r['slice'])
        assert(r['plot'] is None)
        r['plot'] = self.axs[r['ax']].plot(lx,ly,'b',clip_on=True)[0] 
        self.ui.set_message(msg='distance = {:.1f}'.format(r['l']))
        self.canvas.draw()

    # display or remove measuremnt in the current slice
    def update_measurements(self):
        slice = np.copy(self.ui.currentslice)
        itoshow = 0
        for i,r in enumerate(self.ui.roi[self.ui.s][1:]):
            if r['slice'] == slice:
                itoshow = i + 1 # record index for display
            elif r['plot'] is not None:
                # unsolved bug in this function possibly due to slice scrolling too
                # fast too keep up, is sometimes not removing a line properly
                # possibly in the slider bar as opposed to mousewheel
                # so assuming only 1 line per slice and remove all lines to be sure
                self.clear_line(r)
                self.ui.clear_message()
                self.canvas.draw()
        if itoshow: # display last so distance value can be shown as message
            self.show_measurement(roi=itoshow)
