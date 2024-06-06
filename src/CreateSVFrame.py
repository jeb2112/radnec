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

from src.CreateFrame import *


##############
# Slice Viewer. 
##############

# a base class with a few general methods which BLAST and overlay mode slice viewers inherit
# still some redudancy between the three sliceviewer classes, some tidyup needed

class CreateSliceViewerFrame(CreateFrame):
    def __init__(self,parentframe,ui=None,padding='10',style=None):
        super().__init__(parentframe,ui=ui,padding=padding,style=style)

        # ui variables
        self.currentslice = tk.IntVar(value=75)
        self.currentsagslice = tk.IntVar(value=120)
        self.currentcorslice = tk.IntVar(value=120)
        self.labels = {'Im_A':None,'Im_B':None,'Im_C':None,'W_A':None,'L_A':None,'W_B':None,'L_B':None}
        self.lines = {k:{'h':None,'v':None} for k in ['A','B','C','D']}
        self.measurement = {'ax':None,'x0':None,'y0':None,'plot':None,'l':None}
        self.chdisplay = tk.StringVar(value='t1+')
        # self.overlaytype = tk.IntVar(value=self.config.OverlayType)
        self.slicevolume_norm = tk.IntVar(value=1)
        # blast window/level values for T1,T2. replace with self.wl
        self.window = np.array([1.,1.],dtype='float')
        self.level = np.array([0.5,0.5],dtype='float')
        # window/level values for overlays and images. hard-coded for now.
        # RELCCBV raw units off scanner are [0,4095]
        self.wl = {'t1':[600,300],'flair':[600,300],'z':[12,6],'cbv':[2047,1023],'tempo':[2,2]}
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

        # misc frame for various functions
        self.normal_frame = ttk.Frame(self.parentframe,padding='0')
        self.normal_frame.grid(row=3,column=0,sticky='NW')

        self.create_blank_canvas()

        # dummy frame to hold canvas and slider bars
        self.fstyle.configure('canvasframe.TFrame',background='#000000')
        self.canvasframe = ttk.Frame(self.frame)
        self.canvasframe.configure(style='canvasframe.TFrame')
        self.canvasframe.grid(row=1,column=0,columnspan=3,sticky='NW')

        # dummy frame to hide base image selection
        self.dummy_frame = ttk.Frame(self.parentframe,padding='0')
        self.dummy_frame.grid(row=3,column=0,sticky='news')

        # t1/t2 base layer selection
        chdisplay_label = ttk.Label(self.normal_frame, text='base image: ')
        chdisplay_label.grid(row=0,column=0,padx=(50,0),sticky='e')
        self.chdisplay_button = {}
        self.chdisplay_button['t1'] = ttk.Radiobutton(self.normal_frame,text='T1',variable=self.chdisplay,value='t1',
                                                    command=self.updateslice)
        self.chdisplay_button['t1'].grid(column=1,row=0,sticky='w')
        self.chdisplay_button['t1+'] = ttk.Radiobutton(self.normal_frame,text='T1+',variable=self.chdisplay,value='t1+',
                                                    command=self.updateslice)
        self.chdisplay_button['t1+'].grid(column=2,row=0,sticky='w')
        self.chdisplay_button['t2'] = ttk.Radiobutton(self.normal_frame,text='T2',variable=self.chdisplay,value='t2',
                                                    command=self.updateslice)
        self.chdisplay_button['t2'].grid(column=3,row=0,sticky='w')
        self.chdisplay_button['flair'] = ttk.Radiobutton(self.normal_frame,text='FLAIR',variable=self.chdisplay,value='flair',
                                                    command=self.updateslice)
        self.chdisplay_button['flair'].grid(column=4,row=0,sticky='w')
        self.chdisplay_button['dwi'] = ttk.Radiobutton(self.normal_frame,text='DWI',variable=self.chdisplay,value='dwi',
                                                    command=self.updateslice)
        self.chdisplay_button['dwi'].grid(column=5,row=0,sticky='w')
        # self.chdisplay_keys = ['t1','t1+','flair','flair']

        # overlay type contour mask
        if False:
            overlay_type_label = ttk.Label(self.normal_frame, text='overlay type: ')
            overlay_type_label.grid(row=1,column=0,padx=(50,0),sticky='e')
            self.overlay_type_button = ttk.Radiobutton(self.normal_frame,text='z-score',variable=self.overlay_type,value=0,
                                                        command=Command(self.updateslice,wl=True))
            self.overlay_type_button.grid(row=1,column=1,sticky='w')
            self.overlay_type_button = ttk.Radiobutton(self.normal_frame,text='CBV',variable=self.overlay_type,value=1,
                                                        command=Command(self.updateslice,wl=True))
            self.overlay_type_button.grid(row=1,column=2,sticky='w')

        # messages text box
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
        self.wi = self.hi*2 + 0 * self.hi / (2*slicefovratio)
        if self.wi > event.width/self.ui.dpi:
            self.wi = (event.width-2*int(self.ui.mainframe_padding))/self.ui.dpi
            self.hi = self.wi/(2+0/(2*slicefovratio))
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
        w = self.ui.current_panelsize*(2 + 0/(2*slicefovratio)) * self.ui.dpi
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
            self.tbar.grid(column=0,row=0,columnspan=3,sticky='NW')
        self.frame.configure(width=w,height=h)
     
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

        self.canvas.draw()


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

    # mouse drag event linear measurement
    # only tested in 4panel viewer
    def b1motion_measure(self,event):
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
            self.clear_measurement()
            return
        # calculate data coords for all axes relative to clicked axes
        # mouse event returns display coords but which are still flipped in y compared to matplotlib display coords.
        if True:
            x,y = self.axs[aax].transData.inverted().transform((event.x,event.y))
            y = -y
        else:
            x,y = event.x,event.y
        self.draw_measurement('A',x,y)

        self.updateslice()

        # repeating this here because there are some automatic tk backend events which 
        # can reset it during a sequence of multiple drags
        self.canvas.get_tk_widget().config(cursor='sizing')

    # record coordinates of button click
    def b1click(self,event):
        self.canvas.get_tk_widget().config(cursor='sizing')
        if self.measurement['plot']:
            self.clear_measurement()
        # no adjustment from outside the pane
        if event.y < 0 or event.y > self.ui.current_panelsize*self.config.dpi:
            return
        # which artist axes was clicked
        a = self.tbar.select_artist(event)
        if a is None:
            return
        aax = a.axes._label
        x,y = self.axs[aax].transData.inverted().transform((event.x,event.y))
        y = -y
        self.measurement['ax'] = aax
        self.measurement['x0'] = x
        self.measurement['y0'] = y
        if False:
            self.axs[aax].plot(x,y,'+')
        self.canvas.get_tk_widget().config(cursor='sizing')

    # draw line for current linear measurement
    def draw_measurement(self,ax,x,y):
        if self.measurement['ax'] != ax or self.measurement['x0'] is None:
            return
        if self.measurement['plot']:
            try:
                Artist.remove(self.measurement['plot'])
            except ValueError as e:
                print(e)
        x = np.array([self.measurement['x0'],x])
        y = np.array([self.measurement['y0'],y])
        self.measurement['plot'] = self.axs[ax].plot(x,y,'b',clip_on=True)[0]
        self.measurement['ax'] = ax
        self.measurement['l'] = np.sqrt(np.power(x[1]-x[0],2)+np.power(y[1]-y[0],2))
        self.ui.set_message(msg='distance = {:.1f}'.format(self.measurement['l']))
        return

    # remove existing measurement line
    def clear_measurement(self):
        Artist.remove(self.measurement['plot'])
        self.measurement = {'ax':None,'x0':None,'y0':None,'plot':None,'l':None}
        self.ui.clear_message()
        self.canvas.draw()


    # mouse drag event for 3d crosshair overlay
    # only tested in blast viewer
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

