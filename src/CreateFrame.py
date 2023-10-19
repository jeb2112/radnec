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
matplotlib.use('TkAgg')
import SimpleITK as sitk
from sklearn.cluster import KMeans,MiniBatchKMeans
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

        # window/level values
        self.window = np.array([1.,1.],dtype='float')
        self.level = np.array([0.5,0.5],dtype='float')
        self.wlflag = False
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

        # messages text frame
        self.messagelabel = ttk.Label(self.frame,text=self.ui.message.get(),padding='5',borderwidth=0,relief='groove')
        self.messagelabel.grid(column=0,row=5,columnspan=3,sticky='ew')

        OS = sys.platform
        if OS in ('win32','darwin'):
            # self.frame.bind('<MouseWheel>', self.updatew2())
            # self.frame.bind('<Shift-MouseWheel>', self.updatel2())
            self.ui.root.bind('<Button>',self.touchpad)
        if OS == 'linux':
            self.ui.root.bind('<Button>',self.touchpad)
            # self.ui.root.bind('<ButtonRelease>',self.touchpad)

    # TODO: different bindings and callbacks need some organization
    def updateslice(self,event=None,wl=False,blast=False):
        slice=self.currentslice.get()
        self.ui.set_currentslice(slice)
        if blast: # option for previewing enhancing in 2d
            self.ui.runblast(currentslice=slice)
        self.ax_img.set(data=self.ui.data[self.ui.dataselection][0,slice,:,:])
        self.ax2_img.set(data=self.ui.data[self.ui.dataselection][1,slice,:,:])
        self.vslicenumberlabel['text'] = '{}'.format(slice)
        if self.ui.dataselection == 'enhancefusion_d' or self.ui.dataselection == 'segfusion_d':
            self.ax_img.set(cmap='viridis')
            self.ax2_img.set(cmap='viridis')
        else:
            self.ax_img.set(cmap='gray')
            self.ax2_img.set(cmap='gray')
        if wl:   # eg latency problem here
            self.updatewl_fusion()
        self.canvas.draw()

    # special update for previewing BLAST enhancing lesion in 2d
    def updateslice_blast(self,event=None):
        slice = self.currentslice.get()
        self.ui.set_currentslice(slice)
        self.ui.runblast(currentslice=slice)
        self.vslicenumberlabel['text'] = '{}'.format(slice)
        self.canvas.draw()

    # update for previewing final segmentation in 2d
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
        slice=self.currentslice.get()
        for ax in range(2):
            vmin = self.level[ax] - self.window[ax]/2
            vmax = self.level[ax] + self.window[ax]/2
            if self.wlflag:
                if self.ui.dataselection in ['seg_raw_fusion_d','seg_fusion_d']:
                    for d in ['seg_raw_fusion_d','seg_fusion_d']:
                        if d in self.ui.data.keys():
                            self.ui.data[d] = self.ui.caseframe.rescale(self.ui.data[d[:-2]],vmin=vmin,vmax=vmax)
                    if ax == 0:
                        self.ax_img.set(data=self.ui.data[self.ui.dataselection][ax,slice,:,:])
                    else:
                        self.ax2_img.set(data=self.ui.data[self.ui.dataselection][ax,slice,:,:])
            self.wlflag = False
            
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


class CreateCaseFrame(CreateFrame):
    def __init__(self,parent,ui=None):
        super().__init__(parent,ui=ui)

        self.fd = FileDialog(initdir=self.config.UIdatadir)
        self.caselist = ['00000','00001']
        self.datadir = StringVar()
        self.datadir.set(self.config.UIdatadir)
        self.casename = StringVar()
        self.casename.set(self.caselist[0])
        self.casefile_prefix = None

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
        # self.ui.datadirentry.insert(INSERT,self.config.UIdatadir)
        self.datadirentry.grid(column=3,row=0,columnspan=5)
        caselabel = ttk.Label(caseframe, text='Case: ')
        caselabel.grid(column=0,row=0,sticky='we')
        self.casename.trace_add('write',lambda *args: self.casename.get())
        self.w = ttk.OptionMenu(caseframe,self.casename,*self.caselist,command=self.case_callback)
        self.w.grid(column=1,row=0)

    def select_dir(self):
        self.fd.select_dir()
        self.datadir.set(self.fd.dir)
        self.datadirentry.update()
        self.datadirentry_callback()
        self.casename.set(self.caselist[0])
        self.ui.set_casename(self.caselist[0])

    def case_callback(self,case):
        print('Loading case {}'.format(case))
        self.casename.set(case)
        self.ui.set_casename(case)
        self.loadCase()
        self.ui.dataselection = 'raw'
        self.ui.sliceviewerframe.tbar.home()
        self.ui.updateslice()
        self.ui.starttime()

    def loadCase(self,case=None):
        if case is not None:
            self.casename.set(case)
        self.casedir = os.path.join(self.config.UIdatadir,self.config.UIdataroot+self.casename.get())
        # create t1mprage template
        t1ce = sitk.ReadImage(os.path.join(self.casedir,self.config.UIdataroot + self.casename.get() + "_t1ce_bias.nii"))
        img_arr = sitk.GetArrayFromImage(t1ce)
        # 2 channels hard-coded
        self.ui.data['raw'] = np.zeros((2,)+np.shape(img_arr))
        self.ui.data['raw'][0] = img_arr

        # Create t2flair template 
        t2flair = sitk.ReadImage(os.path.join(self.casedir,self.config.UIdataroot + self.casename.get() + "_flair_bias.nii") )
        img_arr = sitk.GetArrayFromImage(t2flair)
        self.ui.data['raw'][1] = img_arr

        # rescale the data
        self.ui.data['raw'] = self.rescale(self.ui.data['raw'])

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


    # assumes first dim is channel
    def rescale(self,img_arr,vmin=None,vmax=None):
        scaled_arr =  np.zeros(np.shape(img_arr))
        for ch in range(np.shape(img_arr)[0]):
            if vmin is None:
                minv = np.min(img_arr[ch])
            else:
                minv = vmin
            if vmax is None:
                maxv = np.max(img_arr[ch])
            else:
                maxv = vmax
            assert(maxv>minv)
            scaled_arr[ch] = (img_arr[ch]-minv) / (maxv-minv)
            scaled_arr[ch] = np.clip(scaled_arr[ch],a_min=0,a_max=1)
        return scaled_arr

    def datadirentry_callback(self,event=None):
        dir = self.datadir.get().strip()
        if os.path.exists(dir):
            files = os.listdir(dir)
            self.casefile_prefix = re.match('(^.*)0[0-9]{4}',files[0]).group(1)
            casefiles = [re.match('.*(0[0-9]{4})',f).group(1) for f in files if re.search('_0[0-9]{4}$',f)]
            self.ui.set_message('')
            if len(casefiles):
                # TODO: will need a better sort here
                self.caselist = sorted(casefiles)
                self.w.config(state='normal')
                self.w.set_menu(*self.caselist,default=self.caselist[0])
            else:
                print('No cases found in directory {}'.format(dir))
                self.ui.set_message('No cases found in directory {}'.format(dir))
                self.w.config(state='disable')
        else:
            print('Directory {} not found.'.format(dir))
            self.ui.set_message('Directory {} not found.'.format(dir))
            self.w.config(state='disable')
            self.datadirentry.update()
        return
