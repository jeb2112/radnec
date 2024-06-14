import os,sys
import re
import numpy as np
import pickle
import copy
import logging
import time
import asyncio
import concurrent
import tkinter as tk
from tkinter import ttk
import tk_async_execute as tae
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import _Mode
matplotlib.use('TkAgg')
import SimpleITK as sitk
import nibabel as nb
from skimage.morphology import disk,square,binary_dilation,binary_closing,flood_fill,ball,cube,reconstruction
from skimage.measure import find_contours
from scipy.spatial.distance import dice
from scipy.ndimage import binary_closing as scipy_binary_closing
from scipy.io import savemat

from src.OverlayPlots import *
from src.CreateFrame import CreateFrame,Command
from src.ROI import ROILinear
from src.CreateTranscription import *

# contains various ROI methods and variables for 'overlay' mode
class Create4PanelROIFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)

        #tracks current measurement
        self.currentroi = tk.IntVar(value=0)
        self.roilist = []
        # variables for the record button
        self.recordtext = tk.StringVar(value='off')
        self.record = tk.IntVar(value=0)


        ########################
        # layout for the buttons
        ########################

        # dummy frame to hide
        self.dummy_frame = ttk.Frame(self.parentframe,padding='0')
        self.dummy_frame.grid(row=2,column=4,rowspan=5,sticky='news')

        # actual frame
        self.frame.grid(row=2,column=4,rowspan=5,sticky='ne')
        # record button
        self.fstyle.configure('frame.TCheckbutton',background='#AAAAAA')
        self.recordbutton = ttk.Checkbutton(self.frame,style='Toolbutton',textvariable=self.recordtext,variable=self.record,command=self.record_transcript)
        self.recordbutton.configure(style='frame.TCheckbutton')
        self.recordbutton.grid(row=0,column=1,sticky='w')
        self.recordbuttonlabel = ttk.Label(self.frame,text='record:',padding='5')
        self.recordbuttonlabel.grid(row=0,column=0,sticky='e')

        # roi list
        # for multiple roi's, n'th roi number choice
        roinumberlabel = ttk.Label(self.frame,text='ROI number:')
        roinumberlabel.grid(row=0,column=4,sticky='w')
        self.currentroi.trace_add('write',self.set_currentroi)
        self.roinumbermenu = ttk.OptionMenu(self.frame,self.currentroi,*self.roilist,command=self.roinumber_callback)
        self.roinumbermenu.config(width=2)
        self.roinumbermenu.grid(row=0,column=5,sticky='w')
        self.roinumbermenu.configure(state='disabled')

        # save ROI button
        saveROI = ttk.Button(self.frame,text='save ROI',command = self.saveROI)
        saveROI.grid(row=1,column=6,sticky='w')

        # clear ROI button
        clearROI = ttk.Button(self.frame,text='clear ROI',command = self.clearROI)
        clearROI.grid(row=1,column=5,sticky='w')
        self.frame.update()

        return


    #############
    # ROI methods
    ############# 
        
    # record and transcribe
    def record_transcript(self):
        if self.record.get():
            self.recordtext.set('on')
            self.frame.update()
            self.T = Transcription(self.ui.root)

            # with asyncio.run(), can exit with CtrLC and handling errors
            if False:
                try:
                    # for a mic stream there is no completion event within the stream iteslf,
                    # requires user input. 
                    asyncio.run(self.T.basic_transcribe())
                except asyncio.CancelledError as e:
                    self.recordtext.set('off')
                    self.record.set(0)
                except KeyboardInterrupt as e:
                    print('keyboard interrupt')
                except RuntimeError as e:
                    self.recordtext.set('off')
                    self.record.set(0)
                finally:
                    self.recordtext.set('off')
                    self.record.set(0)

            # with tae module, no CtrlC and don't need to handle errors
            tae.async_execute(self.T.basic_transcribe(),wait=False)

        else:
            self.recordtext.set('off')
            self.record.set(0)
            self.transcript = self.T.handler.transcript
            try:
                tae.stop()
            # there is a runtime error being thrown at line 515 in base_events _check_closed
            # not sure how to handle it yet
            except RuntimeError as e:
                print('event loop is closed')
            tae.start()
        return

    def resetROI(self):
        return
    
    def resetCursor(self,event=None):
        self.ui.sliceviewerframe.canvas.get_tk_widget().config(cursor='watch')
        self.ui.sliceviewerframe.canvas.get_tk_widget().update_idletasks()
    
    # methods for roi number choice menu
    def roinumber_callback(self,item=None):

        self.ui.set_currentroi()
        # reference or copy
        self.ui.updateslice()
        self.ui.sliceviewerframe.show_measurement()
        return
    
    def update_roinumber_options(self,n=None):
        if n is None:
            n = len(self.ui.roi[self.ui.s])
        menu = self.roinumbermenu['menu']
        menu.delete(0,'end')
        # 1-based indexing
        for s in [str(i) for i in range(1,n)]:
            menu.add_command(label=s,command = tk._setit(self.currentroi,s,self.roinumber_callback))
        self.roilist = [str(i) for i in range(1,n)]
        if n>1:
            self.roinumbermenu.configure(state='active')
        else:
            self.roinumbermenu.configure(state='disabled')

    def set_currentroi(self,var,index,mode):
        if mode == 'write':
            self.ui.set_currentroi()    

    # records a measurement in a new ROI object
    def createROI(self,x0,y0,x1,y1,slice,channel):
        roi = ROILinear(x0,y0,x1,y1,slice,channel)
        self.ui.roi[self.ui.s].append(roi)
        self.currentroi.set(self.currentroi.get() + 1)
        self.update_roinumber_options()

    # alters a measurement in an existing ROI 
    def updateROI(self,event):
        compartment = self.layer.get()
        roi = self.ui.roi[self.ui.s][self.ui.get_currentroi()]
        roi.coords['x'] = int(event.xdata)
        roi.coords['y'] = int(event.ydata)
        roi.coords['slice'] = self.ui.get_currentslice()

    # for exporting ROI measurements. 
    def saveROI(self,roi=None):
        # Save ROI data
        outputpath = self.ui.caseframe.casedir
        fileroot = os.path.join(outputpath,self.ui.caseframe.casefile_prefix + self.ui.caseframe.casename.get())
        filename = fileroot+'_stats.pkl'
        # t1mprage template? need to save separately?

        # BLAST outputs. combined ROI or separate? doing separate for now
        roisuffix = ''
        for img in ['seg','ET','TC','WT']:
            for roi in self.roilist:
                if len(self.roilist) > 1:
                    roisuffix = '_roi'+roi
                outputfilename = fileroot + '_blast_' + img + roisuffix + '.nii'
                if self.ui.roi[self.ui.s][int(roi)].data[img] is not None:
                    self.WriteImage(self.ui.roi[self.ui.s][int(roi)].data[img],outputfilename,affine=self.ui.affine['t1'])

        sdict = {}
        bdict = {}
        for i,r in enumerate(self.ui.roi[self.ui.s][1:]): # skip dummy 
            sdict['roi'+str(i)] = r.stats
            bdict['roi'+str(i)] = dict((k,r.data[k]) for k in ('ET','TC','WT','blast','raw'))

        with open(filename,'ab') as fp:
            pickle.dump((sdict,bdict),fp)
        # matlab compatible output
        filename = filename[:-3] + 'mat'
        with open(filename,'ab') as fp:
            savemat(filename,sdict,bdict)

    # eliminate latest ROI if there are multiple ROIs in current case
    def clearROI(self):
        n = len(self.ui.roi[self.ui.s])
        if n>1:    
            self.ui.roi[self.ui.s].pop(self.ui.currentroi)
            n -= 1
            if self.ui.currentroi > 1 or n==1:
                # new current roi is decremented as an arbitrary choice
                # or if all rois are now gone
                self.currentroi.set(self.currentroi.get()-1)
            self.update_roinumber_options()
            if n > 1:
                self.roinumber_callback()
            if n==1:
                self.resetROI()
                self.ui.updateslice()

    # eliminate all ROIs, ie for loading another case
    def resetROI(self):
        self.currentroi.set(0)
        self.ui.roi[self.ui.s] = [0]
        self.update_roinumber_options()

    def append_roi(self,d):
        for k,v in d.items():
            if isinstance(v,dict):
                self.append_roi(d)
            else:
                v.append(0)
