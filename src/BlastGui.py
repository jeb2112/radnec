import os,sys
import copy
import time
from tkinter import messagebox,ttk,PhotoImage
import tkinter as tk
from cProfile import Profile
from pstats import SortKey,Stats

import Config
import Blastbratsv3
from CreateFrame import CreateCaseFrame,CreateSliceViewerFrame
from CreateROIFrame import CreateROIFrame
from OverlayPlots import *

# main gui class
class BlastGui(object):
    def __init__(self, root, toolsFlag, config, debug=False):
        self.root = root
        self.toolsFlag = toolsFlag
        self.titletext = 'BLAST User Interface'
        self.root.title(self.titletext)
        self.config = config
        self.logoImg = os.path.join(self.config.UIResourcesPath,'sunnybrook.png')
        self.blastImage = PhotoImage(file=self.logoImg)
        self.normalslice = None
        self.currentslice = None # tracks the current slice widget variable
        self.dataselection = 'raw'

        self.data = {'gates':[None,None,None], 'wt':None,'et':None,'tc':None}
        # self.data = {}
        self.data['params'] = {'stdt1':1,'stdt2':1,'meant1':1,'meant2':1}

        self.roi = []
        self.currentroi = -1 # tracks the currentroi widget variable
        self.currentlayer = 0
        self.OS = sys.platform
        self.tstart = time.time()

        self.message = tk.StringVar(value='')
        self.debug = debug

        self.createGeneralLayout()

        # hard-coded for debugging
        if self.debug:
            self.caseframe.casename.set('00005')
            self.sliceviewerframe.currentslice.set(75)
            self.updateslice()
            self.sliceviewerframe.normalslice_callback()
            self.sliceviewerframe.currentslice.set(105)
            self.updateslice()

            # adjusted BLAST
            # self.roiframe.currentt2threshold.set(0.2)
            self.roiframe.currentt1threshold.set(0.1)
            self.roiframe.updatet1threshold()
            # self.roiframe.updatet2threshold()
            self.roiframe.enhancingROI_callback()
            self.updateslice()

            # adjusted window/level
            # self.sliceviewerframe.window=np.array([.6,1],dtype='float')
            # self.sliceviewerframe.level = np.array([.3,.5])

            # create roi
            # self.roiframe.createROI(132,102,75)
            # self.roiframe.ROIclick(event=None)
            # self.roiframe.updateROI()
            # self.roiframe.finalROI_overlay_value.set(True)
            # self.roiframe.update_layermenu_options('seg')
            # self.roiframe.enhancingROI_overlay_value.set(False)
            # self.currentroi = 0
            # self.roiframe.currentroi.set(0)
            # self.roiframe.update_roinumber_options()
            # self.roiframe.roinumber_callback(item=None)


    #########
    # main UI
    #########

    def createGeneralLayout(self):
        # create the main holder frame
        self.mainframe = ttk.Frame(self.root, padding='10')
        self.mainframe.grid(column=0, row=0, sticky='NEW')
        self.mainframe.columnconfigure(0,weight=1)
        # self.mainframe.columnconfigure(1,weight=1)

        # slice viewer frame
        self.sliceviewerframe = CreateSliceViewerFrame(self.mainframe,ui=self,padding='10')        

        # roi functions
        self.roiframe = CreateROIFrame(self.sliceviewerframe.frame,ui=self,padding='0')

        # create case frame
        self.caseframe = CreateCaseFrame(self.sliceviewerframe.frame,ui=self,load=self.config.AutoLoad)

        self.mainframe.update()


    ##############
    # BLAST method
    ##############

    def runblast(self,currentslice=None):
        if currentslice: # 2d in a single slice
            currentslice=self.currentslice
        else: # entire volume
            self.root.config(cursor='watch')
            self.root.update_idletasks()

        if self.config.WLClip:
            self.sliceviewerframe.clipwl_raw()

        if False:
            with Profile() as profile:
                self.data['seg_raw'],self.data['seg_raw_fusion'] = self.blast.run_blast(self.data,
                                                self.roiframe.t1slider.get(),
                                                self.roiframe.t2slider.get(),self.roiframe.bcslider.get(),
                                                currentslice=currentslice)
                (
                    Stats(profile)
                    .strip_dirs()
                    .sort_stats(SortKey.TIME)
                    .print_stats(15)
                )
        else:
            try:
                retval = Blastbratsv3.run_blast(
                                    self.data,self.roiframe.t1slider.get(),
                                    self.roiframe.t2slider.get(),self.roiframe.bcslider.get(),
                                    currentslice=currentslice)
                if retval is not None:
                    self.data['seg_raw'],self.data['gates'] = retval
            except ValueError as e:
                self.set_message(e)

        if self.config.WLClip:
            self.sliceviewerframe.restorewl_raw()
            self.sliceviewerframe.window = np.array([1.,1.],dtype='float')
            self.sliceviewerframe.level = np.array([0.5,0.5],dtype='float')
            self.updateslice()


        self.data['seg_raw_fusion'] = generate_overlay(self.data['raw'],self.data['seg_raw'],self.roiframe.layer.get(),overlay_intensity=self.config.OverlayIntensity)
        self.data['seg_raw_fusion_d'] = copy.copy(self.data['seg_raw_fusion'])
        self.data['params']['t1gate_count'] = self.data['gates'][3]
        self.data['params']['t2gate_count'] = self.data['gates'][4]
            
        if self.roiframe.finalROI_overlay_value.get() == True:
            self.dataselection = 'seg_fusion_d'
        else:
            self.dataselection = 'seg_raw_fusion_d'
                
        if currentslice is None:
            self.updateslice(wl=True)
        else:
            self.updateslice()
        
        # in this 2d preview mode, the enhancing lesion is only being calculated slice by slice
        # nonetheless the latency is still measurable, so only want to update when button click
        # is released
        if currentslice:
            self.sliceviewerframe.vsliceslider['command'] = None
            if self.roiframe.enhancingROI_overlay_value.get() == True:
                self.sliceviewerframe.vsliceslider.bind("<ButtonRelease-1>",self.sliceviewerframe.updateslice_blast)
            elif self.roiframe.finalROI_overlay_value.get() == True:
                self.sliceviewerframe.vsliceslider.bind("<ButtonRelease-1>",self.sliceviewerframe.updateslice_roi)
        else:
            self.sliceviewerframe.vsliceslider.unbind("<ButtonRelease-1>")
            self.sliceviewerframe.vsliceslider['command'] = self.updateslice
        
        self.root.config(cursor='arrow')
        self.root.update_idletasks()

        return None
    
    #############################
    ###### Utility methods ######
    #############################

    def set_currentslice(self,val=None):
        self.currentslice = self.sliceviewerframe.currentslice.get()

    def get_currentslice(self):
        return self.currentslice
    
    def set_casename(self,val=None):
        self.casename = self.caseframe.casename.get()

    def get_casename(self):
        return self.casename
    
    def updateslice(self,event=None,**kwargs):
        self.sliceviewerframe.updateslice(event,**kwargs)

    def set_currentroi(self,val=None):
        self.currentroi = self.roiframe.currentroi.get()

    def get_currentroi(self):
        return self.currentroi

    def starttime(self):
        self.tstart = time.time()

    def endtime(self):
        self.roi[self.currentroi].stats['elapsed_time'] = time.time() - self.tstart

    def clear_message(self):
        self.sliceviewerframe.messagelabel['text'] = ''
        self.message.set('')

    def set_message(self, msg=None):
        if msg is None:
            msg = self.message.get()
        else:
            self.message.set(msg)
        self.sliceviewerframe.messagelabel['text'] = msg
