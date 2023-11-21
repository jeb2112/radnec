import os,sys
import copy
import time
import subprocess
from tkinter import messagebox,ttk,PhotoImage
import tkinter as tk
from cProfile import Profile
from pstats import SortKey,Stats
import matplotlib.pyplot as plt
from importlib import metadata

from src import Blastbratsv3
from src.CreateFrame import CreateCaseFrame,CreateSliceViewerFrame
from src.CreateROIFrame import CreateROIFrame
from src.OverlayPlots import *

# main gui class
class BlastGui(object):
    def __init__(self, root, toolsFlag, config, debug=False):
        self.root = root
        self.toolsFlag = toolsFlag
        self.version = None
        self.version = metadata.version('blast')
        self.titletext = 'BLAST User Interface v' + self.version

        self.root.title(self.titletext)
        self.config = config
        self.config.dpi = self.root.winfo_fpixels('1i')
        self.config.swi = self.root.winfo_screenwidth()/self.config.dpi
        self.config.shi = self.root.winfo_screenheight()/self.config.dpi

        self.logoImg = os.path.join(self.config.UIResourcesPath,'sunnybrook.png')
        self.blastImage = PhotoImage(file=self.logoImg)
        self.normalslice = None
        self.currentslice = None # tracks the current slice widget variable
        self.dataselection = 'raw'

        # initialize data struct. other keys are added elsewhere
        self.data = {'blast':{'gates':{'ET':None,'T2 hyper':None,'brain ET':None,'brain T2 hyper':None},
                            'T2 hyper':None,
                            'ET':None,
                            'params':{'ET':{'t1':0.0,'t2':0.0,'bc':3.0},
                               'T2 hyper':{'t1':0.0,'t2':0.0,'bc':0.0},
                               'stdt1':1,
                               'stdt2':1,
                               'meant1':1,
                               'meant2':1}
                               },
                    }

        self.roi = [0] # dummy value for Roi indexing 1-based
        self.currentroi = 0 # tracks the currentroi widget variable
        self.currentlayer = 0
        self.OS = sys.platform
        self.tstart = time.time()

        self.message = tk.StringVar(value='')
        self.debug = debug

        self.createGeneralLayout()

        # hard-coded for debugging
        if self.debug:
            # 00005 75,105
            # 00002 53,81
            self.caseframe.n4_check_value.set(0)
            self.caseframe.casename.set('00006')
            self.caseframe.case_callback()
            if self.sliceviewerframe.slicevolume_norm.get() == 0:
                self.sliceviewerframe.currentslice.set(53)
                self.updateslice()
                self.sliceviewerframe.normalslice_callback()
            self.sliceviewerframe.currentslice.set(81)
            self.updateslice()

            # adjusted BLAST
            if False:
                self.roiframe.currentt2threshold.set(-0.8)
                self.roiframe.currentt1threshold.set(-0.7)
                self.roiframe.currentbcsize.set(1.0)
                self.roiframe.updatebcsize()
                self.roiframe.updatet1threshold()
                self.roiframe.updatet2threshold()
                self.roiframe.enhancingROI_callback()
                self.updateslice()

            # adjusted window/level
            # self.sliceviewerframe.window=np.array([.6,1],dtype='float')
            # self.sliceviewerframe.level = np.array([.3,.5])

            # create roi. might be a bug arising from this automation that isn't seen manually
            if False:
                # self.roiframe.createROI(132,102,75) # 00000
                self.roiframe.createROI(155,99,87) # 00002
                self.roiframe.ROIclick(event=None)
                self.roiframe.updateROI()
                self.roiframe.finalROI_overlay_value.set(True)
                self.roiframe.update_layermenu_options('seg')
                self.roiframe.enhancingROI_overlay_value.set(False)
                self.currentroi = 0
                self.roiframe.currentroi.set(0)
                self.roiframe.update_roinumber_options()
                self.roiframe.roinumber_callback(item=None)


    #########
    # main UI
    #########

    def createGeneralLayout(self):
        # create the main holder frame
        self.mainframe = ttk.Frame(self.root, padding='10')
        self.mainframe.grid(column=0, row=0, sticky='NSEW')

        # create case frame
        self.caseframe = CreateCaseFrame(self.mainframe,ui=self)

        # slice viewer frame
        self.sliceviewerframe = CreateSliceViewerFrame(self.mainframe,ui=self,padding='10')

        # roi functions
        self.roiframe = CreateROIFrame(self.mainframe,ui=self,padding='0')

        # initialize default directory.
        self.caseframe.datadirentry_callback()

        for row_num in range(self.mainframe.grid_size()[1]):
            if row_num == 1:
                self.mainframe.rowconfigure(row_num,weight=1)
            else:
                self.mainframe.rowconfigure(row_num,weight=0)
        # self.sliceviewerframe.frame.bind('<Configure>',self.sliceviewerframe.resizer)
        self.mainframe.bind('<Configure>',self.sliceviewerframe.resizer)
        self.mainframe.update()
        self.sliceviewerframe.normal_frame_minsize = self.sliceviewerframe.normal_frame.winfo_height()
        self.mainframe.rowconfigure(2,minsize=self.sliceviewerframe.normal_frame_minsize)


    ##############
    # BLAST method
    ##############

    def runblast(self,currentslice=None,layer=None):
        if currentslice: # 2d in a single slice
            currentslice=None # for now will run full 3d by default every update
        else: # entire volume
            self.root.config(cursor='watch')
            self.root.update_idletasks()

        if layer is None:   
            layer = self.roiframe.layer.get()
        clustersize = self.get_currentbcsize()

        # if self.config.WLClip:
        #     self.sliceviewerframe.clipwl_raw()

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
                                    self.roiframe.t2slider.get(),clustersize,layer,
                                    currentslice=currentslice)
                if retval is not None:
                    self.data['blast'][layer],self.data['blast']['gates']['brain '+layer],self.data['blast']['gates'][layer] = retval
                    self.update_blast(layer=layer)
            except ValueError as e:
                self.set_message(e)

        # if self.config.WLClip:
        #     self.sliceviewerframe.restorewl_raw()
        #     self.sliceviewerframe.window = np.array([1.,1.],dtype='float')
        #     self.sliceviewerframe.level = np.array([0.5,0.5],dtype='float')
        #     self.updateslice()

        self.data['seg_raw_fusion'] = generate_overlay(self.data['raw'],self.data['seg_raw'],layer=self.roiframe.layer.get(),
                                                       overlay_intensity=self.config.OverlayIntensity)
        self.data['seg_raw_fusion_d'] = copy.deepcopy(self.data['seg_raw_fusion'])
            
        if self.roiframe.finalROI_overlay_value.get() == True:
            self.dataselection = 'seg_fusion_d'
        else:
            self.dataselection = 'seg_raw_fusion_d'
                
        if currentslice is None:
            self.updateslice(wl=True,layer=layer)
        else:
            self.updateslice()
        
        # in this 2d preview mode, the enhancing lesion is only being calculated slice by slice
        # nonetheless the latency is still measurable, so only want to update when button click
        # is released. not using 2d preview anymore
        if False:
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
        if val:
            self.casename = val
        else:   
            self.casename = self.caseframe.casename.get()

    def get_casename(self):
        return self.casename
    
    def updateslice(self,event=None,**kwargs):
        self.sliceviewerframe.updateslice(event,**kwargs)

    def set_currentroi(self,val=None):
        self.currentroi = self.roiframe.currentroi.get()

    def get_currentroi(self):
        return self.currentroi

    def get_currentbcsize(self):
        return self.roiframe.currentbcsize.get()

    def update_roidata(self):
        self.roiframe.updateROIData()

    def update_blast(self,**kwargs):
        self.roiframe.updateBLAST(**kwargs)

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

    def resetUI(self):
        self.normalslice = None
        self.currentslice = None
        self.dataselection = 'raw'

        # self.data = {'gates':{'ET':None,'T2 hyper':None,'brain ET':None,'brain T2 hyper':None},'blast T2 hyper':None,'blast ET':None}
        # self.data['blast']['params'] = {'stdt1':1,'stdt2':1,'meant1':1,'meant2':1}
        self.data = {'blast':{'gates':{'ET':None,'T2 hyper':None,'brain ET':None,'brain T2 hyper':None},
                            'T2 hyper':None,
                            'ET':None,
                            'params':{'ET':{'t1':0.0,'t2':0.0,'bc':3.0},
                               'T2 hyper':{'t1':0.0,'t2':0.0,'bc':0.0},
                               'stdt1':1,
                               'stdt2':1,
                               'meant1':1,
                               'meant2':1}
                               }
                    }

        self.roi = [0] # dummy value for Roi indexing 1-based
        self.currentroi = 0 # tracks the currentroi widget variable
        self.currentlayer = 0
        self.tstart = time.time()
        self.message = tk.StringVar(value='')
