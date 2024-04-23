import os,sys
import copy
import time
import subprocess
import screeninfo
from tkinter import messagebox,ttk,PhotoImage
import tkinter as tk
from cProfile import Profile
from pstats import SortKey,Stats
import matplotlib.pyplot as plt
from importlib import metadata

from src.CreateSVFrame import CreateSliceViewerFrame
from src.CreateCaseFrame import CreateCaseFrame
from src.CreateROIFrame import CreateROIFrame
from src.OverlayPlots import *

# main gui class
class BlastGui(object):
    def __init__(self, root, toolsFlag, config, debug=False):
        self.root = root
        self.dpi = self.root.winfo_fpixels('1i')
        current_screen = self.get_monitor_from_coord(self.root.winfo_x(), self.root.winfo_y())
        self.default_panelsize = current_screen.height/self.dpi * 0.5
        self.current_panelsize = self.default_panelsize

        self.toolsFlag = toolsFlag
        self.version = ''
        # self.version = metadata.version('radnec')
        self.titletext = 'RADNEC User Interface v' + self.version

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
                            # t12,flair are slider setting for y-axis t1/t2 depending on layer, and flair common x-axis
                            # std,mean are the cluster stats. 
                            'params':{'ET':{'t12':0.0,'bc':0.0,'flair':0.0,'stdt12':1,'stdflair':1,'meant12':1,'meanflair':1},
                               'T2 hyper':{'t12':0.0,'bc':0.0,'flair':0.0,'stdt12':1,'stdflair':1,'meant12':1,'meanflair':1},
                               },
                    },
        }

        self.affine = {'t1pre':None,'t1':None,'t2':None,'flair':None}
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
                self.roiframe.t2threshold.set(-0.8)
                self.roiframe.t1threshold.set(-0.7)
                self.roiframe.bcsize.set(1.0)
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
        self.mainframe_padding = '10'
        self.mainframe = ttk.Frame(self.root, padding=self.mainframe_padding)
        self.mainframe.grid(column=0, row=0, sticky='NSEW')

        # create case frame
        self.caseframe = CreateCaseFrame(self.mainframe,ui=self)

        # slice viewer frame
        self.sliceviewerframe = CreateSliceViewerFrame(self.mainframe,ui=self,padding='0')

        # roi functions
        self.roiframe = CreateROIFrame(self.mainframe,ui=self,padding='0')

        # initialize default directory.
        if False:
            self.caseframe.datadirentry_callback()

        for row_num in range(self.mainframe.grid_size()[1]):
            if row_num == 1:
                self.mainframe.rowconfigure(row_num,weight=1)
            else:
                self.mainframe.rowconfigure(row_num,weight=0)
        self.mainframe.columnconfigure(0,minsize=self.caseframe.frame.winfo_width(),weight=1)
        self.mainframe.bind('<Configure>',self.sliceviewerframe.resizer)
        self.mainframe.update()

    
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

    def get_bcsize(self,layer=None):
        if layer is None:
            layer = self.roiframe.layer.get()
        return self.roiframe.thresholds[layer]['bc'].get()
    
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

        self.data = {'blast':{'gates':{'ET':None,'T2 hyper':None,'brain ET':None,'brain T2 hyper':None},
                            'T2 hyper':None,
                            'ET':None,
                            'params':{'ET':{'t12':0.0,'bc':0.0,'flair':0.0,'stdt12':1,'stdflair':1,'meant12':1,'meanflair':1},
                               'T2 hyper':{'t12':0.0,'bc':0.0,'flair':0.0,'stdt12':1,'stdflair':1,'meant12':1,'meanflair':1},
                               },
                    },
        }
    
        self.roi = [0] # dummy value for Roi indexing 1-based
        self.currentroi = 0 # tracks the currentroi widget variable
        self.currentlayer = 0
        self.tstart = time.time()
        self.message = tk.StringVar(value='')
        self.roiframe.resetROI()

    def get_monitor_from_coord(self,x, y):
        monitors = screeninfo.get_monitors()
        for m in reversed(monitors):
            if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
                return m
        return monitors[0]
