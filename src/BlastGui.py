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

from src import Blastbratsv3
from src.sliceviewer.CreateOverlaySVFrame import CreateOverlaySVFrame
from src.sliceviewer.CreateBlastSVFrame import CreateBlastSVFrame
from src.sliceviewer.CreateSAMSVFrame import CreateSAMSVFrame
from src.sliceviewer.Create4PanelSVFrame import Create4PanelSVFrame
from src.CreateCaseFrame import CreateCaseFrame
from src.roi.CreateSAMROIFrame import CreateSAMROIFrame
from src.CreateOverlayFrame import CreateOverlayFrame
from src.roi.Create4PanelROIFrame import Create4PanelROIFrame
from src.CreateFrame import Command
from src.OverlayPlots import *
from src.sam.SAM import SAM

# main gui class
class BlastGui(object):
    def __init__(self, root, toolsFlag, config, debug=False):
        self.root = root
        self.dpi = self.root.winfo_fpixels('1i')
        current_screen = self.get_monitor_from_coord(self.root.winfo_x(), self.root.winfo_y())
        self.default_panelsize = current_screen.height/self.dpi * 0.5
        self.current_panelsize = self.default_panelsize

        self.toolsFlag = toolsFlag
        try:
            self.version = metadata.version('radnec')
        except metadata.PackageNotFoundError:
            self.version = ''
        self.titletext = 'Viewer v' + self.version

        self.root.title(self.titletext)
        self.config = config
        self.config.dpi = self.root.winfo_fpixels('1i')
        self.config.swi = self.root.winfo_screenwidth()/self.config.dpi
        self.config.shi = self.root.winfo_screenheight()/self.config.dpi

        self.logoImg = os.path.join(self.config.UIResourcesPath,'sunnybrook.png')
        self.blastImage = PhotoImage(file=self.logoImg)

        self.normalslice = None # for 2d normal stats, probably not needed anymore
        self.currentslice = None # tracks the current slice widget variable
        self.s = 0 # convenience attribute, tracks the current study number in caseframe widget
        # top level of data structure, could be raw data or overlay image
        self.dataselection = 'raw'
        # 2nd level of data structure is image channel, either as raw data or underlay data
        self.chselection = self.config.DefaultChannel 

        # viewer functions. overlay mode, or BLAST segmentation mode
        self.functionlist = {'overlay':0,'BLAST':1,'4panel':2,'SAM':3}
        self.function = tk.StringVar(value=self.config.DefaultViewer)

        # data structure. data is a dict of studies. see DcmCase
        self.data = {} 
        # additional data related to BLAST, per study
        self.blastdatadict = {'blast':
                              {'gates':{'ET':None,'T2 hyper':None,'brain ET':None,'brain T2 hyper':None},
                            'T2 hyper':None,
                            'ET':None,
                            'params':{'ET':{'t12':0.0,'bc':0.0,'flair':0.0,'stdt12':1,'stdflair':1,'meant12':1,'meanflair':1},
                               'T2 hyper':{'t12':0.0,'bc':0.0,'flair':0.0,'stdt12':1,'stdflair':1,'meant12':1,'meanflair':1},
                               },
                            },
                            'blastpoint':
                            {'params':{'ET':{'stdt12':[],'stdflair':[],'meant12':[],'meanflair':[],'pt':[]},
                                       'T2 hyper':{'stdt12':[],'stdflair':[],'meant12':[],'meanflair':[],'pt':[]}}}
        }
        # currently hard-coded for two studies
        self.blastdata = {0:copy.deepcopy(self.blastdatadict),1:copy.deepcopy(self.blastdatadict)}
        # time points to display (index of temporal order, ie '1' is recent/current and displayed on left panel)
        self.timepoints = [1,0]

        # affine is now a key in the main data structure so this attribute isn't needed
        # affine is no longer used though, so can be removed entirely
        self.affine = {'t1pre':None,'t1':None,'t2':None,'flair':None}
        # a list of ROI's for each study. '0' is dummy value so the Roi indexing ends up as 1-based. hard-coded for two studies
        # in addition, tally blast and sam rois separately.
        roidefault = {0:[0],1:[0]} 
        self.rois = {'blast':copy.deepcopy(roidefault),'sam':copy.deepcopy(roidefault)}
        self.roi = self.rois['blast']
        self.currentroi = 0 # tracks the currentroi widget variable
        # container for point selection of BLAST raw seg. hard-coded for two studies
        self.pt = {0:[],1:[]}
        self.currentpt = 0 # tracks the current point

        self.OS = sys.platform
        self.tstart = time.time()

        self.message = tk.StringVar(value='')
        self.debug = debug

        self.createGeneralLayout()
        self.caseframe.datadir.set(os.path.join(self.config.UIlocaldir))
        self.caseframe.datadirentry_callback()

        # SAM for inferences
        self.sam = SAM(remote=self.config.AWS,ui=self)

        # hard-coded entries for debugging
        if self.debug:

            # load a nifti case for BLAST and create a ROI
            if False:
                if True:
                    caseselect = 'M00001'
                    caseslice = 85
                    pointxyz = (165,129,85)
                    ETt1set = 0.2
                    ETflairset = -0.6
                    WTt1set = 0.0
                    WTflairset = 1.0
                else:
                    caseselect = 'M0002'
                    caseslice = 122
                    pointxyz = (80,145,122)
                    flairset = 1.4
                self.caseframe.casename.set(caseselect)
                self.caseframe.case_callback()
                self.function.set('SAM')
                self.function_callback(update=True)
                if False:
                    self.sliceviewerframe.currentslice.set(caseslice)
                    self.sliceviewerframe.normalslice_callback()
                    self.roiframe.thresholds['ET']['t12'].set(ETt1set)
                    self.roiframe.thresholds['ET']['flair'].set(ETflairset)
                    self.roiframe.updateslider('ET','t12')
                    self.roiframe.updateslider('ET','flair')
                if False:
                    self.roiframe.layer_callback(layer='T2 hyper')
                    self.roiframe.thresholds['T2 hyper']['t12'].set(WTt1set)
                    self.roiframe.thresholds['T2 hyper']['flair'].set(WTflairset)
                    self.roiframe.updateslider('T2 hyper','t12')
                    self.roiframe.updateslider('T2 hyper','flair')
                if False:
                    self.roiframe.createROI(pointxyz[0],pointxyz[1],pointxyz[2])
                    self.roiframe.ROIclick(event=None)

            # load a nifti case for BLAST 
            if True:
                debugdir = os.path.join(os.path.expanduser('~'),'data','brats2024_nifti')
                self.caseframe.datadir.set(debugdir)
                self.caseframe.datadirentry_callback()
                caseselect = 'M00002'
                # caseslice = 75
                self.caseframe.casename.set(caseselect)
                self.caseframe.case_callback()
                # self.sliceviewerframe.currentslice.set(caseslice)
                self.caseframe.s.current(0)
                self.caseframe.study_callback()
                self.set_studynumber()


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

        # slice viewer frames, roi frames 
        self.sliceviewerframes = {'BLAST':None,'4panel':None,'overlay':None}
        self.roiframes = {'BLAST':None,'4panel':None,'overlay':None}
        if self.function.get() == '4panel':
            self.sliceviewerframes['4panel'] = Create4PanelSVFrame(self.mainframe,ui=self,padding='0')
            self.roiframes['4panel'] = Create4PanelROIFrame(self.mainframe,ui=self,padding='0')
        elif self.function.get() == 'BLAST':
            self.sliceviewerframes['BLAST'] = CreateBlastSVFrame(self.mainframe,ui=self,padding='0')
            self.roiframes['BLAST'] = CreateSAMROIFrame(self.mainframe,ui=self,padding='0')
        elif self.function.get() == 'overlay':
            self.sliceviewerframes['overlay'] = CreateOverlaySVFrame(self.mainframe,ui=self,padding='0')
            self.roiframes['overlay'] = CreateOverlayFrame(self.mainframe,ui=self,padding='0')
        elif self.function.get() == 'SAM':
            self.sliceviewerframes['SAM'] = CreateSAMSVFrame(self.mainframe,ui=self,padding='0')
            self.roiframes['SAM'] = CreateSAMROIFrame(self.mainframe,ui=self,padding='0')

        
        # overlay/blast function mode
        self.functionmenu = ttk.OptionMenu(self.mainframe,self.function,self.functionlist['overlay'],
                                        *self.functionlist,command=Command(self.function_callback,update=True))
        if False: # not for general use
            self.functionmenu.grid(row=0,column=1,sticky='e')
        self.function_callback()

        # initialize default directory. no longer needed?
        if False:
            self.caseframe.datadirentry_callback()

        for row_num in range(self.mainframe.grid_size()[1]):
            if row_num == 1:
                self.mainframe.rowconfigure(row_num,weight=1)
            else:
                self.mainframe.rowconfigure(row_num,weight=0)
        self.mainframe.columnconfigure(0,minsize=self.caseframe.frame.winfo_width(),weight=1)
        for sv in self.sliceviewerframes.values():
            if sv is not None:
                self.mainframe.bind('<Configure>',sv.resizer)
        self.mainframe.update()

        # resize root window according to frames
        self.sliceviewerframe.resize()

    # switching mode between BLAST segmentation, overlay, 4panel
    def function_callback(self,event=None,update=False):
        f = self.function.get()
        self.set_frame(self.sliceviewerframes[f],frame='normal_frame')
        self.sliceviewerframe = self.sliceviewerframes[f]
        self.sliceviewerframe.frame.lift()
        self.set_frame(self.roiframes[f])
        self.roiframe = self.roiframes[f]
        # state of current data selection whether overlay or base
        # for now just revert to a base display
        self.dataselection = 'raw'
        self.chselection = self.config.DefaultChannel
        if update:
            self.sliceviewerframe.updateslice()
        # if blast or 4panel, activate study menu
        if f == 'BLAST' or f == '4panel' or f == 'SAM':
            self.caseframe.s.configure(state='enable')
        else:
            self.caseframe.s.configure(state='disabled')
        return
    


    ##############
    # BLAST method
    ##############

    def runblast(self,currentslice=None,layer=None,showblast=False):
        if currentslice: # 2d in a single slice
            currentslice=None # for now will run full 3d by default every update
        else: # entire volume
            self.root.config(cursor='watch')
            self.root.update_idletasks()

        # current study
        s = self.caseframe.s.current()

        if layer is None:   
            layer = self.roiframe.roioverlayframe.layer.get()
        clustersize = self.get_bcsize(layer=layer)
        t12_threshold = self.roiframe.sliderframe.sliders[layer]['t12'].get()
        flair_threshold = self.roiframe.sliderframe.sliders[layer]['flair'].get()

        try:
            retval = Blastbratsv3.run_blast(
                                self.data[s],
                                self.blastdata[s],
                                t12_threshold,
                                flair_threshold,
                                clustersize,layer,
                                currentslice=currentslice
                                )
            if retval is not None:
                self.blastdata[s]['blast'][layer],self.blastdata[s]['blast']['gates']['brain '+layer],self.blastdata[s]['blast']['gates'][layer] = retval
                self.update_blast(layer=layer)
        except ValueError as e:
            self.set_message(e)

        chlist = [self.chselection]
        if self.function.get() == 'BLAST':
            chlist.append('flair')

        for ch in chlist:
            self.data[s].dset['seg_raw_fusion'][ch]['d'+layer] = generate_blast_overlay(self.data[s].dset['raw'][ch]['d'],
                                                        self.data[s].dset['seg_raw'][self.chselection]['d'],
                                                        layer=self.roiframe.roioverlayframe.layer.get(),
                                                        overlay_intensity=self.config.OverlayIntensity)
            self.data[s].dset['seg_raw_fusion'][ch]['ex'] = True            
                
        # in the SAM viewer, may not need to see the raw BLAST overlay anymore
        if showblast:
            if self.roiframe.roioverlayframe.overlay_value['finalROI'].get() == True:
                self.dataselection = 'seg_fusion'
            else:
                self.dataselection = 'seg_raw_fusion'
            if currentslice is None:
                self.updateslice(wl=True,layer=layer)
            else:
                self.updateslice()
        else:
            # in the new workflow, once a SAM roi exists keep the SAM roi displayed at all times
            if self.roiframe.roioverlayframe.overlay_value['SAM'].get():
                self.updateslice()
            else:
                self.roiframe.roioverlayframe.set_overlay()
                self.dataselection = 'raw'
                self.updateslice()
                
        self.root.config(cursor='arrow')
        self.root.update_idletasks()

        return None


    #############################
    ###### Utility methods ######
    #############################

    # raise frame to top and blank other frames with dummy
    def set_frame(self,frameobj,frame='frame'):
        frameobj.dummy_frame.lift()
        getattr(frameobj,frame).lift()

    def set_currentslice(self,val=None):
        if val is not None:
            self.sliceviewerframe.currentslice.set(val)
        self.currentslice = self.sliceviewerframe.currentslice.get()

    def get_currentslice(self,ax=None):
        if ax == 'ax' or ax is None:
            return self.sliceviewerframe.currentslice.get()
        elif ax == 'sag':
            return self.sliceviewerframe.currentsagslice.get()
        elif ax == 'cor':
            return self.sliceviewerframe.currentcorslice.get()
    
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
        # if self.currentroi > 0:
        #     self.roiframe.roinumber.configure(state='active')

    def get_currentroi(self):
        return self.currentroi
    
    def set_currentpt(self,val=None):
        self.currentpt = self.roiframe.blastpointframe.currentpt.get()
 
    def get_currentpt(self):
        return self.currentpt

    def get_bcsize(self,layer=None):
        if layer is None:
            layer = self.roiframe.roioverlayframe.layer.get()
        return self.roiframe.sliderframe.thresholds[layer]['bc'].get()
    
    def update_roidata(self):
        self.roiframe.updateROIData()

    def update_blast(self,**kwargs):
        self.roiframe.updateBLAST(**kwargs)

    def clear_message(self):
        self.sliceviewerframe.messagelabel['text'] = ''
        self.message.set('')

    def set_message(self, msg=None):
        if msg is None:
            msg = self.message.get()
        else:
            self.message.set(msg)
        self.sliceviewerframe.messagelabel['text'] = msg
        self.root.update_idletasks()

    # not sure if this will be needed
    def set_dataselection(self,s=None):
        if s is not None:
            self.dataselection = s
        else:
            self.dataselection = 'raw'

    def set_chselection(self):
        self.chselection = self.sliceviewerframe.chdisplay.get()

    def set_studynumber(self,val=None):
        self.s = self.caseframe.s.current()

    def get_studynumber(self):
        return self.s

    def reset_roi(self):
        self.currentroi = 0
        roidefault = {0:[0],1:[0]} # dummy value for Roi indexing 1-based
        self.rois = {'blast':copy.deepcopy(roidefault),'sam':copy.deepcopy(roidefault)}
        self.roi = self.rois['blast']

    def reset_pt(self):
        self.currentpt = 0
        self.roiframe.blastpointframe.currentpt.set(0)
        self.pt = {0:[],1:[]}

    def resetUI(self):
        self.normalslice = None
        self.currentslice = None
        self.dataselection = 'raw'
        self.chselection = self.config.DefaultChannel

        self.data = {}
        self.blastdata = {0:copy.deepcopy(self.blastdatadict),1:copy.deepcopy(self.blastdatadict)}
    
        roidefault = {0:[0],1:[0]} # dummy value for Roi indexing 1-based
        self.rois = {'blast':copy.deepcopy(roidefault),'sam':copy.deepcopy(roidefault)}
        self.roi = self.rois['blast']
        self.currentroi = 0 # tracks the currentroi widget variable
        self.tstart = time.time()
        self.message = tk.StringVar(value='')
        if self.roiframe is not None:
            self.roiframe.resetROI(data=False)

    def get_monitor_from_coord(self,x, y):
        monitors = screeninfo.get_monitors()
        for m in reversed(monitors):
            if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
                return m
        return monitors[0]
