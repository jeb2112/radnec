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
from src.CreateOverlaySVFrame import CreateOverlaySVFrame
from src.CreateBlastSVFrame import CreateBlastSVFrame
from src.CreateCaseFrame import CreateCaseFrame
from src.CreateROIFrame import CreateROIFrame
from src.CreateOverlayFrame import CreateOverlayFrame
from src.CreateFrame import Command
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

        self.normalslice = None # for 2d normal stats, probably not needed anymore
        self.currentslice = None # tracks the current slice widget variable
        self.s = 0 # convenience attribute, tracks the current study number in caseframe widget
        # top level of data structure, could be raw data or overlay image
        self.dataselection = 'raw'
        # 2nd level of data structure is image channel, either as raw data or underlay data
        self.chselection = 't1+' 

        # viewer functions. overlay mode, or BLAST segmentation mode
        self.functionlist = {'overlay':0,'BLAST':1}
        self.function = tk.StringVar(value='overlay')

        # data structure. data is a dict of studies. see DcmCase
        self.data = {} 
        # additional data related to BLAST, per study
        self.blastdatadict = {'blast':{'gates':{'ET':None,'T2 hyper':None,'brain ET':None,'brain T2 hyper':None},
                            'T2 hyper':None,
                            'ET':None,
                            'params':{'ET':{'t12':0.0,'bc':0.0,'flair':0.0,'stdt12':1,'stdflair':1,'meant12':1,'meanflair':1},
                               'T2 hyper':{'t12':0.0,'bc':0.0,'flair':0.0,'stdt12':1,'stdflair':1,'meant12':1,'meanflair':1},
                               },
                    },
        }
        # currently hard-coded for two studies
        self.blastdata = {0:copy.deepcopy(self.blastdatadict),1:copy.deepcopy(self.blastdatadict)}
        # time points to display (index of temporal order, ie '1' is recent/current and displayed on left panel)
        self.timepoints = [1,0]

        # affine is now a key in the main data structure so this attribute isn't needed
        # affine is no longer used though, so can be removed entirely
        self.affine = {'t1pre':None,'t1':None,'t2':None,'flair':None}
        self.roi = {0:[0],1:[0]} # a list of ROI's for each study. '0' is dummy value so the Roi indexing ends up as 1-based. hard-coded for two studies
        self.currentroi = 0 # tracks the currentroi widget variable
        self.OS = sys.platform
        self.tstart = time.time()

        self.message = tk.StringVar(value='')
        self.debug = debug

        self.createGeneralLayout()

        # hard-coded entries for debugging
        if self.debug:
            # load a nifti case for BLAST and create a ROI
            if True:
                self.caseframe.datadir.set('/media/jbishop/WD4/brainmets/sunnybrook/radnec/dicom2nifti/M0001')
                self.caseframe.datadirentry_callback()
                self.caseframe.casename.set('M0001')
                self.caseframe.case_callback()
                self.roiframe.overlay_value.set(True)
                self.roiframe.overlay_type.set('z')
                self.roiframe.overlay_callback()
                self.function.set('BLAST')
                self.function_callback(update=True)
                self.sliceviewerframe.normalslice_callback()
                self.sliceviewerframe.currentslice.set(55)
                self.roiframe.thresholds['T2 hyper']['flair'].set(1.2)
                self.roiframe.updateslider('T2 hyper','flair')
                if True:
                    self.roiframe.createROI(65,65,55) # case M00001
                    self.roiframe.ROIclick(event=None)


            # load a tempo case
            if False:
                self.caseframe.datadir.set('/media/jbishop/WD4/brainmets/sunnybrook/radnec/dicom2nifti/M0001')
                self.caseframe.datadirentry_callback()
                self.caseframe.casename.set('M0001')
                self.caseframe.case_callback()
                self.roiframe.overlay_value.set(True)                
                self.roiframe.overlay_type.set('tempo')
                self.roiframe.mask_value.set(True)
                self.sliceviewerframe.currentslice.set(55)
                self.roiframe.overlay_callback()



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
        self.sliceviewerframes = {}
        self.sliceviewerframes['BLAST'] = CreateBlastSVFrame(self.mainframe,ui=self,padding='0')
        self.sliceviewerframes['overlay'] = CreateOverlaySVFrame(self.mainframe,ui=self,padding='0')

        # blast/overlay functions
        self.roiframes = {}
        self.roiframes['BLAST'] = CreateROIFrame(self.mainframe,ui=self,padding='0')
        self.roiframes['overlay'] = CreateOverlayFrame(self.mainframe,ui=self,padding='0')
        
        # overlay/blast function mode
        self.functionmenu = ttk.OptionMenu(self.mainframe,self.function,self.functionlist['overlay'],
                                        *self.functionlist,command=Command(self.function_callback,update=True))
        self.functionmenu.grid(row=0,column=4,sticky='e')
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
            self.mainframe.bind('<Configure>',sv.resizer)
        self.mainframe.update()

    # switching mode between BLAST segmentation and overlay
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
        self.chselection = 't1+'
        if update:
            self.sliceviewerframe.updateslice()
        # if blast, activate study menu
        if f == 'BLAST':
            self.caseframe.s.configure(state='enable')
        else:
            self.caseframe.s.configure(state='disabled')
        return
    


    ##############
    # BLAST method
    ##############

    def runblast(self,currentslice=None,layer=None):
        if currentslice: # 2d in a single slice
            currentslice=None # for now will run full 3d by default every update
        else: # entire volume
            self.root.config(cursor='watch')
            self.root.update_idletasks()

        # current study
        s = self.caseframe.s.current()

        if layer is None:   
            layer = self.roiframe.layer.get()
        clustersize = self.get_bcsize(layer=layer)
        t12_threshold = self.roiframe.sliders[layer]['t12'].get()
        flair_threshold = self.roiframe.sliders[layer]['flair'].get()

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

        # if self.config.WLClip:
        #     self.sliceviewerframe.restorewl_raw()
        #     self.sliceviewerframe.window = np.array([1.,1.],dtype='float')
        #     self.sliceviewerframe.level = np.array([0.5,0.5],dtype='float')
        #     self.updateslice()

        chlist = [self.chselection]
        if self.function.get() == 'BLAST':
            chlist.append('flair')

        for ch in chlist:
            self.data[s].dset['seg_raw_fusion'][ch]['d'+layer] = generate_blast_overlay(self.data[s].dset['raw'][ch]['d'],
                                                        self.data[s].dset['seg_raw'][self.chselection]['d'],layer=self.roiframe.layer.get(),
                                                        overlay_intensity=self.config.OverlayIntensity)
            self.data[s].dset['seg_raw_fusion'][ch]['ex'] = True
            self.data[s].dset['seg_raw_fusion_d'][ch]['d'+layer] = copy.deepcopy(self.data[s].dset['seg_raw_fusion'][ch]['d'+layer])
            
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

    # raise frame to top and blank other frames with dummy
    def set_frame(self,frameobj,frame='frame'):
        frameobj.dummy_frame.lift()
        getattr(frameobj,frame).lift()
        
    # this approach didn't work can be discarded
    if False:
        def set_sliceviewerframe(self,frameobj,frame='frame',above=None,below=None):
            frameobj.frame.lift()
            frameobj.dummy_frame.lower(belowThis=frameobj.frame)
            self.sliceviewerframe = frameobj
        def set_roiframe(self,frameobj,frame='frame',above=None,below=None):
            frameobj.frame.lift()
            frameobj.dummy_frame.lower(belowThis=frameobj.frame)
            self.roiframe = frameobj

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

    def resetUI(self):
        self.normalslice = None
        self.currentslice = None
        self.dataselection = 'raw'
        self.chselection = 't1+'

        self.data = {}
        self.blastdata = {0:copy.deepcopy(self.blastdatadict),1:copy.deepcopy(self.blastdatadict)}
    
        # self.roi = [0] # dummy value for Roi indexing 1-based
        self.roi = {0:[0],1:[0]} # dummy value for Roi indexing 1-based
        self.currentroi = 0 # tracks the currentroi widget variable
        self.tstart = time.time()
        self.message = tk.StringVar(value='')
        if self.roiframe is not None:
            self.roiframe.resetROI()

    def get_monitor_from_coord(self,x, y):
        monitors = screeninfo.get_monitors()
        for m in reversed(monitors):
            if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
                return m
        return monitors[0]
