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
        self.normalslice = None
        self.currentslice = None # tracks the current slice widget variable
        self.dataselection = 'raw'
        self.chselection = 't1+' # current display, could be base image or overlay image
        self.base = 't1+' # tracks chdisplay variable in sliceviewer. not fully implemented yet

        # function
        self.functionlist = {'overlay':0,'BLAST':1}
        self.function = tk.StringVar(value='overlay')

        # data is a dict of studies
        self.data = {} 
        # time points to display (index of temporal order)
        self.timepoints = [1,0]

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

            # load a nifti case
            if True:
                self.caseframe.datadir.set('/media/jbishop/WD4/brainmets/sunnybrook/radnec/dicom2nifti/M0001')
                self.caseframe.datadirentry_callback()
                self.caseframe.casename.set('M0001')
                self.caseframe.case_callback()
                self.roiframe.overlay_value.set(True)
                self.roiframe.overlaytype.set('z')
                self.roiframe.overlay_callback()
                self.function.set('BLAST')
                self.function_callback(update=True)
                self.sliceviewerframe.normalslice_callback()
                self.sliceviewerframe.currentslice.set(55)
                # self.roiframe.thresholds['ET']['flair'].set(1.1)
                # self.roiframe.thresholds['T2 hyper']['flair'].set(1.1)
                # self.roiframe.createROI(75,75,55) # 00002
                # self.roiframe.ROIclick(event=None)

            # 00005 75,105
            # 00002 53,81
            if False:
                self.caseframe.n4_check_value.set(0)
                self.caseframe.casename.set('00006')
                self.caseframe.case_callback()
                if self.sliceviewerframe.slicevolume_norm.get() == 0:
                    self.sliceviewerframe.currentslice.set(53)
                    self.updateslice()
                    self.sliceviewerframe.normalslice_callback()
                self.sliceviewerframe.currentslice.set(81)
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
                self.roiframe.overlay_value.set(False)
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
        self.sliceviewerframes = {}
        self.sliceviewerframes['BLAST'] = CreateBlastSVFrame(self.mainframe,ui=self,padding='0')
        self.sliceviewerframes['overlay'] = CreateOverlaySVFrame(self.mainframe,ui=self,padding='0')
        # self.set_frame(self.sliceviewerframes['overlay'],frame='normal_frame')

        # blast/overlay functions
        self.roiframes = {}
        self.roiframes['BLAST'] = CreateROIFrame(self.mainframe,ui=self,padding='0')
        self.roiframes['overlay'] = CreateOverlayFrame(self.mainframe,ui=self,padding='0')
        # self.set_frame(self.roiframes['overlay'])
        
        # overlay/blast function mode
        self.functionmenu = ttk.OptionMenu(self.mainframe,self.function,self.functionlist['overlay'],
                                        *self.functionlist,command=Command(self.function_callback,update=True))
        self.functionmenu.grid(row=0,column=4,sticky='e')
        self.function_callback()

        # initialize default directory.
        if False:
            self.caseframe.datadirentry_callback()

        for row_num in range(self.mainframe.grid_size()[1]):
            if row_num == 1:
                self.mainframe.rowconfigure(row_num,weight=1)
            else:
                self.mainframe.rowconfigure(row_num,weight=0)
        self.mainframe.columnconfigure(0,minsize=self.caseframe.frame.winfo_width(),weight=1)
        for s in self.sliceviewerframes.values():
            self.mainframe.bind('<Configure>',s.resizer)
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
                                    self.data[0],
                                    self.blastdata,
                                    t12_threshold,
                                    flair_threshold,
                                    clustersize,layer,
                                    currentslice=currentslice
                                    )
                if retval is not None:
                    self.blastdata['blast'][layer],self.blastdata['blast']['gates']['brain '+layer],self.blastdata['blast']['gates'][layer] = retval
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
            # seg_raw doesn't have 'flair' yet
            self.data[0].dset['seg_raw_fusion'][ch]['d'] = generate_blast_overlay(self.data[0].dset['raw'][ch]['d'],
                                                        self.data[0].dset['seg_raw'][self.chselection]['d'],layer=self.roiframe.layer.get(),
                                                        overlay_intensity=self.config.OverlayIntensity)
            self.data[0].dset['seg_raw_fusion'][ch]['ex'] = True
            self.data[0].dset['seg_raw_fusion_d'][ch]['d'] = copy.deepcopy(self.data[0].dset['seg_raw_fusion'][ch]['d'])
            
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

    def set_frame(self,frameobj,frame='frame'):
        frameobj.dummy_frame.lift()
        getattr(frameobj,frame).lift()
        # self.sliceviewerframe = frameobj
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
    def set_dataselection(self):
        self.dataselection = self.sliceviewerframe.chdisplay.get()

    def set_chselection(self):
        self.chselection = self.sliceviewerframe.chdisplay.get()

    def resetUI(self):
        self.normalslice = None
        self.currentslice = None
        self.dataselection = 'raw'
        self.chselection = 't1+'

        self.data = {}
        self.blastdata = {'blast':{'gates':{'ET':None,'T2 hyper':None,'brain ET':None,'brain T2 hyper':None},
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
        if self.roiframe is not None:
            self.roiframe.resetROI()

    def get_monitor_from_coord(self,x, y):
        monitors = screeninfo.get_monitors()
        for m in reversed(monitors):
            if m.x <= x <= m.width + m.x and m.y <= y <= m.height + m.y:
                return m
        return monitors[0]
