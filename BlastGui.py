import os,sys
import copy
from tkinter import messagebox,ttk,PhotoImage
from cProfile import Profile
from pstats import SortKey,Stats
from collections import defaultdict

import Config
import UIActions
import Blastbratsv3
from CreateFrame import CreateCaseFrame,CreateSliceViewerFrame
from CreateROIFrame import CreateROIFrame
from OverlayPlots import *
from ROI import ROI

# main gui class
class BlastGui(object):
    def __init__(self, root, toolsFlag, debug=True):
        self.root = root
        self.toolsFlag = toolsFlag
        self.titletext = 'BLAST User Interface'
        self.root.title(self.titletext)
        self.config = Config.Config()
        self.logoImg = os.path.join(self.config.UIResourcesPath,'sunnybrook.png')
        self.blastImage = PhotoImage(file=self.logoImg)
        self.uiactions = UIActions.UIActions(self)
        self.normalslice = None
        self.currentslice = None
        self.dataselection = 'raw'

        self.data = {'gates':[None,None,None], 'wt':None,'et':None,'tc':None}
        # self.data = {}
        self.data['params'] = {'stdt1':1,'stdt2':1,'meant1':1,'meant2':1}

        self.roi = []
        self.currentroi = -1
        self.currentlayer = 0
        self.OS = sys.platform
        # ROI stats
        self.stats = nested_dict()


        self.createGeneralLayout()

        # hard-coded for debugging
        if debug:
            self.caseframe.datadirentry_callback()
            self.set_currentslice(105)
            self.casename = '00000'
            self.caseframe.loadCase()
            self.roiframe.normalslice_callback()
            self.set_currentslice(75)
            self.updateslice()
            self.roiframe.createROI(132,102,75)
            self.roiframe.ROIclick(event=None)
            self.roiframe.updateROI()
            self.roiframe.finalROI_overlay_value.set(True)
            self.roiframe.update_layermenu_options('seg')
            self.roiframe.enhancingROI_overlay_value.set(False)
            self.roiframe.createROI(141,150,75)
            self.roiframe.ROIclick(event=None)
            self.roiframe.updateROI()
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
        self.mainframe.grid(column=0, row=0, sticky='NEW')
        self.mainframe.columnconfigure(0,weight=1)
        # self.mainframe.columnconfigure(1,weight=1)

        # slice viewer frame
        self.sliceviewerframe = CreateSliceViewerFrame(self.mainframe,ui=self,padding='10')        

        # create case frame
        self.caseframe = CreateCaseFrame(self.sliceviewerframe.frame,ui=self)

        # roi functions
        self.roiframe = CreateROIFrame(self.sliceviewerframe.frame,ui=self,padding='0')

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
            self.data['seg_raw'],self.data['gates'] = Blastbratsv3.run_blast(
                                self.data,self.roiframe.t1slider.get(),
                                self.roiframe.t2slider.get(),self.roiframe.bcslider.get(),
                                currentslice=currentslice)
        self.data['seg_raw_fusion'] = generate_overlay(self.data['raw'],self.data['seg_raw'],self.roiframe.layer.get())
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

    def set_currentslice(self,val):
        self.currentslice = val

    def get_currentslice(self):
        return self.currentslice
    
    def set_casename(self,val):
        self.casename = val

    def get_casename(self):
        return self.casename
    
    def updateslice(self,event=None,**kwargs):
        self.sliceviewerframe.updateslice(event,**kwargs)

    # def updateslice(self,event=None):
    #     self.sliceviewerframe.updateslice(event)

    def clearMsg(self):
        self.uiactions.setStatusMessage('')


    ### stats ###
    #  @param self The UI object
    #  @param analysis The type of analysis tool to run, allowed values = ['SNR', 'RMS', 'Distortion', 'Ghosting']
    def compute(self, analysis='stats'):
        try:
            activeElem = self.rxinfo.seqList.curInd
            #     scriptpath = Config.ScriptPath
            #     if scriptpath not in sys.path:
            #         sys.path.append(scriptpath)

            imfile = os.path.join(Config.TempDataStorePath, self.uiactions.getCurrentImageFile())
            if analysis == 'stats':
                # import TemporalSNR
                # app = TemporalSNR.TemporalSNR(imfile)
                # result = app.run()
                result = 'Test'
                msg = 'Stats Results = {}'.format(result)
            messagebox.showinfo(title="{}".format(analysis), message=msg)

        except Exception as e:
            self.uiactions.setCommunicationMessage("Error running stats procedure")

    def SendMessage(self, msg=None):
        if msg is not None:
            self.actions.sendMessage(msg)


def nested_dict():
    return defaultdict(nested_dict)
