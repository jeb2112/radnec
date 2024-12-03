import os,sys
import re
import numpy as np
import pickle
import copy

import tkinter as tk
from tkinter import ttk
import matplotlib
import matplotlib.pyplot as plt

from src.CreateFrame import CreateFrame,Command
from src.roi.ROI import ROIBLAST,ROISAM,ROIPoint

#################################################
# frame layout for setting BLAST params by slider
#################################################

class CreateROISliderFrame(CreateFrame):
    def __init__(self,frame,ui=None,padding='10'):
        super().__init__(frame,ui=ui,padding=padding)

        roidict = {'ET':{'t12':None,'flair':None,'bc':None},'T2 hyper':{'t12':None,'flair':None,'bc':None}}
        self.thresholds = copy.deepcopy(roidict)
        self.sliders = copy.deepcopy(roidict)
        self.sliderlabels = copy.deepcopy(roidict)
        self.thresholds['ET']['t12'] = tk.DoubleVar(value=self.ui.config.T1default)
        self.thresholds['T2 hyper']['t12'] = tk.DoubleVar(value=self.ui.config.T2default)
        self.thresholds['ET']['flair'] = tk.DoubleVar(value=self.ui.config.T2default)
        self.thresholds['T2 hyper']['flair'] = tk.DoubleVar(value=self.ui.config.T2default)
        self.thresholds['ET']['bc'] = tk.DoubleVar(value=self.ui.config.BCdefault[0])
        self.thresholds['T2 hyper']['bc'] = tk.DoubleVar(value=self.ui.config.BCdefault[1])
        
        ########################
        # layout for the sliders
        ########################

        self.s = ttk.Style()
        self.s.configure('debugframe.TFrame',background='green')
        # frames for sliders
        self.sliderframe = {}
        # dummy frame to hide slider bars
        # slider bars might not be used in the SAM viewer anymore
        self.sliderframe['dummy'] = ttk.Frame(self.frame,padding='0')
        self.sliderframe['dummy'].grid(row=3,column=2,columnspan=7,sticky='nesw')

        self.sliderframe['ET'] = ttk.Frame(self.frame,padding='0')
        self.sliderframe['ET'].grid(row=3,column=2,columnspan=7,sticky='e')
        self.sliderframe['T2 hyper'] = ttk.Frame(self.frame,padding='0')
        self.sliderframe['T2 hyper'].grid(row=3,column=2,columnspan=7,sticky='e')

        # ET sliders
        # t1 slider
        t1label = ttk.Label(self.sliderframe['ET'], text='T1/T2')
        t1label.grid(column=0,row=0,sticky='w')

        self.sliders['ET']['t12'] = ttk.Scale(self.sliderframe['ET'],from_=-4,to=4,variable=self.thresholds['ET']['t12'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'ET','t12'),orient='horizontal')
        self.sliders['ET']['t12'].grid(row=0,column=1,sticky='e')
        self.sliderlabels['ET']['t12'] = ttk.Label(self.sliderframe['ET'],text=self.thresholds['ET']['t12'].get())
        self.sliderlabels['ET']['t12'].grid(row=0,column=2,sticky='e')

        #flairt1 slider
        flairt1label = ttk.Label(self.sliderframe['ET'], text='flair')
        flairt1label.grid(row=1,column=0,sticky='w')
        self.sliders['ET']['flair'] = ttk.Scale(self.sliderframe['ET'],from_=-4,to=4,variable=self.thresholds['ET']['flair'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'ET','flair'),orient='horizontal')
        self.sliders['ET']['flair'].grid(row=1,column=1,sticky='e')
        self.sliderlabels['ET']['flair'] = ttk.Label(self.sliderframe['ET'],text=self.thresholds['ET']['flair'].get())
        self.sliderlabels['ET']['flair'].grid(row=1,column=2,sticky='e')

        #braint1 cluster slider
        bclabel = ttk.Label(self.sliderframe['ET'],text='b.c.')
        bclabel.grid(row=2,column=0,sticky='w')
        self.sliders['ET']['bc'] = ttk.Scale(self.sliderframe['ET'],from_=0,to=4,variable=self.thresholds['ET']['bc'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'ET','bc'),orient='horizontal')
        self.sliders['ET']['bc'].grid(row=2,column=1,sticky='e')
        self.sliderlabels['ET']['bc'] = ttk.Label(self.sliderframe['ET'],text=self.thresholds['ET']['bc'].get())
        self.sliderlabels['ET']['bc'].grid(row=2,column=2,sticky='e')

        # T2 hyper sliders
        # t2 slider
        t2label = ttk.Label(self.sliderframe['T2 hyper'], text='T1/T2')
        t2label.grid(row=0,column=0,sticky='w')
        self.sliders['T2 hyper']['t12'] = ttk.Scale(self.sliderframe['T2 hyper'],from_=-4,to=4,variable=self.thresholds['T2 hyper']['t12'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'T2 hyper','t12'),orient='horizontal')
        self.sliders['T2 hyper']['t12'].grid(row=0,column=1,sticky='e')
        self.sliderlabels['T2 hyper']['t12'] = ttk.Label(self.sliderframe['T2 hyper'],text=self.thresholds['T2 hyper']['t12'].get())
        self.sliderlabels['T2 hyper']['t12'].grid(row=0,column=2,sticky='e')

        #flairt2 slider
        flairt2label = ttk.Label(self.sliderframe['T2 hyper'], text='flair')
        flairt2label.grid(row=1,column=0,sticky='w')
        self.sliders['T2 hyper']['flair'] = ttk.Scale(self.sliderframe['T2 hyper'],from_=-4,to=4,variable=self.thresholds['T2 hyper']['flair'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'T2 hyper','flair'),orient='horizontal')
        self.sliders['T2 hyper']['flair'].grid(row=1,column=1,sticky='e')
        self.sliderlabels['T2 hyper']['flair'] = ttk.Label(self.sliderframe['T2 hyper'],text=self.thresholds['T2 hyper']['flair'].get())
        self.sliderlabels['T2 hyper']['flair'].grid(row=1,column=2,sticky='e')

        #braint2 cluster slider
        bclabel = ttk.Label(self.sliderframe['T2 hyper'],text='b.c.')
        bclabel.grid(row=2,column=0,sticky='w')
        self.sliders['T2 hyper']['bc'] = ttk.Scale(self.sliderframe['T2 hyper'],from_=0,to=4,variable=self.thresholds['T2 hyper']['bc'],state='disabled',
                                  length='3i',command=Command(self.updatesliderlabel,'T2 hyper','bc'),orient='horizontal')
        self.sliders['T2 hyper']['bc'].grid(row=2,column=1,sticky='e')
        self.sliderlabels['T2 hyper']['bc'] = ttk.Label(self.sliderframe['T2 hyper'],text=self.thresholds['T2 hyper']['bc'].get())
        self.sliderlabels['T2 hyper']['bc'].grid(row=2,column=2,sticky='e')


    ###############################################
    # callbacks for the BLAST threshold slider bars
    ###############################################

    def updateslider(self,layer,slider,event=None,doblast=True):
        self.overlay_value['BLAST'].set(True)
        if self.overlay_value['finalROI'].get() == True:
            self.overlay_value['finalROI'].set(False)
            self.enhancingROI_overlay_callback()
        # layer = self.layer.get()
        self.updatesliderlabel(layer,slider)

        # updates to blastdata
        if slider == 'bc':
            self.ui.blastdata[self.ui.s]['blast']['gates']['brain '+layer] = None
        self.ui.blastdata[self.ui.s]['blast']['gates'][layer] = None
        self.ui.update_blast(layer=layer)

        # rerun blast with new value
        if doblast:
            self.ui.runblast(currentslice=True)

    def updatesliderlabel(self,layer,slider):
        # if 'T2 hyper' in self.sliderlabels.keys() and 'T2 hyper' in self.sliders.keys():
        try:
            self.sliderlabels[layer][slider]['text'] = '{:.1f}'.format(self.sliders[layer][slider].get())
        except KeyError as e:
            print(e)

    # switch to show sliders and values according to current layer being displayed
    def updatesliders(self):
        if self.overlay_value['SAM'].get() == True:
            return
        if self.overlay_value['BLAST'].get() == True:
            layer = self.layer.get()
        elif self.overlay_value['finalROI'].get() == True:
            # ie display slider values that were used for current ROI
            layer = self.layerROI.get()
            if layer == 'WT':
                layer = 'T2 hyper'
            else:
                layer = 'ET'
        for sl in ['t12','flair','bc']:
            self.thresholds[layer][sl].set(self.ui.blastdata[self.ui.s]['blast']['params'][layer][sl])
            self.updatesliderlabel(layer,sl)