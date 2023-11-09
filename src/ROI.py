import os,sys
import copy
from tkinter import messagebox,ttk,PhotoImage
from cProfile import Profile
from pstats import SortKey,Stats

class ROI():
    def __init__(self,xpos,ypos,slice):
        self.data = {'wt':None,'et':None,'tc':None}
        self.data['params'] = {'stdt1':1,'stdt2':1,'meant1':1,'meant2':1}
        self.casename = None

        # ROI selection coordinates
        self.x = xpos
        self.y = ypos
        self.slice = slice

        # threshold gates saved as intermediate values
        self.brain = None
        self.et_gate = None
        self.wt_gate = None


        self.stats = {'spec':{'et':0,'tc':0,'wt':0},
            'sens':{'et':0,'tc':0,'wt':0},
            'dsc':{'et':0,'tc':0,'wt':0},
            'vol':{'et':0,'tc':0,'wt':0,'manual_et':0,'manual_tc':0,'manual_wt':0},
            'gatecount':{'t1':0,'t2':0},
            'elapsed_time':0}
