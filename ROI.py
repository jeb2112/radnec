import os,sys
import copy
from tkinter import messagebox,ttk,PhotoImage
from cProfile import Profile
from pstats import SortKey,Stats

import Config

class ROI():
    def __init__(self,xpos,ypos,slice):
        self.data = {'wt':None,'et':None,'tc':None}
        self.data['params'] = {'stdt1':1,'stdt2':1,'meant1':1,'meant2':1}
        self.casename = None
        # ROI selection coordinates
        self.x = xpos
        self.y = ypos
        self.slice = slice

        self.brain = None
        self.et_gate = None
        self.wt_gate = None

