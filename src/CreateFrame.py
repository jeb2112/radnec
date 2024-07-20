import os,sys
import numpy as np
import glob
import copy
import re
import logging
import subprocess
import tkinter as tk
import nibabel as nb
from nibabel.processing import resample_from_to
import pydicom as pd
from pydicom.fileset import FileSet
from tkinter import ttk,StringVar,DoubleVar,PhotoImage
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from cProfile import Profile
from pstats import SortKey,Stats
from enum import Enum

matplotlib.use('TkAgg')
import SimpleITK as sitk
from sklearn.cluster import KMeans,MiniBatchKMeans,DBSCAN
from scipy.spatial.distance import dice

from src.NavigationBar import NavigationBar
from src.FileDialog import FileDialog

# convenience for indexing data dict
class data(Enum):
    T1 = 0
    FLAIR = 1
    T2 = 2
# utility class for callbacks with args
class Command():
    def __init__(self, callback, *args, **kwargs):
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def __call__(self,event=None):
        return self.callback(*self.args, **self.kwargs)
    
class EventCallback():
    def __init__(self, callback, *args, **kwargs):
        self.callback = callback
        self.args = args
        self.kwargs = kwargs

    def __call__(self,event):
        return self.callback(event,*self.args, **self.kwargs)

# base for various frames
class CreateFrame():
    def __init__(self,frame,ui=None,padding='10',style=None,gridparams=None):
        self.ui = ui
        self.parentframe = frame # parent
        self.frame = ttk.Frame(self.parentframe,padding=padding,style=style)
        self.config = self.ui.config
        self.padding = padding
        self.fstyle = ttk.Style()
        if gridparams is not None:
            self.frame.grid(**gridparams)

