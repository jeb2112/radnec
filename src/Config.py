import os, sys
import logging
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import numpy as np
import re

class Config(object):
    def __init__(self):
        
        ### Paths that tend to change based on local install

        if os.name == 'nt':
            self.InstallPath = os.path.join('C:\\','Users','Chris Heyn Lab','src','radnec')
        else:
            self.InstallPath = os.path.join('/home','src','radnec')

        # Path where R&D user interface built in Python is located
        self.UIPath = os.path.join(self.InstallPath, 'src')

        # Path to HD-bet code
        self.HDBETPath = os.path.join(os.path.expanduser('~'),'src','hd-bet')

        # Path for UI based resources (images, icons, etc)
        # use this hack for relative path for installation of a wheel distro into a virtual env
        self.UIResourcesPath = os.path.join(os.path.dirname(__file__), '..','resources')

        # Path to log files
        self.logger = logging.getLogger('radnec')
        self.logger.setLevel(logging.DEBUG)
        self.LogFileName = 'RADNEC_UI.log'
        if self.LogFileName is not None:
            self.LogFilePath = os.path.join(os.path.expanduser('~'), 'Documents', 'RADNEC', 'Log')
            if not os.path.exists(self.LogFilePath):
                os.makedirs(self.LogFilePath)
            filehandler = logging.FileHandler(os.path.join(self.LogFilePath,self.LogFileName),mode='w')
            self.logger.addHandler(filehandler)

        # Path to data files
        self.DataFilePath = os.path.join(os.path.expanduser('~'), 'Documents', 'RADNEC', 'Data')
        if not os.path.exists(self.LogFilePath):
            os.makedirs(self.LogFilePath)

        # Path to dataset images
        if os.name == 'nt':
            if True: # default True
                self.UIdatadir = os.path.join('C:\\','Users','chint','data')
                self.UIlocaldir = os.path.join('C:\\','Users','chint','data','radnec_sam')
            else:  # True for local debugging
                self.UIdatadir = os.path.join('C:\\','Users','Chris Heyn Lab','data')
                self.UIlocaldir = os.path.join('C:\\','Users','Chris Heyn Lab','data','dicom2nifti_sam')
        elif os.name == 'posix':
            self.UIdatadir = '/media/jbishop/WD4/brainmets/sunnybrook/metastases'
            self.UIlocaldir = '/media/jbishop/WD4/brainmets/sunnybrook/metastases/BraTS_2024'
            # self.UIdatadir = os.path.join(os.path.expanduser('~'),'Data','radnec')
        self.UIdataroot = 'BraTS2021_'

        # optional root filename for ground truth masks
        self.UIgroundtruth = re.compile('BraTS-MET-[0-9]{5}-000-seg.nii.gz')

        # pytorch env for SAM and nnUNet
        self.UIpytorch = 'pytorch_sam'
        # self.UIpytorch = 'pytorch118_310'

        # automatically load first case in directory
        self.AutoLoad = False

        # autoclip to window/level
        self.WLClip = False

        # default BLAST slider values
        self.T1default = 0.
        self.T2default = 0.
        self.BCdefault = (2.,2.)
        self.thresholddefaults = {'t12':0,'flair':0,'bc':3}

        # < 1 overlay by alpha compositing, == 1 replacement
        self.OverlayIntensity = 1.0

        # image panel size
        self.PanelSize = 3 # inches at 100 dpi

        # default size for image array
        self.ImageDim = (155,240,240)

        # max/min z score in parameter space.
        self.zmin = 0
        self.zmax = 12
        self.zinc = 0.1
        self.cbvmin = 0
        self.cbvmax = 2000
        self.cbvinc = 100

        # default 'z-score' or 'CBV' overlay
        self.OverlayType = 'z'

        # default BLAST overlay, contour or area
        self.BlastOverlayType = 1
        self.MaskType = 'ET'

        # colormaps
        self.OverlayCmap = {'z':'viridis','cbv':'viridis','tempo':'tempo'}
        cmap_tempo = ListedColormap(np.array([[0 ,.5, 0, 1],[0,0,0,1],[0, 1, 0, 1]]))
        cm.register_cmap(name='tempo',cmap=cmap_tempo)

        # default viewer type
        self.DefaultViewer = 'SAM'
