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


        # Image paths

        # UIdatadir - intended as a path to fixed resources that wouldn't generally change during  
        # use of the viewer. currently only the mni talairach atlas uses this path.
        # UIlocaldir - a convenient default starting point for input data directory in the files
        # and directory dialog. Once the dialog is used to select some different directory, 
        # this hard-coded value is no longer used. 
        if os.name == 'nt':
            self.UIdatadir = os.path.join(os.path.expanduser('~'),'data')
            self.UIlocaldir = os.path.join(os.path.expanduser('~'),'data','radnec_sam')
        elif os.name == 'posix':
            # brats dataset
            # self.UIdatadir = '/media/jbishop/WD4/brainmets/sunnybrook/metastases'
            # self.UIlocaldir = '/media/jbishop/WD4/brainmets/sunnybrook/metastases/BraTS_2024'
            # self.UIawsdir = '/home/ec2-user'
            # dicoms
            self.UIdatadir = '/media/jbishop/WD4/brainmets/sunnybrook/radnec2'
            self.UIlocaldir = '/media/jbishop/WD4/brainmets/sunnybrook/radnec2/dicom2nifti'

        # the value here is no longer being using, instead the default is assigned in CreateCaseFrame
        self.UIdataroot = 'BraTS2021_'

        # a regex for root filename of ground truth masks in the BraTS context
        # but could be generalized
        self.UIgroundtruth = re.compile('BraTS-MET-[0-9]{5}-000-seg.nii.gz')

        # pytorch env for SAM and nnUNet, hard-coded here according to whatever machine
        # is being used

        # self.UIpytorch = 'pytorch_sam'
        self.UIpytorch = 'pytorch118_310'

        # sam model
        # base SAM
        # self.SAMModel = 'sam_vit_b_01ec64.pth'
        self.SAMModel = 'sam_vit_h_4b8939.pth'
        # SAM fine-tuned on BraTS2024 MET
        # self.SAMModel = 'sam_brats2024_10sep24_9000_50epoch.pth'
        self.SAMModelAWS = 'best_base_AdamW_lr=7e-06_wd=0.0002_bs=8_mp=fp16_bbox_0_3_loss=dice_20set.pth'
        # improve 3d SAM with orthogonal segmentations
        self.SAMortho = True
        # combine orthogonal segmentation as raw probability otherwise AND the binary final mask
        self.SAMRawCombine = False
        # auto-update the SAM 2d in current slice during assembly of the BLAST ROI
        self.SAM2dauto = False

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

        # default channel. 
        self.DefaultChannel = 'flair'
        self.DefaultLayer = 'WT'
        self.DefaultBlastLayer = 'T2 hyper'

        # use aws cloud
        self.AWS = False

        # load an optional mask (eg BraTS segmentation)
        self.UseBraTSMask = False

        # optional colorization for SAM grayscale
        self.Colorize = False