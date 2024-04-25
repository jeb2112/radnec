import os, sys
import logging

class Config(object):
    def __init__(self):
        
        ### Paths that tend to change based on local install

        if os.name == 'nt':
            self.InstallPath = os.path.join('C:\\','Users','chint','src','radnec')
        else:
            self.InstallPath = os.path.join('/home','src','radnec')

        # Path where R&D user interface built in Python is located
        self.UIPath = os.path.join(self.InstallPath, 'src')

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
            self.UIdatadir = os.path.join(os.path.expanduser('~'),'Data')
        elif os.name == 'posix':
            self.UIdatadir = '/media/jbishop/WD4/brainmets/sunnybrook/radnec'
            self.UIlocaldir = '/media/jbishop/WD4/brainmets/sunnybrook/radnec/dicom2nifti'
            # self.UIdatadir = os.path.join(os.path.expanduser('~'),'Data','radnec')
        self.UIdataroot = 'BraTS2021_'

        # automatically load first case in directory
        self.AutoLoad = False

        # autoclip to window/level
        self.WLClip = False

        # < 1 overlay by alpha compositing, == 1 replacement
        self.OverlayIntensity = 0.6

        # image panel size
        self.PanelSize = 4 # inches at 100 dpi

        # default size for image array
        self.ImageDim = (155,240,240)

        # max/min z score in parameter space.
        self.maxZ = 4

        # default 'contour' or 'mask' overlay
        self.OverlayType = 1