import os, sys
import logging

class Config(object):
    def __init__(self):
        
        ### Paths that tend to change based on local install

        if os.name == 'nt':
            self.InstallPath = os.path.join('C:\\','Users','chint','src','blast')
        else:
            self.InstallPath = os.path.join('/home','src','blast')

        # Path where R&D user interface built in Python is located
        self.UIPath = os.path.join(self.InstallPath, 'src')

        # Path for UI based resources (images, icons, etc)
        # use this hack for relative path for installation of a wheel distro into a virtual env
        self.UIResourcesPath = os.path.join(os.path.dirname(__file__), '..','resources')

        # Path to log files
        self.logger = logging.getLogger('blast')
        self.logger.setLevel(logging.DEBUG)
        self.LogFileName = 'BlastUI.log'
        if self.LogFileName is not None:
            self.LogFilePath = os.path.join(os.path.expanduser('~'), 'Documents', 'Blast', 'Log')
            if not os.path.exists(self.LogFilePath):
                os.makedirs(self.LogFilePath)
            filehandler = logging.FileHandler(os.path.join(self.LogFilePath,self.LogFileName),mode='w')
            self.logger.addHandler(filehandler)

        # Path to data files
        self.DataFilePath = os.path.join(os.path.expanduser('~'), 'Documents', 'Blast', 'Data')
        if not os.path.exists(self.LogFilePath):
            os.makedirs(self.LogFilePath)

        # Path to dataset images
        if os.name == 'nt':
            self.UIdatadir = os.path.join('C:\\','Users','chint','Data')
        elif os.name == 'posix':
            # self.UIdatadir = '/media/jbishop/WD4/brainmets/raw/BraTS2021'
            self.UIdatadir = '/home/jbishop/Data/BraTS2021'
        self.UIdataroot = 'BraTS2021_'

        # automatically load first case in directory
        self.AutoLoad = False

        # autoclip to window/level
        self.WLClip = False

        # < 1 overlay by alpha compositing, == 1 replacement
        self.OverlayIntensity = 0.6

        # default slider values
        self.T1default = 0.
        self.T2default = 0.
        self.BCdefault = 3.

        # image panel size
        self.PanelSize = 4 # inches at 100 dpi
        self.dpi = 100

        # default size for image array
        self.ImageDim = (155,240,240)