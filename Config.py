import os, sys

class Config(object):
    def __init__(self):
        
        ### Paths that tend to change based on local install

        if os.name == 'nt':
            self.InstallPath = os.path.join('C:\\','Program Files (x86)','Blast')
        else:
            self.InstallPath = os.path.join('/home','src')

        # Path where R&D user interface built in Python is located
        self.UIPath = os.path.join(self.InstallPath, 'blast')

        # Path for UI based resources (images, icons, etc)
        self.UIResourcesPath = os.path.join(self.UIPath, 'resources')

        # Path where temporary image files used in UI are saved
        self.UIWorkingFilesPath = os.path.join(self.UIPath, 'temp')

        # Path to log files
        self.LogFilePath = os.path.join(os.path.expanduser('~'), 'Documents', 'Blast', 'Log')

        # Path to data files
        self.DataFilePath = os.path.join(os.path.expanduser('~'), 'Documents', 'Blast', 'Data')

        # Path to dataset images
        self.UIdatadir = '/media/jbishop/WD4/brainmets/blast/'
        self.UIdataroot = 'BraTS2021_'

