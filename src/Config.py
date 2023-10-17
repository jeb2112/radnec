import os, sys

class Config(object):
    def __init__(self):
        
        ### Paths that tend to change based on local install

        if os.name == 'nt':
            self.InstallPath = os.path.join('C:\\','Users','Chris Heyn Lab','src','blast')
        else:
            self.InstallPath = os.path.join('/home','src','blast')

        # Path where R&D user interface built in Python is located
        self.UIPath = os.path.join(self.InstallPath, 'src')

        # Path for UI based resources (images, icons, etc)
        self.UIResourcesPath = os.path.join(self.InstallPath, 'resources')

        # Path to log files
        self.LogFilePath = os.path.join(os.path.expanduser('~'), 'Documents', 'Blast', 'Log')

        # Path to data files
        self.DataFilePath = os.path.join(os.path.expanduser('~'), 'Documents', 'Blast', 'Data')

        # Path to dataset images
        if os.name == 'nt':
            self.UIdatadir = os.path.join('D:\\','BLAST','test')
        elif os.name == 'posix':
            self.UIdatadir = '/media/jbishop/WD4/brainmets/blast/'
        self.UIdataroot = 'BraTS2021_'

