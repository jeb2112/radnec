
from datetime import datetime
import numpy as np
import os

class UIActions(object):
    """Class to control common interactions with UI"""

    def __init__(self, uiobj):
        self.ui = uiobj
        self.gainvals = []

    ### General User Interface interaction methods ###

    # Save the current case 
    def saveCase(self, ind=None):
        return None

    ## Create file system folders for sequence and parameter files
    def makeFolders(self):
        return None

    def SetStatusMessage(self):
        return None