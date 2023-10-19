import os
from tkinter import *
import sys
import logging as Log

import BlastGui
import Config

from twisted.internet import tksupport, reactor, protocol

class Gui():

    def __init__(self, optionsFlag = 0, debug=False):
        try:
            self.root = Tk()
            self.root.rowconfigure(0,minsize=600,weight=1)
            self.root.columnconfigure(0,minsize=600,weight=1)
            self.root.protocol("WM_DELETE_WINDOW", self.windowCloseHandler)
            self.config = Config.Config()
            if (sys.platform.startswith('win')):
                iconfile = os.path.join(self.config.UIResourcesPath,'sunnybrook.ico')
                self.root.iconbitmap(default=iconfile)
            else:
                iconfile = os.path.join(self.config.UIResourcesPath,'sunnybrook.png')
                self.root.call('wm','iconphoto',self.root._w,PhotoImage(file=iconfile))
            self.UI = BlastGui.BlastGui(self.root, optionsFlag, debug=debug)

            tksupport.install(self.root)
            reactor.run()
        except Exception as e:
            Log.error(logname='BlastUI', group=self.__class__.__name__).exception("{}: {}".format(e, sys.exc_info()[0]))
        else:
            print("Exit")

    def windowCloseHandler(self):
        reactor.stop()
        self.root.destroy()
