import os
import tkinter as tk
import tk_async_execute as tae
import sys
import logging

from src import BlastGui
from src import Config

from twisted.internet import tksupport, reactor, protocol

class Gui():

    def __init__(self, optionsFlag = 0, debug=False):
        try:
            self.config = Config.Config()
            
            # said to help reduce ibus-daemon burning cpu and scrambling keyboard in put in conjunction wtih tkinter
            os.environ['XMODIFIERS'] = "@im=none"

            self.root = tk.Tk()
            # self.root.rowconfigure(0,minsize=100,weight=1)
            # self.root.columnconfigure(0,minsize=550,weight=1)
            self.root.rowconfigure(0,weight=1)
            self.root.columnconfigure(0,weight=1)
            self.root.protocol("WM_DELETE_WINDOW", self.windowCloseHandler)
            if (sys.platform.startswith('win')):
                iconfile = os.path.join(self.config.UIResourcesPath,'sunnybrook.ico')
                self.root.iconbitmap(default=iconfile)
            else:
                iconfile = os.path.join(self.config.UIResourcesPath,'sunnybrook.png')
                if False:
                    self.root.call('wm','iconphoto',self.root._w,tk.PhotoImage(file=iconfile))
                else:
                    self.root.iconphoto(False,tk.PhotoImage(file=iconfile))
            self.UI = BlastGui.BlastGui(self.root, optionsFlag, self.config, debug=debug)

            # start a loop for asyncio. might not need this here though.
            tae.start()
            # tiwsted isn't compatible with the tae loop so just use mainloop()
            if False:
                tksupport.install(self.root)
                reactor.run()
            else:
                self.root.mainloop()
        except KeyError as e:
            self.config.logger.error("{}: {}".format(e.args[0], sys.exc_info()[0]))
            print("{}: {}".format(e.args[0], sys.exc_info()[0]))
        except Exception as e:
            self.config.logger.error("{}: {}".format(e.args[0], sys.exc_info()[0]))
            print("{}: {}".format(e.args[0], sys.exc_info()[0]))
            # on windows 11, all errors get handled by this top-level try/except, so need to print
            # stacktrace explicitly. formerly on windows10 and linux, most errors were not
            # being handled by this try/except, so the stack trace was printing directly
            # and this handler was rarely used
            if hasattr(e,'characters_written'):
                print(e.characters_written)
        else:
            print("Exit")

    def windowCloseHandler(self):
        if False:
            reactor.stop()
        tae.stop()
        self.root.quit()
        self.root.destroy()
